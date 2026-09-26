"""Download pinned, small public benchmark files for real-source adapter checks.

Full CLERC, KILT, and HotpotQA corpora are deliberately excluded. Their sizes
exceed the development machine's free disk space and belong on the later VM.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MIRAGE_REV = "392943af99cd94cafd50a0de2e7fca24bbf65494"
TEXTBOOKS_REV = "9c72838920a1323ffa867467d3f7aa7b36b0f994"
CLERC_REV = "ef042f8ab436f78704f17faa0a866d1b2b862f6f"
QASPER_REV = "16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9"
HOTPOT_MIRROR_REV = "7e54db4656209750ff487f6fdf8e39a66dba136b"
HOTPOT_MIRROR_SHA256 = "7fcef81c55cf6a3583e001a3ca54f0ca6bff23528fddef838d20251f25eea9a1"
EXPECTED_SHA256 = json.loads((Path(__file__).resolve().parents[1] / "configs/rag/small_source_hashes.json").read_text())


def _request(url):
    return urllib.request.Request(url, headers={"User-Agent": "jev-rag-source-audit/1.0"})


DATASETS = {"mirage", "clerc", "hotpotqa", "qasper", "kilt_fever"}


def files_to_fetch(datasets=None):
    selected = set(datasets or DATASETS)
    if not selected or selected - DATASETS:
        raise ValueError(f"Unknown dataset selection: {sorted(selected - DATASETS)}")
    files = [
        ("mirage/benchmark.json", f"https://raw.githubusercontent.com/gzxiong/MIRAGE/{MIRAGE_REV}/benchmark.json"),
        ("clerc/queries/test.single-removed.direct.tsv", f"https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/{CLERC_REV}/queries/test.single-removed.direct.tsv"),
        ("clerc/qrels/qrels-passage.test.direct.tsv", f"https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/{CLERC_REV}/qrels/qrels-passage.test.direct.tsv"),
        ("hotpotqa/hotpot_dev_fullwiki_v1.json", f"https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/{HOTPOT_MIRROR_REV}/hotpot_dev_fullwiki_v1.json"),
        ("kilt_fever/fever-dev-kilt.jsonl", "https://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl"),
    ]
    files = [(relative, url) for relative, url in files if relative.split("/", 1)[0] in selected]
    if "qasper" in selected:
        for split in ("train", "validation", "test"):
            relative = f"qasper/qasper/qasper-{split}.parquet"
            url = f"https://huggingface.co/datasets/allenai/qasper/resolve/{QASPER_REV}/qasper/qasper-{split}.parquet"
            files.append((relative, url))
    if "mirage" in selected:
        manifest_url = f"https://huggingface.co/api/datasets/MedRAG/textbooks/tree/{TEXTBOOKS_REV}?recursive=true"
        with urllib.request.urlopen(_request(manifest_url), timeout=30) as response:
            for entry in json.load(response):
                if entry.get("path", "").startswith("chunk/") and entry["path"].endswith(".jsonl"):
                    relative = f"mirage/medrag/{entry['path']}"
                    url = f"https://huggingface.co/datasets/MedRAG/textbooks/resolve/{TEXTBOOKS_REV}/{entry['path']}"
                    files.append((relative, url))
    expected = {relative for relative in EXPECTED_SHA256 if relative.split("/", 1)[0] in selected}
    if {relative for relative, _ in files} != expected:
        raise ValueError("Pinned source file list changed")
    return files


def _sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(root, relative, url):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        checksum = _sha256(target)
        if checksum != EXPECTED_SHA256[relative]:
            raise ValueError(f"Existing source checksum mismatch: {relative}")
        return {"path": relative, "url": url, "bytes": target.stat().st_size, "sha256": checksum, "reused": True}
    pending = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as response, pending.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            checksum = _sha256(pending)
            if checksum != EXPECTED_SHA256[relative]:
                raise ValueError(f"Downloaded source checksum mismatch: {relative}")
            if relative == "hotpotqa/hotpot_dev_fullwiki_v1.json" and checksum != HOTPOT_MIRROR_SHA256:
                raise ValueError("HotpotQA mirror checksum differs from its published SHA256")
            pending.replace(target)
            return {"path": relative, "url": url, "bytes": target.stat().st_size, "sha256": checksum, "reused": False}
        except Exception:
            pending.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/sources"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    files = files_to_fetch(args.datasets)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, args.root, relative, url): relative for relative, url in files}
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(f"{result['path']}: {result['bytes']} bytes", flush=True)
    manifest = {"source_revisions": {"mirage": MIRAGE_REV, "medrag_textbooks": TEXTBOOKS_REV,
                                      "clerc": CLERC_REV, "qasper_parquet": QASPER_REV,
                                      "hotpotqa_dev_mirror": HOTPOT_MIRROR_REV},
                "files": sorted(records, key=lambda item: item["path"])}
    manifest_name = "small_source_manifest.json" if args.datasets is None else f"small_source_manifest_{'_'.join(sorted(args.datasets))}.json"
    manifest_path = args.root / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
