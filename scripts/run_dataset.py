"""One-command, resumable, ranking-only run for one benchmark dataset.

Planning is the default. --execute downloads sources, builds indexes, runs the
15 retrieval/reranker cells, and writes a report. No generator or LLM judge is
called. Large corpora require a VM with a separate data volume.
"""

import argparse
import bz2
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/rag/experiment.yaml"
REVISIONS = REPO / "configs/rag/model_revisions.json"
DATASETS = ("qasper", "clerc", "hotpotqa", "kilt_fever")
REPORTING_COUNTS = {"qasper": 4151, "clerc": 1200, "hotpotqa": 7405, "kilt_fever": 10444}
LARGE = {
    "clerc": (
        "clerc/collection/collection.passage.tsv.gz",
        "https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/ef042f8ab436f78704f17faa0a866d1b2b862f6f/collection/collection.passage.tsv.gz",
        9_190_980_544,
        None,
    ),
    "hotpotqa": (
        "hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        1_553_565_403,
        "01edf64cd120ecc03a2745352779514c",
    ),
    "kilt_fever": (
        "kilt_fever/kilt_knowledgesource.json",
        "https://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json",
        37_318_876_722,
        None,
    ),
}


def config_scalar(name):
    match = re.search(rf"^{re.escape(name)}:\s*\"?([^\"#\n]+)", CONFIG.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name} in {CONFIG}")
    return match.group(1).strip()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def file_hash(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_large(root, dataset, log):
    relative, url, expected_size, expected_md5 = LARGE[dataset]
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = root / dataset / "large_source_manifest.json"
    partial = target.with_name(target.name + ".part")
    if target.exists() and target.stat().st_size != expected_size:
        raise ValueError(f"Existing {relative} has the wrong byte length; remove it or restore a verified copy")
    if not target.exists():
        for attempt in range(8):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                raise ValueError(f"Partial {relative} exceeds expected byte length")
            if offset == expected_size:
                break
            headers = {"User-Agent": "jev-rag-source-audit/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    if offset:
                        content_range = response.headers.get("Content-Range", "")
                        if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError(f"Server did not honor resume offset {offset} for {relative}")
                    mode = "ab" if offset else "wb"
                    with partial.open(mode) as output:
                        while block := response.read(8 * 1024 * 1024):
                            output.write(block)
                if partial.stat().st_size == expected_size:
                    break
                log(f"WARNING incomplete download {relative}: {partial.stat().st_size}/{expected_size} bytes")
            except (OSError, TimeoutError) as exc:
                log(f"WARNING download retry {attempt + 1}/8 for {relative}: {type(exc).__name__}: {exc}")
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 30))
        if not partial.exists() or partial.stat().st_size != expected_size:
            raise ValueError(f"Download incomplete for {relative}")
        if expected_md5 and file_hash(partial, "md5") != expected_md5:
            raise ValueError(f"Published MD5 mismatch for {relative}")
        partial.replace(target)
    if expected_md5 and file_hash(target, "md5") != expected_md5:
        raise ValueError(f"Published MD5 mismatch for {relative}")
    checksum = file_hash(target)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("url") != url or previous.get("sha256") != checksum:
            raise ValueError(f"Large source changed since its previous manifest: {relative}")
    record = {"url": url, "path": relative, "bytes": target.stat().st_size,
              "sha256": checksum, "published_md5": expected_md5}
    json_write(manifest_path, record)
    log(f"Verified {relative}: {record['bytes']} bytes, SHA-256 {record['sha256']}")
    return target


def hotpot_gold_titles(dev_path):
    titles = {}
    for row in json.loads(dev_path.read_text()):
        for title, _ in row.get("supporting_facts", []):
            folded = title.casefold()
            if folded in titles and titles[folded] != title:
                raise ValueError(f"Ambiguous HotpotQA gold title casing: {title}")
            titles[folded] = title
    return titles


def convert_hotpot_archive(archive, dev_path, output, log):
    """Stream the official intro archive into sentence-preserving JSONL."""
    if output.exists():
        if output.stat().st_size == 0:
            raise ValueError(f"Empty HotpotQA wiki output: {output}")
        log(f"Reusing {output}")
        return
    titles = hotpot_gold_titles(dev_path)
    pending = output.with_name(output.name + ".part")
    pending.unlink(missing_ok=True)
    pages = 0
    found_gold = set()
    try:
        with pending.open("w", encoding="utf-8") as writer, tarfile.open(archive, "r|bz2") as members:
            for member in members:
                if not member.isfile():
                    continue
                source = members.extractfile(member)
                if source is None:
                    continue
                with (bz2.BZ2File(source) if member.name.endswith(".bz2") else source) as stream:
                    for raw in stream:
                        if not raw.strip():
                            continue
                        article = json.loads(raw)
                        title = article["title"]
                        paragraphs = article["text"]
                        if paragraphs and isinstance(paragraphs[0], str):
                            paragraphs = [paragraphs]
                        intro = next((paragraph for paragraph in paragraphs
                                      if len("".join(paragraph).strip()) > 50), None)
                        if intro is None:
                            continue
                        title = titles.get(title.casefold(), title)
                        if title.casefold() in titles:
                            found_gold.add(title.casefold())
                        writer.write(json.dumps({"title": title, "sentences": intro}, ensure_ascii=False) + "\n")
                        pages += 1
                        if pages % 500_000 == 0:
                            log(f"Converted {pages} HotpotQA wiki pages")
        missing = set(titles) - found_gold
        if missing:
            raise ValueError(f"HotpotQA wiki lacks {len(missing)} gold titles; examples: {sorted(missing)[:5]}")
        pending.replace(output)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    log(f"Converted {pages} HotpotQA wiki pages; all {len(titles)} gold titles found")


def forecast(dataset, limit, price, existing_spend):
    depth = int(config_scalar("candidate_depth"))
    count = REPORTING_COUNTS[dataset] if limit == "all" else min(int(limit), REPORTING_COUNTS[dataset])
    calls = count * 3 * depth
    upper = calls * (Decimal(32_000) * Decimal(str(price["input_usd_per_million"]))
                     + Decimal(64) * Decimal(str(price["output_usd_per_million"]))) / Decimal(1_000_000)
    return {"dataset": dataset, "reporting_queries": count, "cells": 15,
            "max_jev_requests_before_cache": calls, "conservative_api_upper_usd": str(upper),
            "minimum_hours_at_configured_rpm": round(calls / int(price["rpm"]) / 60, 2),
            "already_spent_usd": str(existing_spend), "budget_usd": config_scalar("budget_usd")}


def configured_secret(name):
    if os.environ.get(name):
        return True
    env_file = REPO / ".env"
    if not env_file.exists():
        return False
    for line in env_file.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name and value.strip().strip("\"'"):
            return True
    return False


def free_bytes(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise FileNotFoundError(path)
        path = path.parent
    return shutil.disk_usage(path).free


def available_memory_bytes():
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", path.read_text(), re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def summary_markdown(report, dataset, split, limit, log_path):
    lines = [f"# {dataset} ranking-only summary", "", f"Split: {split}; requested limit: {limit}.",
             f"Log: `{log_path}`.", "", "OpenRouter generation and LLM judging were disabled.", "",
             "| Retriever | Reranker | Queries | Failed | nDCG@10 | Recall@10 | MRR@10 |", "|---|---|---:|---:|---:|---:|---:|"]
    for key, cell in sorted(report["quality"].items()):
        _, retriever, reranker = key.split("/")
        metrics = cell["metrics"]
        def metric(name):
            value = metrics.get(f"ranking.{name}", {}).get("mean")
            return f"{value:.4f}" if value is not None else "n/a"
        lines.append(f"| {retriever} | {reranker} | {cell['attempted']} | {cell['failed']} | "
                     f"{metric('ndcg@10')} | {metric('recall@10')} | {metric('mrr@10')} |")
    lines += ["", f"Global API ledger attempts: {report['budget']['attempts']}; cumulative ledger cost: ${report['budget']['cost_usd']}.",
              "", "This is a ranking comparison on the selected queries. It does not measure final answer quality.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--limit", default="10", help="Reporting queries per cell, or 'all' (default: 10)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--volume", type=Path, help="Store source files and runs on this VM volume")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--limits-json", type=Path, default=REPO / "data/limits.json")
    parser.add_argument("--execute", action="store_true", help="Download, index, and run paid Jev ranking")
    args = parser.parse_args()
    if args.limit != "all" and (not args.limit.isdecimal() or int(args.limit) < 1):
        parser.error("--limit must be a positive integer or 'all'")
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    volume = args.volume.resolve() if args.volume else None
    if volume is not None and not volume.is_dir():
        parser.error(f"--volume must be an existing directory: {volume}")
    source_root = (args.source_root or (volume / "sources" if volume else REPO / "data/sources")).resolve()
    run_root = (args.run_root or (volume / "runs" if volume else REPO / "runs")).resolve()
    budget = Decimal(config_scalar("budget_usd"))
    run_id = config_scalar("experiment_id")
    ledger = run_root / run_id / "budget.sqlite"
    spent = Decimal(0)
    past_attempts = 0
    if ledger.exists():
        import sqlite3

        with sqlite3.connect(ledger) as db:
            past_attempts, cost = db.execute("SELECT attempts,cost_usd FROM budget_state WHERE id=1").fetchone()
            spent = Decimal(str(cost))
    limits = json.loads(args.limits_json.read_text()) if args.limits_json.exists() else None
    price = (limits or {}).get("typesafe")
    required = {"rpm", "tpm", "inflight", "token_budget", "request_budget",
                "input_usd_per_million", "output_usd_per_million"}
    try:
        valid_price = isinstance(price, dict) and required <= set(price) and all(
            int(price[field]) > 0 for field in ("rpm", "tpm", "inflight", "token_budget", "request_budget")
        ) and all(Decimal(str(price[field])) >= 0 for field in ("input_usd_per_million", "output_usd_per_million"))
    except (ValueError, TypeError):
        valid_price = False
    plan = forecast(args.dataset, args.limit, price, spent) if valid_price else {
        "dataset": args.dataset, "reporting_queries": REPORTING_COUNTS[args.dataset] if args.limit == "all" else min(int(args.limit), REPORTING_COUNTS[args.dataset]),
        "cells": 15, "budget_usd": str(budget), "missing_or_invalid_limits": str(args.limits_json),
    }
    if not args.execute:
        print(json.dumps({**plan, "execution": "plan_only", "openrouter_calls": 0,
                          "source_root": str(source_root), "run_root": str(run_root)}, indent=2))
        return
    if not valid_price:
        parser.error(f"Account-specific TypeSafe rate, request, token, and price limits are required in {args.limits_json}")
    if not configured_secret("TYPESAFE_API_KEY") or not configured_secret("JEV_MODEL"):
        parser.error("TYPESAFE_API_KEY and JEV_MODEL must be set in the environment or ignored .env")
    if spent + Decimal(plan["conservative_api_upper_usd"]) > budget:
        parser.error(f"Conservative Jev upper bound ${plan['conservative_api_upper_usd']} exceeds remaining ${budget - spent}; use a smaller --limit or revise the budget after measuring tokens")
    if past_attempts + plan["max_jev_requests_before_cache"] > int(price["request_budget"]):
        parser.error("Conservative Jev request count exceeds the configured request budget; use a smaller --limit or increase the verified account cap")
    if shutil.which("uv") is None:
        parser.error("uv is required; install it from https://docs.astral.sh/uv/getting-started/installation/")
    minimum = 8 * 1024**3 if args.dataset == "qasper" else 250 * 1024**3
    for label, path in (("source", source_root), ("run", run_root)):
        free = free_bytes(path)
        if free < minimum:
            parser.error(f"Need at least {minimum / 1024**3:.0f} GiB free on the {label} volume for {args.dataset}; found {free / 1024**3:.1f} GiB")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = run_root / run_id / "launcher" / args.dataset / timestamp
    output_root.mkdir(parents=True, exist_ok=False)
    log_path = output_root / "run.log"
    state_path = output_root / "status.json"
    state = {"dataset": args.dataset, "split": "reporting", "limit": args.limit,
             "status": "running", "plan": plan, "stages": [], "warnings": [], "errors": [],
             "log": str(log_path), "started_at_utc": timestamp,
             "config_sha256": file_hash(CONFIG), "revisions_sha256": file_hash(REVISIONS)}

    def log(message):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")
        if re.search(r"\b(?:WARNING|WARN)\b", message, re.IGNORECASE):
            state["warnings"].append(message[:1000])
        if re.search(r"\bERROR\b|Traceback", message, re.IGNORECASE):
            state["errors"].append(message[:1000])

    def command(stage, argv):
        log(f"START {stage}: {' '.join(map(str, argv))}")
        started = time.monotonic()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with subprocess.Popen(list(map(str, argv)), cwd=REPO, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log(line.rstrip("\n"))
            code = process.wait()
        state["stages"].append({"stage": stage, "exit_code": code,
                                "elapsed_seconds": round(time.monotonic() - started, 3)})
        json_write(state_path, state)
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {log_path}")
        log(f"COMPLETE {stage}")

    try:
        json_write(state_path, state)
        command("dependencies", ["uv", "sync", "--locked", "--extra", "local", "--extra", "observability"])
        executable = REPO / ".venv/bin/jev-eval"
        command("validate", [executable, "rag", "validate", CONFIG])
        command("small_sources", [sys.executable, REPO / "scripts/fetch_small_sources.py",
                                  "--root", source_root, "--datasets", args.dataset])
        if args.dataset in LARGE:
            log(f"START large source for {args.dataset}")
            archive = download_large(source_root, args.dataset, log)
            if args.dataset == "hotpotqa":
                convert_hotpot_archive(archive, source_root / "hotpotqa/hotpot_dev_fullwiki_v1.json",
                                       source_root / "hotpotqa/wiki.jsonl", log)
            log(f"COMPLETE large source for {args.dataset}")
        prepared = run_root / run_id / args.dataset / "source_manifest.json"
        if not prepared.exists():
            if prepared.parent.exists():
                raise ValueError(f"Incomplete prepared dataset directory: {prepared.parent}")
            command("prepare", [executable, "rag", "prepare", CONFIG, args.dataset,
                                "--source-root", source_root, "--run-root", run_root])
        else:
            log(f"Reusing prepared source: {prepared}")
        dense_root = run_root / run_id / args.dataset
        manifest = json.loads(prepared.read_text())
        for relative, checksum in manifest["source_files_sha256"].items():
            path = source_root / args.dataset / relative
            if not path.is_file() or file_hash(path) != checksum:
                raise ValueError(f"Prepared source changed or is missing: {path}")
        chunks = int(manifest["chunks"])
        # This implementation holds both float32 vectors and an exact FAISS copy.
        dense_bytes = chunks * 1024 * 4
        if not (dense_root / "dense.npy").exists():
            ram = available_memory_bytes()
            required_ram = int(dense_bytes * 1.5) + 4 * 1024**3
            if ram is not None and ram < required_ram:
                raise MemoryError(f"Exact dense indexing needs roughly {required_ram / 1024**3:.1f} GiB available RAM for {chunks} chunks; found {ram / 1024**3:.1f} GiB")
            if free_bytes(run_root) < int(dense_bytes * 2.4):
                raise OSError(f"Dense vectors plus exact FAISS need at least {dense_bytes * 2.4 / 1024**3:.1f} GiB free in the run volume")
        vector, meta = dense_root / "dense.npy", dense_root / "dense.meta.json"
        if vector.exists() != meta.exists():
            raise ValueError(f"Incomplete dense index under {dense_root}")
        if not vector.exists():
            command("dense_index", [executable, "rag", "index-dense", CONFIG, args.dataset,
                                    "--run-root", run_root, "--revisions-json", REVISIONS,
                                    "--batch-size", args.batch_size])
        else:
            log(f"Reusing dense index: {vector}")
        queries = json.loads((dense_root / "queries.json").read_text())
        actual = sum(item["split"] == "reporting" for item in queries)
        if actual != REPORTING_COUNTS[args.dataset]:
            raise ValueError(f"Reporting query count changed: expected {REPORTING_COUNTS[args.dataset]}, found {actual}")
        selected = actual if args.limit == "all" else min(int(args.limit), actual)
        matrix_path = output_root / "matrix.json"
        command("ranking_matrix", [executable, "rag", "matrix", CONFIG, "--datasets", args.dataset,
                                   "--run-root", run_root, "--split", "reporting", "--limit", selected,
                                   "--workers", args.workers, "--revisions-json", REVISIONS,
                                   "--limits-json", args.limits_json, "--output-json", matrix_path, "--execute"])
        matrix = json.loads(matrix_path.read_text())
        if matrix["status"] != "complete" or len(matrix["results"]) != 15:
            raise ValueError("Ranking matrix did not complete all 15 cells")
        results = [entry["path"] for entry in matrix["results"]]
        failed_rows = []
        for path in results:
            for row in json.loads(Path(path).read_text()):
                if row.get("status") == "failed":
                    failed_rows.append({"result_path": path, "query_id": row.get("query_id"),
                                        "retriever": row.get("retriever"), "reranker": row.get("reranker"),
                                        "error_type": row.get("error_type"), "error": row.get("error")})
        failures_path = output_root / "failures.json"
        json_write(failures_path, failed_rows)
        for row in failed_rows[:20]:
            log(f"ERROR {row['retriever']}/{row['reranker']}/{row['query_id']}: "
                f"{row['error_type']}: {row['error']}")
        if len(failed_rows) > 20:
            log(f"WARNING {len(failed_rows) - 20} additional query failures in {failures_path}")
        report_path = output_root / "report.json"
        command("report", [executable, "rag", "report", CONFIG, *results,
                           "--run-root", run_root, "--output", report_path])
        command("traces", [executable, "rag", "export-traces", CONFIG, "--run-root", run_root,
                           "--output", output_root / "traces"])
        report = json.loads(report_path.read_text())
        summary = output_root / "summary.md"
        summary.write_text(summary_markdown(report, args.dataset, "reporting", selected, log_path))
        failures = len(failed_rows)
        if failures:
            log(f"WARNING {failures} query/cell failures; inspect result JSON and run log")
        state.update({"status": "complete_with_failures" if failures else "complete",
                      "failed_query_cells": failures, "matrix": str(matrix_path), "report": str(report_path),
                      "summary": str(summary), "failures_json": str(failures_path), "result_paths": results,
                      "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        json_write(state_path, state)
        log(f"COMPLETE dataset run; summary: {summary}")
        if failures:
            raise SystemExit(2)
    except Exception as exc:
        state.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}",
                      "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        json_write(state_path, state)
        log(f"ERROR {state['failure']}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        state.update({"status": "interrupted", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        json_write(state_path, state)
        log("WARNING run interrupted; completed source and stage artifacts remain for resume")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
