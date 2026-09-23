import time
import zipfile
from pathlib import Path

import httpx
import numpy as np

from .config import Settings, load_config
from .contracts import Pool
from .data import ingest_beir, iter_documents
from .io import digest, file_digest, read_json, write_json
from .metrics import evaluate_query, export_trec, rank
from .models import DenseModel, LocalReranker, reranker_text
from .retrieve import make_pool, rrf
from .rubrics import TASKS, build_request, rubric
from .selection import select


def run_dir(config):
    return Path("runs") / config.experiment_id


def download_beir(name):
    if name not in {"scifact", "nfcorpus", "fiqa", "arguana"}:
        raise ValueError(
            "Automatic download is restricted to the four core BEIR releases"
        )
    dest = Path("data") / name
    if (dest / "corpus.jsonl").exists():
        return dest
    archive = Path("data") / f"{name}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    )
    partial = archive.with_suffix(".zip.part")
    last_error = None
    for attempt in range(3):
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=120, trust_env=True
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            partial.replace(archive)
            break
        except (httpx.HTTPError, OSError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(
            f"Could not download {name} from {url}. Check network/DNS or use "
            "prepare --source PATH with a downloaded BEIR directory."
        ) from last_error
    with zipfile.ZipFile(archive) as files:
        for name in files.namelist():
            target = (archive.parent / name).resolve()
            if not target.is_relative_to(archive.parent.resolve()):
                raise ValueError("Unsafe archive path")
        files.extractall(archive.parent)
    return dest


def prepare(config_path, source=None):
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    load_dotenv()
    config = load_config(config_path)
    dest = run_dir(config)
    source = Path(source) if source else download_beir(config.dataset)
    if not (dest / "dataset_manifest.json").exists():
        manifest = ingest_beir(
            source, dest, config.split, config.sample_queries, config.seed
        )
        if config.dataset == "scifact" and manifest["corpus_count"] != 5183:
            raise ValueError(
                "Unexpected SciFact corpus count; inspect dataset revision"
            )
    manifest = read_json(dest / "dataset_manifest.json")
    if config.corpus_revision and manifest["revision"] != config.corpus_revision:
        raise ValueError("Requested corpus revision does not match source files")
    if (dest / "resolved_config.json").exists():
        existing = read_json(dest / "input_config.json")
        if existing != config.model_dump():
            raise ValueError("Input config changed; use a new experiment_id")
        return dest
    api = HfApi()
    models = {}
    for name in (config.dense_model, config.reranker_model, "BAAI/bge-reranker-v2-m3"):
        requested = (
            config.dense_revision
            if name == config.dense_model
            else (config.reranker_revision if name == config.reranker_model else None)
        )
        info = api.model_info(name, revision=requested, files_metadata=True)
        models[name] = {
            "revision": info.sha,
            "safetensors_bytes": sum(
                f.size or 0
                for f in info.siblings
                if f.rfilename.endswith(".safetensors")
            ),
            "license": getattr(info.card_data, "license", None),
            "training_exposure": "not_audited",
        }
    write_json(dest / "model_manifest.json", models)
    tokenizers = {
        name: AutoTokenizer.from_pretrained(name, revision=m["revision"])
        for name, m in models.items()
    }
    queries = read_json(dest / "queries.json")
    reference = tokenizers[config.reranker_model]
    # One common view fits every pilot query and both rerankers; model adapters
    # reject later overflow rather than independently truncating evidence.
    longest = max(
        (q["text"] for q in queries.values()), key=lambda q: len(reference.encode(q))
    )
    bge = tokenizers["BAAI/bge-reranker-v2-m3"]
    longest_bge = max(
        (q["text"] for q in queries.values()), key=lambda q: len(bge.encode(q))
    )
    views = {}
    for document in iter_documents(dest / "corpus.sqlite"):
        raw = (document.title + "\n" + document.text).strip()
        tokens = reference.encode(raw, add_special_tokens=False)[
            : config.max_document_tokens
        ]
        while tokens:
            text = reference.decode(tokens, skip_special_tokens=True)
            if (
                len(
                    reference.encode(
                        reranker_text(longest, text, TASKS[config.task]),
                        add_special_tokens=False,
                    )
                )
                <= config.max_input_tokens
                and len(bge(longest_bge, text)["input_ids"]) <= config.max_input_tokens
                and len(tokenizers[config.dense_model].encode(text))
                <= config.max_input_tokens
            ):
                break
            tokens = tokens[:-8]
        if not tokens:
            raise ValueError("Query/prompt overhead leaves no document text")
        views[document.doc_id] = {
            "text": text,
            "hash": digest(text),
            "content_hash": document.content_sha256,
        }
    write_json(dest / "text_views.json", views)
    resolved = {
        **config.model_dump(),
        "corpus_revision": manifest["revision"],
        "dense_revision": models[config.dense_model]["revision"],
        "reranker_revision": models[config.reranker_model]["revision"],
    }
    write_json(dest / "input_config.json", config.model_dump())
    write_json(dest / "resolved_config.json", resolved)
    import yaml

    (dest / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    return dest


def load_run(path):
    original = load_config(path)
    dest = run_dir(original)
    if read_json(dest / "input_config.json") != original.model_dump():
        raise ValueError("Config differs from frozen input; use new experiment_id")
    from .config import Experiment

    return Experiment.model_validate(read_json(dest / "resolved_config.json")), dest


def retrieve(config_path):
    import bm25s
    import Stemmer

    config, dest = load_run(config_path)
    views, queries = (
        read_json(dest / "text_views.json"),
        read_json(dest / "queries.json"),
    )
    ids = sorted(views)
    # Core corpus fits memory; ingestion remains streaming and large corpora are
    # explicitly unsupported by this local index builder until profiled.
    if len(ids) > 100000:
        raise ValueError("Large corpus indexing requires a separately profiled adapter")
    texts = [views[d]["text"] for d in ids]
    stemmer = Stemmer.Stemmer("english")
    index = bm25s.BM25(method="lucene", k1=config.bm25_k1, b=config.bm25_b)
    index.index(bm25s.tokenize(texts, stopwords="en", stemmer=stemmer))
    index.save(str(dest / "bm25_index"))
    bm25 = {}
    for q, query in queries.items():
        tokens = bm25s.tokenize(
            query["text"], stopwords="en", stemmer=stemmer, return_ids=False
        )
        bm25[q] = dict(zip(ids, map(float, index.get_scores(tokens[0]))))
    write_json(dest / "bm25_scores.json", bm25)
    model = DenseModel(
        config.dense_model,
        config.dense_revision,
        config.device,
        config.max_input_tokens,
        TASKS[config.task],
    )
    vector_path = dest / "embeddings.npy"
    checkpoint = dest / "embedding_checkpoint.json"
    identity = digest(
        {
            "ids": ids,
            "views": [views[d]["hash"] for d in ids],
            "model": config.dense_revision,
        }
    )
    start = 0
    vectors = None
    if checkpoint.exists():
        saved = read_json(checkpoint)
        if saved["identity"] != identity:
            raise ValueError("Embedding checkpoint provenance mismatch")
        start = saved["completed"]
        vectors = np.load(vector_path, mmap_mode="r+")
    for offset in range(start, len(ids), config.batch_size):
        output = model.encode(texts[offset : offset + config.batch_size])
        if vectors is None:
            vectors = np.lib.format.open_memmap(
                vector_path,
                mode="w+",
                dtype="float32",
                shape=(len(ids), output.shape[1]),
            )
        vectors[offset : offset + len(output)] = output
        vectors.flush()
        write_json(
            checkpoint, {"identity": identity, "completed": offset + len(output)}
        )
        if offset % 100 == 0:
            print(f"Embedded {offset + len(output)}/{len(ids)} documents", flush=True)
    write_json(dest / "embedding_rows.json", ids)
    pools, source_runs = {}, {"bm25": {}, "dense": {}, "rrf": {}}
    for q, query in queries.items():
        vector = model.encode([query["text"]], query=True)[0]
        # One query by blocked documents, never a query-by-corpus matrix.
        dense = {}
        for start in range(0, len(ids), 4096):
            scores = np.asarray(vectors[start : start + 4096]) @ vector
            dense.update(zip(ids[start : start + 4096], map(float, scores)))
        lexical = bm25[q]
        if config.exclude_self_match:
            lexical = {d: s for d, s in lexical.items() if d != q}
            dense = {d: s for d, s in dense.items() if d != q}
        pool = make_pool(
            q,
            lexical,
            dense,
            {d: views[d]["hash"] for d in ids},
            config.depth,
            config.pool_size,
        )
        pools[q] = pool.model_dump()
        br, dr = rank(lexical)[: config.depth], rank(dense)[: config.depth]
        source_runs["bm25"][q] = {d: bm25[q][d] for d in br}
        source_runs["dense"][q] = {d: dense[d] for d in dr}
        source_runs["rrf"][q] = rrf([br, dr])
    write_json(dest / "pools.json", pools)
    write_json(dest / "source_runs.json", source_runs)
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            **c,
            "membership_hash": p["membership_hash"],
            "ordered_input_hash": p["ordered_input_hash"],
        }
        for p in pools.values()
        for c in p["candidates"]
    ]
    pq.write_table(pa.Table.from_pylist(rows), dest / "candidates.parquet")
    return {"queries": len(pools), "corpus": len(ids)}


def rerank(config_path, system="qwen"):
    config, dest = load_run(config_path)
    queries, views = (
        read_json(dest / "queries.json"),
        read_json(dest / "text_views.json"),
    )
    pools = {
        q: Pool.model_validate(p) for q, p in read_json(dest / "pools.json").items()
    }
    for pool in pools.values():
        for c in pool.candidates:
            if (
                views[c.doc_id]["hash"] != digest(views[c.doc_id]["text"])
                or c.text_view_hash != views[c.doc_id]["hash"]
            ):
                raise ValueError("Candidate evidence differs from its frozen text view")
    settings = Settings.load() if system == "jev" else None
    output_path = dest / f"scores_{system}.json"
    saved = (
        read_json(output_path)
        if output_path.exists()
        else {"scores": {}, "status": "running"}
    )
    expected = digest(
        {
            "pools": {q: p.ordered_input_hash for q, p in pools.items()},
            "config": config.model_dump(),
            "system": system,
            "queries": queries,
            "rubric": rubric(config.task),
            "gateway_model": settings.model if settings else None,
            "adapter_sha256": file_digest(
                Path(__file__).with_name("jev.py" if system == "jev" else "models.py")
            ),
        }
    )
    if saved.get("provenance", expected) != expected:
        raise ValueError("Score checkpoint provenance changed")
    saved["provenance"] = expected
    if system == "jev":
        from .jev import JevClient

        capabilities = read_json("manifests/capabilities.json")
        if capabilities["status"] != "noul_functional_checks_passed":
            raise ValueError(
                "Jev capability preflight must pass before benchmark requests"
            )
        if config.jev_revision and config.jev_revision != settings.model:
            raise ValueError(
                "Configured Jev revision differs from the requested Gateway model"
            )
        client = JevClient(
            settings,
            dest,
            config.token_budget,
            config.request_budget,
            config.max_attempts,
        )
        client.sentinel()
    else:
        models = read_json(dest / "model_manifest.json")
        name = config.reranker_model if system == "qwen" else "BAAI/bge-reranker-v2-m3"
        model = LocalReranker(
            name,
            models[name]["revision"],
            config.device,
            config.max_input_tokens,
            TASKS[config.task],
        )
    try:
        for q, pool in pools.items():
            ids = [c.doc_id for c in pool.candidates]
            existing = saved["scores"].get(q, {})
            if set(existing) == set(ids):
                rank(existing, ids)
                continue
            scores = dict(existing)
            todo = [d for d in ids if d not in scores]
            if system == "jev":
                size = (
                    config.contextual_batch_size
                    if "contextual" in config.jev_mode
                    else 1
                )
                # Preserve original group boundaries during resume; partial groups
                # repeat the same semantic request and use the request cache.
                for start in range(0, len(ids), size):
                    group = ids[start : start + size]
                    if all(d in scores for d in group):
                        continue
                    candidates = []
                    for d in group:
                        candidate = next(c for c in pool.candidates if c.doc_id == d)
                        features = {
                            k: v
                            for k, v in candidate.model_dump().items()
                            if k
                            in {
                                "bm25_score",
                                "dense_score",
                                "bm25_rank",
                                "dense_rank",
                                "in_bm25",
                                "in_dense",
                            }
                        }
                        candidates.append(
                            {
                                "doc_id": d,
                                "text": views[d]["text"],
                                "features": features,
                            }
                        )
                    state, questions, mapping = build_request(
                        queries[q]["text"], candidates, config.task, config.jev_mode
                    )
                    response = client.evaluate(
                        state,
                        questions,
                        provenance={
                            "query_id": q,
                            "pool": pool.ordered_input_hash,
                            "corpus": config.corpus_revision,
                            "views": [views[d]["hash"] for d in group],
                            "mode": config.jev_mode,
                        },
                    )
                    for key, d in mapping.items():
                        scores[d] = response["answers"][key]["noul"]
                    saved["scores"][q] = scores
                    write_json(output_path, saved)
            else:
                for i in range(0, len(todo), config.batch_size):
                    batch = todo[i : i + config.batch_size]
                    scores.update(
                        zip(
                            batch,
                            model.score(
                                queries[q]["text"], [views[d]["text"] for d in batch]
                            ),
                        )
                    )
                    saved["scores"][q] = scores
                    write_json(output_path, saved)
            rank(scores, ids)
            print(
                f"{system}: completed query {q} ({len(saved['scores'])}/{len(pools)})",
                flush=True,
            )
        saved["status"] = "complete"
    except Exception as exc:
        saved["status"] = "incomplete"
        saved["failure_type"] = type(exc).__name__
        write_json(output_path, saved)
        raise
    write_json(output_path, saved)
    return {"status": saved["status"], "queries": len(saved["scores"])}


def evaluate(config_path):
    config, dest = load_run(config_path)
    judgments = read_json(dest / "qrels.json")
    pools = {
        q: Pool.model_validate(p) for q, p in read_json(dest / "pools.json").items()
    }
    sources = read_json(dest / "source_runs.json")
    systems = {
        "rrf": {
            q: {c.doc_id: sources["rrf"][q][c.doc_id] for c in p.candidates}
            for q, p in pools.items()
        }
    }
    failures = {"rrf": []}
    external_names = [
        p.stem.removeprefix("scores_")
        for p in sorted(dest.glob("scores_external_*.json"))
    ]
    for name in ("qwen", "bge", "jev", *external_names):
        path = dest / f"scores_{name}.json"
        if not path.exists():
            continue
        raw = read_json(path)["scores"]
        complete, failed = {}, []
        for q, p in pools.items():
            ids = {c.doc_id for c in p.candidates}
            if set(raw.get(q, {})) == ids:
                complete[q] = raw[q]
            else:
                failed.append(q)
        systems[name] = complete
        failures[name] = failed
        if failed:
            systems[name + "_operational_fallback"] = {
                q: complete.get(q, systems["rrf"][q]) for q in pools
            }
            failures[name + "_operational_fallback"] = failed
    # Pilot fusion is fixed RRF; numerical weights remain development-only.
    if "jev" in systems:
        systems["b_d_j_rrf"] = {}
        failures["b_d_j_rrf"] = failures["jev"]
        for q, jev_scores in systems["jev"].items():
            ids = set(jev_scores)
            full = rrf(
                [rank(sources["bm25"][q]), rank(sources["dense"][q]), rank(jev_scores)]
            )
            systems["b_d_j_rrf"][q] = {d: full[d] for d in ids}
    rows, summary = [], {}
    for name, runs in systems.items():
        export_trec(dest / f"{name}.trec", runs, name)
        selections = {}
        values = []
        for q, scores in runs.items():
            ids = [c.doc_id for c in pools[q].candidates]
            ordered = rank(scores, ids)
            result = evaluate_query(
                ordered, judgments[q], ids, config.binary_threshold, config.gain
            )
            rows.append({"query_id": q, "system": name, **result})
            selections[q] = select(scores, ids, config.selection_k)
            if result["eligible"]:
                values.append(result)
        write_json(dest / f"selection_{name}.json", selections)
        summary[name] = {
            "eligible_queries": len(values),
            "failed_queries": failures[name],
            "failure_fraction": len(failures[name]) / len(pools),
            **{
                key: sum(r[key] for r in values) / len(values) if values else None
                for key in (
                    "ndcg@10",
                    "recall@10",
                    "mrr@10",
                    "precision@10",
                    "judged@10",
                    "candidate_recall",
                )
            },
        }
    write_json(dest / "per_query_metrics.json", rows)
    write_json(dest / "summary.json", summary)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), dest / "per_query_metrics.parquet")
    score_rows = [
        {"query_id": q, "doc_id": d, "system": name, "score": s}
        for name, runs in systems.items()
        for q, scores in runs.items()
        for d, s in scores.items()
    ]
    pq.write_table(pa.Table.from_pylist(score_rows), dest / "scores.parquet")
    lines = [
        f"# {config.dataset} {config.split} retrieval evaluation\n",
        "Confirmatory status: "
        + str(config.confirmatory)
        + ". No calibration or superiority claim is implied.\n",
        "| System | Queries | nDCG@10 | Recall@10 | Failures |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name, result in summary.items():
        lines.append(
            f"| {name} | {result['eligible_queries']} | {result['ndcg@10']} | {result['recall@10']} | {len(result['failed_queries'])} |\n"
        )
    lines.append(
        "\nMissing systems have not run. Completed-query and operational fallback results are distinct.\n"
    )
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/{config.experiment_id}.md").write_text("".join(lines))
    return summary
