"""Explicit, resumable Jev full-corpus retrieval and matched dense comparison."""

from .config import Settings
from .io import digest, read_json, write_json
from .jev import JevClient
from .metrics import evaluate_query, export_trec, rank
from .pipeline import load_run
from .rubrics import build_request


def scan(config_path, query_count, approved_pairs, request_budget, token_budget):
    config, dest = load_run(config_path)
    queries = read_json(dest / "queries.json")
    views = read_json(dest / "text_views.json")
    if query_count < 1 or query_count > len(queries):
        raise ValueError("Query count must be between 1 and the prepared query count")
    if not views or any(v["hash"] != digest(v["text"]) for v in views.values()):
        raise ValueError("Frozen corpus text view is missing or has changed")
    selected = list(queries)[:query_count]
    ids = sorted(views)
    pairs = len(selected) * len(ids)
    if approved_pairs != pairs:
        raise ValueError(f"Full scan requires --approve-pairs {pairs} exactly")
    if request_budget < pairs or token_budget < 1:
        raise ValueError("Explicit request/token budgets are insufficient")
    capabilities = read_json("manifests/capabilities.json")
    if capabilities["status"] != "noul_functional_checks_passed":
        raise ValueError("Jev Noul preflight has not passed")
    settings = Settings.load()
    if config.jev_revision and config.jev_revision != settings.model:
        raise ValueError("Configured Jev revision differs from Gateway model")
    path = dest / "jev_scan_scores.json"
    identity = digest(
        {
            "queries": queries,
            "views": {d: views[d]["hash"] for d in ids},
            "config": config.model_dump(),
            "model": settings.model,
            "mode": "independent_noul_full_corpus",
        }
    )
    saved = read_json(path) if path.exists() else {"identity": identity, "scores": {}}
    if saved["identity"] != identity:
        raise ValueError("Jev scan checkpoint provenance changed")
    if set(saved["scores"]) - set(selected):
        raise ValueError("Jev scan checkpoint contains unexpected queries")
    client = JevClient(
        settings, dest, token_budget, request_budget, config.max_attempts
    )
    try:
        client.sentinel()
        for q in selected:
            scores = saved["scores"].setdefault(q, {})
            if set(scores) - set(ids):
                raise ValueError("Jev scan checkpoint contains unexpected documents")
            for index, d in enumerate(ids, 1):
                if d in scores:
                    continue
                state, questions, mapping = build_request(
                    queries[q]["text"],
                    [{"doc_id": d, "text": views[d]["text"]}],
                    config.task,
                    "independent",
                )
                response = client.evaluate(
                    state,
                    questions,
                    provenance={
                        "role": "full_corpus_retrieval",
                        "query_id": q,
                        "document_id": d,
                        "text_hash": views[d]["hash"],
                        "corpus": config.corpus_revision,
                    },
                )
                scores[d] = response["answers"][next(iter(mapping))]["noul"]
                if index % 100 == 0:
                    write_json(path, saved)
            write_json(path, saved)
            print(
                f"Jev scan: completed query {q} ({len(saved['scores'])}/{len(selected)})",
                flush=True,
            )
        saved["status"] = "complete"
    except Exception as error:
        saved["status"] = "incomplete"
        saved["failure_type"] = type(error).__name__
        write_json(path, saved)
        raise
    write_json(path, saved)
    return {"queries": len(selected), "documents": len(ids), "pairs": pairs}


def compare(config_path):
    config, dest = load_run(config_path)
    saved = read_json(dest / "jev_scan_scores.json")
    views = read_json(dest / "text_views.json")
    ids = set(views)
    selected = list(saved["scores"])
    if saved.get("status") != "complete" or not selected:
        raise ValueError("Jev full-corpus scan is incomplete")
    if any(set(scores) != ids for scores in saved["scores"].values()):
        raise ValueError("Jev scan does not cover every corpus document")
    queries = read_json(dest / "queries.json")
    expected = digest(
        {
            "queries": queries,
            "views": {d: views[d]["hash"] for d in sorted(ids)},
            "config": config.model_dump(),
            "model": Settings.load().model,
            "mode": "independent_noul_full_corpus",
        }
    )
    if saved["identity"] != expected:
        raise ValueError("Jev scan provenance changed")
    sources = read_json(dest / "source_runs.json")
    qrels = read_json(dest / "qrels.json")
    systems = {
        "bm25": sources["bm25"],
        "dense_cosine": sources["dense"],
        "jev_full_scan": saved["scores"],
    }
    rows, summary = [], {}
    for name, runs in systems.items():
        top_runs = {}
        values = []
        for q in selected:
            ordered = rank(runs[q])[: config.depth]
            top_runs[q] = {d: runs[q][d] for d in ordered}
            result = evaluate_query(
                ordered, qrels[q], ordered, config.binary_threshold, config.gain
            )
            rows.append({"query_id": q, "system": name, **result})
            if result["eligible"]:
                values.append(result)
        summary[name] = {
            "eligible_queries": len(values),
            "evaluated_queries": len(selected),
            "retrieval_depth": config.depth,
            **{
                metric: sum(row[metric] for row in values) / len(values)
                if values
                else None
                for metric in ("ndcg@10", "recall@10", "mrr@10", "candidate_recall")
            },
        }
        export_trec(dest / f"retrieval_{name}.trec", top_runs, name)
    write_json(dest / "retrieval_per_query_metrics.json", rows)
    write_json(dest / "retrieval_summary.json", summary)
    return summary
