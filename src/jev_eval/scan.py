"""Explicit, resumable Jev full-corpus retrieval and matched dense comparison."""

import fcntl
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager

from .config import Settings
from .io import digest, read_json, write_json
from .jev import JevClient
from .metrics import evaluate_query, export_trec, rank
from .pipeline import load_run
from .rubrics import build_request


class SharedRateLimiter:
    """Pace all workers and make provider throttling pause the entire scan."""

    def __init__(self, requests_per_second):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.maximum = requests_per_second
        self.rate = requests_per_second
        self.condition = threading.Condition()
        self.next_at = 0.0
        self.pause_until = 0.0
        self.successes = 0

    def acquire(self):
        with self.condition:
            while True:
                now = time.monotonic()
                wait_for = max(self.next_at, self.pause_until) - now
                if wait_for <= 0:
                    self.next_at = now + 1 / self.rate
                    return
                self.condition.wait(wait_for)

    def penalize(self, retry_after):
        with self.condition:
            now = time.monotonic()
            if now >= self.pause_until:
                self.rate = max(0.1, self.rate / 2)
                self.successes = 0
            self.pause_until = max(
                self.pause_until, now + max(1.0, retry_after)
            )
            self.condition.notify_all()
        print(
            f"Jev throttled: pausing workers for at least {retry_after:.1f}s; "
            f"rate now {self.rate:.2f} requests/s",
            flush=True,
        )

    def succeeded(self):
        with self.condition:
            self.successes += 1
            if self.successes >= 50 and self.rate < self.maximum:
                self.rate = min(self.maximum, self.rate * 1.25)
                self.successes = 0
                self.condition.notify_all()


@contextmanager
def scan_lock(dest):
    path = dest / "jev_scan_typesafe.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another parallel Jev scan is using this run") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def scan(
    config_path,
    query_count,
    approved_pairs,
    request_budget,
    token_budget,
    workers=12,
    requests_per_second=10.0,
):
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    limiter = SharedRateLimiter(requests_per_second)
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
    capabilities = read_json("manifests/capabilities_typesafe.json")
    if capabilities["status"] != "noul_functional_checks_passed":
        raise ValueError("Jev Noul preflight has not passed")
    settings = Settings.load()
    if config.jev_revision and config.jev_revision != settings.model:
        raise ValueError("Configured Jev revision differs from Gateway model")
    path = dest / "jev_scan_typesafe_scores.json"
    identity = digest(
        {
            "queries": queries,
            "views": {d: views[d]["hash"] for d in ids},
            "config": config.model_dump(),
            "source": "typesafe_direct",
            "model": settings.model,
            "mode": "independent_noul_full_corpus",
        }
    )
    saved = read_json(path) if path.exists() else {"identity": identity, "scores": {}}
    if saved["identity"] != identity:
        raise ValueError("Jev scan checkpoint provenance changed")
    if set(saved["scores"]) - set(selected):
        raise ValueError("Jev scan checkpoint contains unexpected queries")
    with scan_lock(dest):
        client_dir = dest / "typesafe_scan"
        client = JevClient(
            settings, client_dir, token_budget, request_budget, config.max_attempts
        )
        clients = []
        clients_lock = threading.Lock()
        local = threading.local()

        def score_pair(q, d):
            worker = getattr(local, "client", None)
            if worker is None:
                worker = JevClient(
                    settings, client_dir, token_budget, request_budget, config.max_attempts
                )
                worker.rate_limiter = limiter
                # Reuse the catalog fetched by the sentinel. Each thread owns its
                # HTTP connection and SQLite connection.
                for name in ("pricing", "pricing_time", "catalog_model", "resolved_model"):
                    if hasattr(client, name):
                        setattr(worker, name, getattr(client, name))
                local.client = worker
                with clients_lock:
                    clients.append(worker)
            state, questions, mapping = build_request(
                queries[q]["text"],
                [{"doc_id": d, "text": views[d]["text"]}],
                config.task,
                "independent",
            )
            response = worker.evaluate(
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
            return response["answers"][next(iter(mapping))]["noul"]

        try:
            client.sentinel()
            saved.pop("failure_type", None)
            saved["status"] = "incomplete"
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for q in selected:
                    scores = saved["scores"].setdefault(q, {})
                    if set(scores) - set(ids):
                        raise ValueError(
                            "Jev scan checkpoint contains unexpected documents"
                        )
                    remaining = iter(d for d in ids if d not in scores)
                    pending = {}
                    first_error = None

                    def fill_queue():
                        while len(pending) < workers * 2:
                            try:
                                d = next(remaining)
                            except StopIteration:
                                return
                            pending[pool.submit(score_pair, q, d)] = d

                    fill_queue()
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            d = pending.pop(future)
                            if future.cancelled():
                                continue
                            try:
                                scores[d] = future.result()
                            except Exception as error:
                                if first_error is None:
                                    first_error = error
                            if len(scores) and len(scores) % 100 == 0:
                                write_json(path, saved)
                                print(
                                    f"Jev scan: query {q}, {len(scores)}/{len(ids)} documents",
                                    flush=True,
                                )
                        if first_error is not None:
                            for future in pending:
                                future.cancel()
                        else:
                            fill_queue()
                    if first_error is not None:
                        raise first_error
                    write_json(path, saved)
                    print(
                        f"Jev scan: completed query {q} "
                        f"({len(saved['scores'])}/{len(selected)})",
                        flush=True,
                    )
            saved["status"] = "complete"
        except Exception as error:
            saved["status"] = "incomplete"
            saved["failure_type"] = type(error).__name__
            write_json(path, saved)
            raise
        finally:
            for worker in [client, *clients]:
                if hasattr(worker, "http"):
                    worker.http.close()
                if hasattr(worker, "ledger"):
                    worker.ledger.db.close()
        write_json(path, saved)
    return {"queries": len(selected), "documents": len(ids), "pairs": pairs}


def compare(config_path):
    config, dest = load_run(config_path)
    saved = read_json(dest / "jev_scan_typesafe_scores.json")
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
            "source": "typesafe_direct",
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
        "jev_typesafe_full_scan": saved["scores"],
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
        export_trec(dest / f"retrieval_typesafe_{name}.trec", top_runs, name)
    write_json(dest / "retrieval_typesafe_per_query_metrics.json", rows)
    write_json(dest / "retrieval_typesafe_summary.json", summary)
    return summary
