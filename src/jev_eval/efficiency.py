"""Frozen small-sample efficiency comparison; never rewrites core experiment runs."""

import argparse
import math
import os
import platform
import random
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from .config import Settings
from .io import digest, file_digest, read_json, write_json
from .jev import JevClient
from .ledger import now
from .metrics import evaluate_query, rank
from .rubrics import TASKS, build_request, rubric
from .scan import SharedRateLimiter, scan_lock
from .stats import paired_comparison
from .tradeoffs import load_archive, percentiles, write_csv

ARMS = ("qwen", "bge", "jev_sequential", "jev_concurrent", "jev_packed")
INPUT_FILES = (
    "resolved_config.json",
    "model_manifest.json",
    "queries.json",
    "pools.json",
    "text_views.json",
    "qrels.json",
    "source_runs.json",
    "scores_qwen.json",
    "scores_bge.json",
    "scores_jev.json",
)


def prepare(
    root, output, experiments, query_count, depths, seed, workers, pack_size, rps
):
    if output.resolve() in {root.resolve(), (root / "runs").resolve()} or any(
        output.resolve().is_relative_to((root / "runs" / experiment).resolve())
        for experiment in experiments
    ):
        raise ValueError("Benchmark output must be separate from core experiment runs")
    if not 1 <= workers <= 32 or not 1 <= pack_size <= 10 or not 0 < rps <= 10:
        raise ValueError(
            "Require workers 1..32, pack size 1..10, rate 0..10 requests/s"
        )
    if not depths or any(d < 1 for d in depths) or len(set(depths)) != len(depths):
        raise ValueError("Depths must be distinct positive integers")
    datasets = []
    for experiment in experiments:
        archive = load_archive(root, experiment)
        eligible = sorted(archive["queries"])
        if not 1 <= query_count <= len(eligible):
            raise ValueError("Query count exceeds the archived queries")
        selected = random.Random(seed).sample(eligible, query_count)
        candidates = {}
        for q in selected:
            ids = [c.doc_id for c in archive["pools"][q].candidates]
            order = rank({d: archive["sources"]["rrf"][q][d] for d in ids})
            if len(order) < max(depths):
                raise ValueError("Requested depth exceeds frozen pool")
            candidates[q] = order[: max(depths)]
        datasets.append(
            {
                "experiment": experiment,
                "dataset": archive["config"]["dataset"],
                "query_ids": selected,
                "candidates": candidates,
                "input_hashes": {
                    name: file_digest(archive["directory"] / name)
                    for name in INPUT_FILES
                },
            }
        )
    # Randomize conditions once, before any live timing or packing quality is observed.
    jobs = [
        (d["experiment"], depth, arm)
        for d in datasets
        for depth in depths
        for arm in ARMS
    ]
    random.Random(seed + 1).shuffle(jobs)
    total_pairs = len(experiments) * query_count * sum(depths)
    manifest = {
        "version": 1,
        "created_at": now(),
        "seed": seed,
        "exploratory": True,
        "query_count_per_dataset": query_count,
        "depths": sorted(depths),
        "workers": workers,
        "pack_size": pack_size,
        "requests_per_second": rps,
        "datasets": datasets,
        "job_order": jobs,
        "unretried_scoring_requests": 2 * total_pairs
        + len(experiments)
        * query_count
        * sum(math.ceil(d / pack_size) for d in depths),
        "timing": "One query at a time; concurrent document scoring within a query. Query latency includes pacing, retries and worker scheduling. Model load and warmup excluded.",
        "packing": "Query in shared state; each question contains only its own target document. Different layout from original independent requests; re-evaluate quality.",
        "repeats": 1,
        "p99_warning": "30 queries cannot establish reliable tail behavior",
    }
    manifest["identity"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        raise ValueError("Manifest already exists; use the run command to resume")
    write_json(path, manifest)
    return manifest


def packed_request(query, candidates, task):
    if not candidates or len({c["doc_id"] for c in candidates}) != len(candidates):
        raise ValueError("Empty or duplicated candidates")
    state = {"task": task, "query": query}
    questions, mapping = {}, {}
    for i, candidate in enumerate(candidates):
        key = f"candidate_{i:03d}"
        question = rubric(task)
        instruction = (
            question["instructions"] + " Evaluate only the document in this question."
        )
        question["instructions"] = {
            "instructions": instruction,
            "document": {"id": "candidate_000", "text": candidate["text"]},
        }
        questions[key], mapping[key] = question, candidate["doc_id"]
    return state, questions, mapping


def verify_manifest(root, output):
    manifest = read_json(output / "manifest.json")
    identity = manifest["identity"]
    if digest({k: v for k, v in manifest.items() if k != "identity"}) != identity:
        raise ValueError("Benchmark manifest has changed")
    archives = {}
    for dataset in manifest["datasets"]:
        experiment = dataset["experiment"]
        directory = root / "runs" / experiment
        for name, expected in dataset["input_hashes"].items():
            if file_digest(directory / name) != expected:
                raise ValueError(f"Frozen input changed: {experiment}/{name}")
        archives[experiment] = load_archive(root, experiment)
    return manifest, archives


def hardware():
    import psutil

    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
    }
    for command in (("lscpu",), ("nvidia-smi",)):
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=10
            )
            result[command[0]] = (
                completed.stdout if completed.returncode == 0 else "unavailable"
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result[command[0]] = "unavailable"
    return result


def totals_since(client, attempt):
    rows = client.ledger.db.execute(
        "SELECT status, actual_usd, reserved_usd, tokens, raw_path FROM attempts WHERE id>? ORDER BY id",
        (attempt,),
    ).fetchall()
    latencies = []
    input_tokens = output_tokens = 0
    for row in rows:
        raw = read_json(row[4]) if row[4] else {}
        if "elapsed_seconds" in raw:
            latencies.append(raw["elapsed_seconds"])
        usage = raw.get("response", {}).get("usage", {})
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
    return {
        "attempts": len(rows),
        "unsuccessful_attempts": sum(r[0] != "success" for r in rows),
        "estimated_cost_usd": float(
            sum((Decimal(r[1]) for r in rows if r[1] is not None), Decimal(0))
        ),
        "uncertain_reserved_usd": float(
            sum((Decimal(r[2]) for r in rows if r[1] is None), Decimal(0))
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attempt_latency_seconds": percentiles(latencies),
    }


def jev_job(archive, dataset, depth, arm, manifest, output, settings, max_cost):
    limiter = SharedRateLimiter(manifest["requests_per_second"])
    width = manifest["workers"] if arm == "jev_concurrent" else 1
    settings = settings.model_copy(
        update={"max_cost_usd": min(settings.max_cost_usd, max_cost)}
    )
    clients = [
        JevClient(
            settings,
            output,
            token_budget=100000000,
            request_budget=25000,
            max_attempts=3,
            ledger_path=output / "budget_typesafe.sqlite",
        )
        for _ in range(width)
    ]
    coordinator = clients[0]
    pool = ThreadPoolExecutor(max_workers=width)
    try:
        coordinator.discover()
        for client in clients:
            client.pricing, client.catalog_model = (
                coordinator.pricing,
                coordinator.catalog_model,
            )
            client.rate_limiter = limiter
        first_query = dataset["query_ids"][0]
        ids = dataset["candidates"][first_query][:depth]

        def score_chunk(worker, q, docs, trial):
            candidates = [
                {"doc_id": d, "text": archive["views"][d]["text"]} for d in docs
            ]
            if arm == "jev_packed":
                state, questions, mapping = packed_request(
                    archive["queries"][q]["text"], candidates, archive["config"]["task"]
                )
            else:
                state, questions, mapping = build_request(
                    archive["queries"][q]["text"],
                    candidates,
                    archive["config"]["task"],
                    "independent",
                )
            response = worker.evaluate(
                state,
                questions,
                provenance={
                    "benchmark": manifest["identity"],
                    "query": q,
                    "depth": depth,
                    "arm": arm,
                    "trial": trial,
                    "ids": docs,
                },
            )
            return {
                mapping[key]: value["noul"]
                for key, value in response["answers"].items()
            }

        # Warm every HTTP connection; UUIDs force paid, uncached requests.
        warm_ids = ids[: manifest["pack_size"]] if arm == "jev_packed" else ids[:1]
        warmups = [
            pool.submit(
                score_chunk, client, first_query, warm_ids, "warmup-" + uuid.uuid4().hex
            )
            for client in clients
        ]
        for future in warmups:
            future.result()
        for q in dataset["query_ids"]:
            path = output / "queries" / f"{dataset['dataset']}-{depth}-{arm}-{q}.json"
            if path.exists() and read_json(path)["status"] == "complete":
                continue
            ids = dataset["candidates"][q][:depth]
            pack_size = manifest["pack_size"] if arm == "jev_packed" else 1
            chunks = [ids[i : i + pack_size] for i in range(0, len(ids), pack_size)]
            lanes = [chunks[i::width] for i in range(width)]
            trial = uuid.uuid4().hex
            before = coordinator.ledger.db.execute(
                "SELECT COALESCE(MAX(id),0) FROM attempts"
            ).fetchone()[0]
            started = time.perf_counter()

            def lane(worker, work):
                scores = {}
                for chunk in work:
                    scores.update(score_chunk(worker, q, chunk, trial))
                return scores

            futures = [
                pool.submit(lane, worker, work) for worker, work in zip(clients, lanes)
            ]
            scores, errors = {}, []
            for future in futures:
                try:
                    scores.update(future.result())
                except Exception as exc:
                    errors.append(type(exc).__name__ + ": " + str(exc))
            elapsed = time.perf_counter() - started
            row = query_row(archive, q, ids, scores, arm, depth, elapsed, errors)
            row.update(totals_since(coordinator, before))
            row["resolved_models"] = sorted(
                {c.resolved_model for c in clients if c.resolved_model}
            )
            if len(row["resolved_models"]) != 1:
                row["status"] = "failed"
                row["metrics"] = None
                row["errors"].append("Different model identities across workers")
            row["trial"] = trial
            write_json(path, row)
            print(
                f"{dataset['dataset']} {arm} depth={depth} {q}: {row['status']} {elapsed:.2f}s",
                flush=True,
            )
            if any("cap reached" in error.lower() for error in errors):
                raise ValueError("Benchmark budget reached; partial results retained")
    finally:
        pool.shutdown(wait=True)
        for client in clients:
            client.http.close()
            client.ledger.db.close()


def query_row(archive, q, ids, scores, arm, depth, elapsed, errors):
    complete = not errors and set(scores) == set(ids)
    return {
        "dataset": archive["config"]["dataset"],
        "query_id": q,
        "arm": arm,
        "depth": depth,
        "status": "complete" if complete else "failed",
        "query_latency_seconds": elapsed,
        "scores": scores,
        "errors": errors,
        "metrics": evaluate_query(
            rank(scores, ids),
            archive["qrels"][q],
            ids,
            archive["config"]["binary_threshold"],
            archive["config"]["gain"],
        )
        if complete
        else None,
    }


def local_job(archive, dataset, depth, arm, manifest, output, device):
    import torch

    from .models import LocalReranker

    config = archive["config"]
    name = config["reranker_model"] if arm == "qwen" else "BAAI/bge-reranker-v2-m3"
    revision = read_json(archive["directory"] / "model_manifest.json")[name]["revision"]
    model = LocalReranker(
        name, revision, device, config["max_input_tokens"], TASKS[config["task"]]
    )
    first = dataset["query_ids"][0]
    # Warm a full batch, using exactly the same frozen text and task.
    warm_ids = dataset["candidates"][first][: config["batch_size"]]
    model.score(
        archive["queries"][first]["text"],
        [archive["views"][d]["text"] for d in warm_ids],
    )
    write_json(
        output / f"model_{dataset['dataset']}_{depth}_{arm}.json",
        {
            "model": name,
            "revision": revision,
            "device": model.device,
            "batch_size": config["batch_size"],
            "dtype": str(next(model.model.parameters()).dtype),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
    )
    try:
        for q in dataset["query_ids"]:
            path = output / "queries" / f"{dataset['dataset']}-{depth}-{arm}-{q}.json"
            if path.exists() and read_json(path)["status"] == "complete":
                continue
            ids = dataset["candidates"][q][:depth]
            scores = {}
            started = time.perf_counter()
            for i in range(0, len(ids), config["batch_size"]):
                batch = ids[i : i + config["batch_size"]]
                values = model.score(
                    archive["queries"][q]["text"],
                    [archive["views"][d]["text"] for d in batch],
                )
                scores.update(zip(batch, values))
            row = query_row(
                archive, q, ids, scores, arm, depth, time.perf_counter() - started, []
            )
            row["device"] = model.device
            write_json(path, row)
            print(
                f"{dataset['dataset']} {arm} depth={depth} {q}: {row['query_latency_seconds']:.2f}s",
                flush=True,
            )
    finally:
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def summarize(output):
    manifest = read_json(output / "manifest.json")
    rows = [read_json(p) for p in sorted((output / "queries").glob("*.json"))]
    summary, comparisons = [], []
    for dataset in manifest["datasets"]:
        for depth in manifest["depths"]:
            for arm in ARMS:
                all_rows = [
                    r
                    for r in rows
                    if (r["dataset"], r["depth"], r["arm"])
                    == (dataset["dataset"], depth, arm)
                ]
                eligible = [r for r in all_rows if r["status"] == "complete"]
                total_seconds = sum(r["query_latency_seconds"] for r in all_rows)
                summary.append(
                    {
                        "dataset": dataset["dataset"],
                        "depth": depth,
                        "arm": arm,
                        "expected_queries": manifest["query_count_per_dataset"],
                        "completed": len(eligible),
                        "failed": len(all_rows) - len(eligible),
                        "missing": manifest["query_count_per_dataset"] - len(all_rows),
                        "successful_query_latency_seconds": percentiles(
                            [r["query_latency_seconds"] for r in eligible]
                        ),
                        "serial_query_throughput_per_second": len(eligible)
                        / total_seconds
                        if total_seconds
                        else None,
                        "ndcg@10_completed_only": sum(
                            r["metrics"]["ndcg@10"] for r in eligible
                        )
                        / len(eligible)
                        if eligible
                        else None,
                        "attempts": sum(r.get("attempts", 0) for r in all_rows),
                        "unsuccessful_attempts": sum(
                            r.get("unsuccessful_attempts", 0) for r in all_rows
                        ),
                        "input_tokens": sum(r.get("input_tokens", 0) for r in all_rows),
                        "output_tokens": sum(
                            r.get("output_tokens", 0) for r in all_rows
                        ),
                        "uncertain_reserved_usd": sum(
                            r.get("uncertain_reserved_usd", 0) for r in all_rows
                        ),
                        "estimated_measured_query_cost_usd": sum(
                            r.get("estimated_cost_usd", 0) for r in all_rows
                        )
                        if arm.startswith("jev")
                        else None,
                    }
                )
            baseline = {
                r["query_id"]: r
                for r in rows
                if r["dataset"] == dataset["dataset"]
                and r["depth"] == depth
                and r["arm"] == "jev_sequential"
                and r["status"] == "complete"
            }
            for arm in ("qwen", "bge", "jev_concurrent", "jev_packed"):
                challenger = {
                    r["query_id"]: r
                    for r in rows
                    if r["dataset"] == dataset["dataset"]
                    and r["depth"] == depth
                    and r["arm"] == arm
                    and r["status"] == "complete"
                }
                common = set(baseline) & set(challenger)
                if common:
                    pairs = {
                        q: (
                            baseline[q]["metrics"]["ndcg@10"],
                            challenger[q]["metrics"]["ndcg@10"],
                        )
                        for q in common
                    }
                    comparisons.append(
                        {
                            "dataset": dataset["dataset"],
                            "depth": depth,
                            "baseline": "jev_sequential",
                            "challenger": arm,
                            "expected_queries": manifest["query_count_per_dataset"],
                            "paired_queries": len(common),
                            "complete_pairing": len(common)
                            == manifest["query_count_per_dataset"],
                            "exploratory": True,
                            **paired_comparison(
                                {dataset["dataset"]: pairs}, seed=manifest["seed"]
                            ),
                        }
                    )
    write_json(output / "summary.json", summary)
    write_json(output / "paired.json", comparisons)
    write_csv(output / "paired.csv", comparisons)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "per_query.csv", rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--input-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiments", nargs="+", default=["core_scifact_v1", "core_nfcorpus_v1"]
    )
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--depths", nargs="+", type=int, default=[25, 100])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pack-size", type=int, default=5)
    parser.add_argument("--requests-per-second", type=float, default=5)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cost-usd", type=Decimal, default=Decimal("5"))
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(
            args.input_root,
            args.output,
            args.experiments,
            args.queries,
            args.depths,
            args.seed,
            args.workers,
            args.pack_size,
            args.requests_per_second,
        )
        print(
            f"Frozen {args.queries} queries/dataset; {manifest['unretried_scoring_requests']} Jev scoring requests before retries/warmups"
        )
    elif args.command == "summarize":
        summarize(args.output)
    else:
        if not args.max_cost_usd.is_finite() or args.max_cost_usd <= 0:
            raise ValueError("Max cost must be finite and positive")
        manifest, archives = verify_manifest(args.input_root, args.output)
        settings = (
            Settings.load() if any(a.startswith("jev") for a in args.arms) else None
        )
        with scan_lock(args.output):
            capture = hardware()
            host_identity = digest(capture)
            hardware_path = args.output / "hardware.json"
            if hardware_path.exists() and read_json(hardware_path) != capture:
                raise ValueError("Hardware changed; use a separate output directory")
            write_json(hardware_path, capture)
            if settings:
                api_identity = {
                    "model": settings.model,
                    "input_usd_per_million_tokens": str(
                        settings.input_usd_per_million_tokens
                    ),
                }
                identity_path = args.output / "api_identity.json"
                if identity_path.exists() and read_json(identity_path) != api_identity:
                    raise ValueError(
                        "API model or configured price changed; use a separate benchmark output"
                    )
                write_json(identity_path, api_identity)
            write_json(
                args.output / f"runner_{host_identity[:12]}.json",
                {
                    "runner_sha256": file_digest(Path(__file__)),
                    "manifest": manifest["identity"],
                    "api_model": settings.model if settings else None,
                    "input_usd_per_million_tokens": str(
                        settings.input_usd_per_million_tokens
                    )
                    if settings
                    else None,
                    "max_cost_usd": str(args.max_cost_usd),
                    "timed_at": now(),
                },
            )
            datasets = {d["experiment"]: d for d in manifest["datasets"]}
            try:
                for experiment, depth, arm in manifest["job_order"]:
                    if arm not in args.arms:
                        continue
                    dataset, archive = datasets[experiment], archives[experiment]
                    if all(
                        (
                            args.output
                            / "queries"
                            / f"{dataset['dataset']}-{depth}-{arm}-{q}.json"
                        ).exists()
                        and read_json(
                            args.output
                            / "queries"
                            / f"{dataset['dataset']}-{depth}-{arm}-{q}.json"
                        )["status"]
                        == "complete"
                        for q in dataset["query_ids"]
                    ):
                        continue
                    if arm.startswith("jev"):
                        jev_job(
                            archive,
                            dataset,
                            depth,
                            arm,
                            manifest,
                            args.output,
                            settings,
                            args.max_cost_usd,
                        )
                    else:
                        local_job(
                            archive,
                            dataset,
                            depth,
                            arm,
                            manifest,
                            args.output,
                            args.device,
                        )
            finally:
                summarize(args.output)
                print(f"Results saved in {args.output}", flush=True)


if __name__ == "__main__":
    main()
