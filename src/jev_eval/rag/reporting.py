"""Dataset-aware quality, latency, failure, and spend summaries."""

import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from jev_eval.io import digest, read_json, write_json
from jev_eval.stats import paired_comparison


def distribution(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "p99_low_sample_warning": len(values) < 100}


def summarize(result_paths: list[str | Path], run_root: str | Path):
    cells = defaultdict(list)
    selected = set()
    for path in result_paths:
        for row in read_json(path):
            key = (row["dataset"], row["retriever"], row["reranker"])
            identity = (*key, row["query_id"])
            if identity in selected:
                raise ValueError(f"Duplicate query/cell in report inputs: {identity}")
            selected.add(identity)
            cells[key].append(row)
    quality = {}
    for key, rows in sorted(cells.items()):
        metric_values = defaultdict(list)
        for row in rows:
            for stage in ("retrieval", "ranking", "answer", "judge"):
                for name, value in row.get(stage, {}).items():
                    if type(value) in (int, float) and math.isfinite(value):
                        metric_values[f"{stage}.{name}"].append(value)
        quality["/".join(key)] = {
            "attempted": len(rows), "failed": sum(row.get("status") == "failed" for row in rows),
            "metrics": {name: {"mean": float(np.mean(values)), "eligible": len(values)}
                        for name, values in sorted(metric_values.items())},
        }
    run_root = Path(run_root)
    stage_times = defaultdict(list)
    for path in (run_root / "artifacts" / "stages").glob("*.json"):
        record = read_json(path)
        stage_times[record["stage"]].append(record["elapsed_ms"])
    api_times, queue_times, status = defaultdict(list), defaultdict(list), Counter()
    for path in (run_root / "raw").glob("*/*.json"):
        raw = read_json(path)
        provider = path.parent.name
        api_times[provider].append(raw.get("elapsed_ms"))
        queue_times[provider].append(raw.get("queue_wait_ms"))
        status[f"{provider}:{raw.get('status_code', raw.get('error', 'unknown'))}"] += 1
    provider_metadata = []
    for path in (run_root / "raw" / "openrouter_metadata").glob("*.json"):
        data = read_json(path).get("response", {}).get("data")
        if data:
            provider_metadata.append({"generation_id": data.get("id"), "provider": data.get("provider_name"),
                                      "latency_raw": data.get("latency"), "generation_time_raw": data.get("generation_time"),
                                      "timing_unit": "provider_documentation_unspecified",
                                      "total_cost_usd": data.get("total_cost")})
    budget = {"attempts": 0, "cost_usd": "0", "reported_cost_usd": "0", "uncertain_reserved_usd": "0", "tokens": 0}
    budget_path = run_root / "budget.sqlite"
    if budget_path.exists():
        with sqlite3.connect(budget_path) as db:
            row = db.execute("SELECT attempts,cost_usd,reported_cost_usd,uncertain_reserved_usd,tokens FROM budget_state WHERE id=1").fetchone()
            budget = dict(zip(budget, row))
    return {"quality": quality, "stage_latency_ms": {name: distribution(values) for name, values in stage_times.items()},
            "api_latency_ms": {name: distribution(values) for name, values in api_times.items()},
            "queue_latency_ms": {name: distribution(values) for name, values in queue_times.items()},
            "api_statuses": dict(status), "budget": budget,
            "provider_metadata": provider_metadata,
            "supplemental_judge": {"status": "reported_if_present", "rubric": "JUDGE_RUBRIC_V1"}}


def write_report(result_paths, run_root, output):
    report = summarize(result_paths, run_root)
    write_json(output, report)
    return report


def compare(baseline_paths, challenger_paths, metric, *, bootstrap=10000, permutations=100000):
    def collect(paths):
        rows = {}
        for path in paths:
            for row in read_json(path):
                key = (row["dataset"], row["retriever"], row["query_id"])
                if key in rows:
                    raise ValueError(f"Duplicate paired row: {key}")
                rows[key] = row
        return rows

    baseline, challenger = collect(baseline_paths), collect(challenger_paths)
    if baseline.keys() != challenger.keys():
        raise ValueError("Baseline and challenger need identical dataset/retriever/query IDs")
    stage, name = metric.split(".", 1)
    datasets = defaultdict(dict)
    for key in sorted(baseline):
        left, right = baseline[key], challenger[key]
        if left.get("pool_hash") != right.get("pool_hash"):
            raise ValueError(f"Candidate pool mismatch for {key}")
        a, b = left.get(stage, {}).get(name), right.get(stage, {}).get(name)
        if type(a) not in (int, float) or type(b) not in (int, float):
            raise ValueError(f"Missing paired metric {metric} for {key}")
        datasets[key[0]][key[2]] = (a, b)
    return {"metric": metric, "paired": paired_comparison(datasets, bootstrap=bootstrap, permutations=permutations)}


def export_trace_parquet(run_root: str | Path, output_dir: str | Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_root, output_dir = Path(run_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [read_json(path) for path in (run_root / "artifacts" / "stages").glob("*.json")]
    charges = {}
    budget_path = run_root / "budget.sqlite"
    if budget_path.exists():
        with sqlite3.connect(budget_path) as db:
            for raw_path, actual, reserved in db.execute("SELECT raw_path,actual_usd,reserved_usd FROM attempts"):
                charges[raw_path] = actual or reserved
    attempts = []
    for path in (run_root / "raw").glob("*/*.json"):
        raw = read_json(path)
        request, response = raw.get("request", {}), raw.get("response", {})
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        attempts.append({"provider": path.parent.name, "raw_path": str(path),
                         "provenance_query_id": (raw.get("provenance") or {}).get("query_id"),
                         "provenance_role": (raw.get("provenance") or {}).get("role"),
                         "request_hash": digest(request),
                         "status_code": raw.get("status_code"), "error": raw.get("error"),
                         "elapsed_ms": raw.get("elapsed_ms"), "queue_wait_ms": raw.get("queue_wait_ms"),
                         "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                         "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                         "cost_usd": charges.get(str(path)),
                         "model_resolved": response.get("model") if isinstance(response, dict) else None,
                         "timestamp_utc": raw.get("timestamp")})
    if stages:
        pq.write_table(pa.Table.from_pylist(stages), output_dir / "stages.parquet")
    if attempts:
        pq.write_table(pa.Table.from_pylist(attempts), output_dir / "attempts.parquet")
    return {"stages": len(stages), "attempts": len(attempts), "output_dir": str(output_dir)}
