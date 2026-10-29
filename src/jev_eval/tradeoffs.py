"""Replay smaller retrieval pools and audit archived Jev usage without API calls."""

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from .contracts import Pool
from .io import digest, file_digest, read_json, write_json
from .metrics import evaluate_query, rank
from .stats import holm, paired_comparison


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "samples": len(ordered),
        "mean": sum(ordered) / len(ordered),
        **{
            f"p{p}": ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]
            for p in (50, 95, 99)
        },
    }


def load_archive(root, experiment):
    directory = Path(root) / "runs" / experiment
    config = read_json(directory / "resolved_config.json")
    if config["jev_mode"] != "independent":
        raise ValueError(
            "Depth replay requires independent scoring, not contextual scores"
        )
    pools = {
        q: Pool.model_validate(value)
        for q, value in read_json(directory / "pools.json").items()
    }
    queries = read_json(directory / "queries.json")
    qrels = read_json(directory / "qrels.json")
    views = read_json(directory / "text_views.json")
    sources = read_json(directory / "source_runs.json")
    if set(pools) != set(queries) or not set(pools) <= set(qrels):
        raise ValueError("Archive query IDs do not agree")
    scores = {}
    for system in ("qwen", "bge", "jev"):
        saved = read_json(directory / f"scores_{system}.json")
        if saved["status"] != "complete" or set(saved["scores"]) != set(pools):
            raise ValueError(f"{system} is not complete on every archived query")
        scores[system] = saved["scores"]
    for q, pool in pools.items():
        ids = [c.doc_id for c in pool.candidates]
        for system in scores:
            rank(scores[system][q], ids)
        for c in pool.candidates:
            view = views[c.doc_id]
            if view["hash"] != digest(view["text"]) or c.text_view_hash != view["hash"]:
                raise ValueError("Frozen candidate text has changed")
    return {
        "directory": directory,
        "config": config,
        "pools": pools,
        "queries": queries,
        "qrels": qrels,
        "views": views,
        "sources": sources,
        "scores": scores,
    }


def replay(archive, depths):
    rows = []
    config = archive["config"]
    for q, pool in archive["pools"].items():
        ids = [c.doc_id for c in pool.candidates]
        incoming = {d: archive["sources"]["rrf"][q][d] for d in ids}
        ordered = rank(incoming)
        for depth in depths:
            if len(ordered) < depth:
                raise ValueError(f"Query {q} has fewer than {depth} candidates")
            selected = ordered[:depth]
            for system in ("rrf", "qwen", "bge", "jev"):
                scores = incoming if system == "rrf" else archive["scores"][system][q]
                metrics = evaluate_query(
                    rank({d: scores[d] for d in selected}),
                    archive["qrels"][q],
                    selected,
                    config["binary_threshold"],
                    config["gain"],
                )
                rows.append(
                    {
                        "dataset": config["dataset"],
                        "experiment": config["experiment_id"],
                        "query_id": q,
                        "depth": depth,
                        "system": system,
                        **metrics,
                    }
                )
    return rows


def summarize_rows(rows, seed=1729, bootstrap=10000, permutations=100000):
    summary, paired, examples = [], [], []
    for dataset in sorted({r["dataset"] for r in rows}):
        largest_depth = max(r["depth"] for r in rows if r["dataset"] == dataset)
        for depth in sorted({r["depth"] for r in rows if r["dataset"] == dataset}):
            selected = [
                r for r in rows if r["dataset"] == dataset and r["depth"] == depth
            ]
            systems = {}
            for system in ("rrf", "qwen", "bge", "jev"):
                eligible = [
                    r for r in selected if r["system"] == system and r["eligible"]
                ]
                systems[system] = {r["query_id"]: r for r in eligible}
                if not eligible:
                    raise ValueError("No eligible queries")
                summary.append(
                    {
                        "dataset": dataset,
                        "depth": depth,
                        "system": system,
                        "queries": len(eligible),
                        "candidate_evaluations": depth * len(eligible),
                        **{
                            key: float(np.mean([r[key] for r in eligible]))
                            for key in (
                                "ndcg@10",
                                "recall@10",
                                "mrr@10",
                                "candidate_recall",
                                "oracle_ndcg@10",
                            )
                        },
                    }
                )
            for baseline in ("rrf", "qwen", "bge"):
                if set(systems[baseline]) != set(systems["jev"]):
                    raise ValueError("Paired comparison has different eligible queries")
                pairs = {
                    q: (systems[baseline][q]["ndcg@10"], systems["jev"][q]["ndcg@10"])
                    for q in systems[baseline]
                }
                result = paired_comparison(
                    {dataset: pairs},
                    seed=seed,
                    bootstrap=bootstrap,
                    permutations=permutations,
                )
                paired.append(
                    {
                        "dataset": dataset,
                        "depth": depth,
                        "baseline": baseline,
                        "challenger": "jev",
                        "metric": "ndcg@10",
                        **result,
                    }
                )
                if baseline == "qwen" and depth == largest_depth:
                    for q, (base, challenge) in pairs.items():
                        examples.append(
                            {
                                "dataset": dataset,
                                "query_id": q,
                                "qwen_ndcg": base,
                                "jev_ndcg": challenge,
                                "delta": challenge - base,
                            }
                        )
    corrected = holm({str(i): r["p_value"] for i, r in enumerate(paired)})
    for i, result in enumerate(paired):
        result["holm_p_value_all_replay_comparisons"] = corrected[str(i)]
        result["exploratory"] = True
    return summary, paired, examples


def archived_path(root, value):
    parts = Path(value).parts
    if "runs" not in parts:
        raise ValueError("Ledger raw path is outside the archived runs directory")
    path = Path(root).resolve() / Path(*parts[parts.index("runs") :])
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Ledger raw path escapes the archive")
    return path


def audit_usage(root, experiment):
    ledger = Path(root).resolve() / "runs" / "budget_typesafe.sqlite"
    if Path(str(ledger) + "-wal").exists():
        raise ValueError("Use a checkpointed SQLite export before offline auditing")
    with sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute(
            "SELECT id, started_at, status, actual_usd, reserved_usd, tokens, raw_path FROM attempts ORDER BY id"
        ).fetchall()
    matching = []
    for row in rows:
        if row[6] and tuple(Path(row[6]).parts[-4:-2]) == ("runs", experiment):
            matching.append(row)
    # Match a relative or absolute path, while keeping scan/preflight subruns separate.
    latencies, failures, missing = [], [], []
    for row in matching:
        path = archived_path(root, row[6])
        if not path.exists():
            missing.append(str(path))
            continue
        raw = read_json(path)
        latency = raw.get("elapsed_seconds")
        if (
            isinstance(latency, (int, float))
            and math.isfinite(latency)
            and latency >= 0
        ):
            latencies.append(latency)
        if row[2] != "success":
            failures.append(
                {
                    "attempt_id": row[0],
                    "status": row[2],
                    "http_status": raw.get("status_code"),
                    "timestamp": row[1],
                }
            )
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} ledger responses")
    directory = Path(root) / "runs" / experiment
    queries = read_json(directory / "queries.json")
    charges = sum((Decimal(r[3]) for r in matching if r[3] is not None), Decimal(0))
    uncertain = sum((Decimal(r[4]) for r in matching if r[3] is None), Decimal(0))
    invocations = []
    for path in sorted((directory / "logs").glob("rerank_jev_*.log")):
        text = path.read_text()
        invocations.append(
            {
                "log": path.name,
                "completed_query_messages": text.count("jev: completed query"),
                "completed": '"status": "complete"' in text,
            }
        )
    resources = {
        s: read_json(directory / f"resources_rerank_{s}.json")
        for s in ("jev", "qwen", "bge")
    }
    return {
        "experiment": experiment,
        "attempts": len(matching),
        "attempt_statuses": dict(Counter(r[2] for r in matching)),
        "estimated_known_cost_usd": str(charges),
        "uncertain_reserved_usd": str(uncertain),
        "estimated_known_cost_per_query_usd": str(charges / len(queries)),
        "reported_tokens_input_plus_output": sum(r[5] or 0 for r in matching),
        "first_attempt_at": matching[0][1] if matching else None,
        "last_attempt_at": matching[-1][1] if matching else None,
        "api_attempt_latency_seconds": percentiles(latencies),
        "failures": failures,
        "invocations": invocations,
        "last_saved_resources": resources,
        "timing_scope": "API attempt latency excludes queueing and backoff; resources may cover only the final resumed invocation",
        "cost_scope": "Configured price times provider input usage, across all archived attempts; not an invoice. Reservations are separate.",
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def figures(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in summary})
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets):
        for system in ("rrf", "qwen", "bge", "jev"):
            rows = [
                r for r in summary if r["dataset"] == dataset and r["system"] == system
            ]
            ax.plot(
                [r["depth"] for r in rows],
                [r["ndcg@10"] for r in rows],
                "o-",
                label=system,
            )
        ax.set(title=dataset, xlabel="Candidates scored per query", ylabel="nDCG@10")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Exploratory replay of frozen independent scores")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"quality_by_depth.{extension}", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset in datasets:
        rows = [r for r in summary if r["dataset"] == dataset and r["system"] == "jev"]
        ax.plot(
            [r["depth"] for r in rows],
            [r["candidate_recall"] for r in rows],
            "o-",
            label=dataset,
        )
    ax.set(
        xlabel="Candidate depth",
        ylabel="Known relevant candidate recall",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"candidate_recall.{extension}", dpi=170)
    plt.close(fig)


def report(output, summary, paired, usage, examples):
    lines = [
        "# Jev quality and operating cost analysis\n",
        "Exploratory held-out replay on SciFact and NFCorpus. Candidate pools are prefixes of the original RRF ranking; all systems receive identical IDs and frozen text at each depth. Scores are reused from independent scoring. No API requests were made.\n",
        "## Ranking quality by candidate depth\n",
        "| Dataset | Depth | RRF | Qwen | BGE | Jev | Candidate recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in sorted({r["dataset"] for r in summary}):
        for depth in sorted({r["depth"] for r in summary}):
            rows = {
                r["system"]: r
                for r in summary
                if r["dataset"] == dataset and r["depth"] == depth
            }
            lines.append(
                f"| {dataset} | {depth} | "
                + " | ".join(
                    f"{rows[s]['ndcg@10']:.4f}" for s in ("rrf", "qwen", "bge", "jev")
                )
                + f" | {rows['jev']['candidate_recall']:.4f} |"
            )
    lines += [
        "\n## Paired Jev versus Qwen comparison\n",
        "Intervals are paired query bootstrap intervals; p-values are unadjusted unless labelled Holm. The CSV also contains comparisons with RRF and BGE, with Holm correction over the complete replay comparison family. Depth selection after this replay remains exploratory.\n",
        "| Dataset | Depth | Delta | 95% interval | p | Holm p | Wins ties losses |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for r in paired:
        if r["baseline"] == "qwen":
            lines.append(
                f"| {r['dataset']} | {r['depth']} | {r['macro_delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['p_value']:.4f} | {r['holm_p_value_all_replay_comparisons']:.4f} | {r['wins']} / {r['ties']} / {r['losses']} |"
            )
    lines += [
        "\n## Archived API cost and attempt timing\n",
        "| Run | Attempts | Estimated known USD | Uncertain reserved USD | Known USD per query | Attempt p50 seconds | Attempt p95 seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for u in usage:
        latency = u["api_attempt_latency_seconds"]
        lines.append(
            f"| {u['experiment']} | {u['attempts']} | {float(u['estimated_known_cost_usd']):.6f} | {float(u['uncertain_reserved_usd']):.6f} | {float(u['estimated_known_cost_per_query_usd']):.6f} | {latency['p50']:.3f} | {latency['p95']:.3f} |"
        )
    lines += [
        "\nCosts use the ledger's configured price and reported usage across all saved attempts, including retries. They exclude VM rental, retrieval, and generation. Smaller-depth monetary cost and query latency have not been measured by this replay. Candidate evaluations are a workload measure, not a measured dollar saving.\n",
        "## Timing limitations\n",
        "SciFact has multiple Jev invocations; its saved resource JSON describes only the last invocation. API attempt percentiles exclude queueing, retries between attempts, and cold starts. No complete-query latency distribution was saved in the original runs. Hosted Jev and locally served Qwen/BGE are different deployments; runtime ratios do not isolate model efficiency. Failure records and invocation counts are retained in usage.json.\n",
        "## Examples for manual review\n",
        "The examples below are selected extremes at the largest replay depth, not a representative sample. Read the frozen documents and qrels before using them as explanations. Complete per-query rows are in per_query.csv.\n",
        "| Dataset | Query | Jev minus Qwen nDCG |",
        "| --- | --- | ---: |",
    ]
    for dataset in sorted({r["dataset"] for r in examples}):
        selected = sorted(
            (r for r in examples if r["dataset"] == dataset),
            key=lambda r: (r["delta"], r["query_id"]),
        )
        for r in selected[:3] + selected[-3:]:
            lines.append(f"| {dataset} | {r['query_id']} | {r['delta']:+.4f} |")
    lines += [
        "\n## Proposed next experiment\n",
        "Use the separately frozen efficiency manifest for a bounded comparison of sequential, concurrent, and packed Jev calls at depths 25 and 100. Collect complete-query timings, errors, live usage and ranking quality. A 30-query sample is a feasibility comparison with limited tail precision; p99 needs a larger sustained run. Packed requests change the placement of document text and must be checked for quality. FiQA is outside this analysis.\n",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiments", nargs="+", default=["core_scifact_v1", "core_nfcorpus_v1"]
    )
    parser.add_argument("--depths", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=100000)
    args = parser.parse_args()
    if any(d < 1 for d in args.depths) or len(set(args.depths)) != len(args.depths):
        raise ValueError("Depths must be distinct positive integers")
    if args.output.resolve().is_relative_to(args.input_root.resolve()):
        raise ValueError("Analysis output must be outside the original archive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, usage, provenance = [], [], []
    for experiment in args.experiments:
        archive = load_archive(args.input_root, experiment)
        rows.extend(replay(archive, sorted(args.depths)))
        usage.append(audit_usage(args.input_root, experiment))
        provenance.append(
            {
                "experiment": experiment,
                "inputs": {
                    p.name: file_digest(p)
                    for p in archive["directory"].glob("*.json")
                    if p.name
                    in {
                        "pools.json",
                        "qrels.json",
                        "resolved_config.json",
                        "scores_qwen.json",
                        "scores_bge.json",
                        "scores_jev.json",
                        "source_runs.json",
                        "text_views.json",
                        "queries.json",
                    }
                },
            }
        )
    summary, paired, examples = summarize_rows(
        rows, bootstrap=args.bootstrap, permutations=args.permutations
    )
    for name, values in (
        ("summary", summary),
        ("paired", paired),
        ("per_query", rows),
        ("examples", examples),
    ):
        write_csv(args.output / f"{name}.csv", values)
        write_json(args.output / f"{name}.json", values)
    write_json(args.output / "usage.json", usage)
    write_json(
        args.output / "provenance.json",
        {
            "input_root": str(args.input_root.resolve()),
            "inputs": provenance,
            "ledger_sha256": file_digest(
                args.input_root / "runs" / "budget_typesafe.sqlite"
            ),
            "analysis_sha256": file_digest(Path(__file__)),
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
        },
    )
    figures(args.output, summary)
    report(args.output, summary, paired, usage, examples)
    print(f"Wrote offline analysis to {args.output}")


if __name__ == "__main__":
    main()
