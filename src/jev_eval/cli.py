"""Commands for the production RAG experiment. No live work at import time."""

import argparse
import json
from pathlib import Path

from .rag.config import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Pluggable RAG reranker evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    rag = sub.add_parser("rag", help="Production RAG experiment")
    rag_sub = rag.add_subparsers(dest="rag_command", required=True)
    for name in ("validate", "plan"):
        command = rag_sub.add_parser(name)
        command.add_argument("config")
    audit = rag_sub.add_parser("audit-sources")
    audit.add_argument("config")
    audit.add_argument("--source-root", default="data/sources")
    prepare = rag_sub.add_parser("prepare")
    prepare.add_argument("config")
    prepare.add_argument("dataset")
    prepare.add_argument("--source-root", default="data/sources")
    prepare.add_argument("--run-root", default="runs")
    dense = rag_sub.add_parser("index-dense")
    dense.add_argument("config")
    dense.add_argument("dataset")
    dense.add_argument("--run-root", default="runs")
    dense.add_argument("--revisions-json", required=True)
    dense.add_argument("--batch-size", type=int, default=32)
    run = rag_sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("dataset")
    run.add_argument("retriever")
    run.add_argument("reranker")
    run.add_argument("--run-root", default="runs")
    run.add_argument("--split", default="development")
    run.add_argument("--limit", type=int, default=5)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--generate", action="store_true")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--provider-only")
    run.add_argument("--revisions-json")
    run.add_argument("--limits-json")
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    if args.rag_command == "validate":
        print(experiment.model_dump_json(indent=2))
    elif args.rag_command == "plan":
        print(
            json.dumps(
                {
                    "experiment_id": experiment.experiment_id,
                    "datasets": [d.id for d in experiment.datasets],
                    "retrievers": [r.id for r in experiment.retrievers],
                    "rerankers": [r.id for r in experiment.rerankers],
                    "ranking_cells": (
                        len(experiment.datasets)
                        * len(experiment.retrievers)
                        * len(experiment.rerankers)
                    ),
                    "generator": experiment.generator.model,
                    "budget_usd": str(experiment.budget_usd),
                    "execution": "plan_only",
                },
                indent=2,
            )
        )
    elif args.rag_command == "audit-sources":
        from .rag.datasets import adapter_for

        print(json.dumps([adapter_for(item.id, args.source_root).audit() for item in experiment.datasets], indent=2))
    elif args.rag_command == "prepare":
        from .rag.experiment import prepare_dataset

        print(json.dumps(prepare_dataset(experiment, args.dataset, args.source_root, args.run_root), indent=2))
    elif args.rag_command == "index-dense":
        from .rag.experiment import ExperimentRuntime

        revisions = json.loads(Path(args.revisions_json).read_text())
        runtime = ExperimentRuntime(experiment, args.run_root, args.dataset, revisions)
        print(json.dumps(runtime.build_dense(args.batch_size), indent=2))
    elif args.rag_command == "run":
        if args.limit < 1 or args.workers < 1:
            parser.error("--limit and --workers must be positive")
        forecast = {
            "dataset": args.dataset, "retriever": args.retriever, "reranker": args.reranker,
            "split": args.split, "query_limit": args.limit, "workers": args.workers,
            "max_jev_calls": args.limit * experiment.candidate_depth if args.reranker == "jev" else 0,
            "max_generation_calls": args.limit if args.generate else 0,
            "budget_usd": str(experiment.budget_usd), "execution": "live" if args.execute else "dry_run",
        }
        if not args.execute:
            print(json.dumps(forecast, indent=2))
            return
        from .rag.experiment import ExperimentRuntime

        revisions = json.loads(Path(args.revisions_json).read_text()) if args.revisions_json else {}
        limits = json.loads(Path(args.limits_json).read_text()) if args.limits_json else {}
        if args.generate and not args.provider_only:
            parser.error("--provider-only is required for a controlled generator run")
        if args.reranker == "jev" and "typesafe" not in limits:
            parser.error("--limits-json needs account-specific typesafe limits and prices")
        if args.generate and "openrouter" not in limits:
            parser.error("--limits-json needs account-specific openrouter limits and prices")
        from decimal import Decimal

        from .ledger import Ledger

        upper = Decimal(0)
        for provider, calls, input_bound, output_bound in (
            ("typesafe", forecast["max_jev_calls"], 32000, 64),
            ("openrouter", forecast["max_generation_calls"], 20000, experiment.generator.max_output_tokens),
        ):
            if calls:
                item = limits[provider]
                upper += calls * (
                    Decimal(str(item["input_usd_per_million"])) * input_bound
                    + Decimal(str(item["output_usd_per_million"])) * output_bound
                ) / Decimal(1_000_000)
        budget_path = Path(args.run_root) / experiment.experiment_id / "budget.sqlite"
        spent = Decimal(Ledger(budget_path).totals()["cost_usd"]) if budget_path.exists() else Decimal(0)
        forecast["conservative_api_upper_usd"] = str(upper)
        forecast["already_reserved_or_spent_usd"] = str(spent)
        if spent + upper > experiment.budget_usd:
            parser.error("Conservative API forecast exceeds remaining global budget")
        runtime = ExperimentRuntime(experiment, args.run_root, args.dataset,
                                    revisions, args.provider_only, limits)
        outcome = runtime.run_slice(retriever=args.retriever, reranker=args.reranker,
                                    split=args.split, limit=args.limit,
                                    generate=args.generate, workers=args.workers)
        print(json.dumps({**forecast, **outcome}, indent=2))
