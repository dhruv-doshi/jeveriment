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
    run.add_argument("--judge", action="store_true")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--provider-only")
    run.add_argument("--revisions-json")
    run.add_argument("--limits-json")
    matrix = rag_sub.add_parser("matrix", help="Run matched cells with shared caches and a global budget")
    matrix.add_argument("config")
    matrix.add_argument("--datasets", nargs="+")
    matrix.add_argument("--retrievers", nargs="+")
    matrix.add_argument("--rerankers", nargs="+")
    matrix.add_argument("--run-root", default="runs")
    matrix.add_argument("--split", default="development")
    matrix.add_argument("--limit", type=int, default=5)
    matrix.add_argument("--workers", type=int, default=4)
    matrix.add_argument("--generate", action="store_true")
    matrix.add_argument("--judge", action="store_true")
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--provider-only")
    matrix.add_argument("--revisions-json")
    matrix.add_argument("--limits-json")
    report = rag_sub.add_parser("report")
    report.add_argument("config")
    report.add_argument("results", nargs="+")
    report.add_argument("--run-root", default="runs")
    report.add_argument("--output", required=True)
    comparison = rag_sub.add_parser("compare")
    comparison.add_argument("config")
    comparison.add_argument("--baseline", nargs="+", required=True)
    comparison.add_argument("--challenger", nargs="+", required=True)
    comparison.add_argument("--metric", required=True)
    comparison.add_argument("--output", required=True)
    fake_load = rag_sub.add_parser("load-fake")
    fake_load.add_argument("config")
    fake_load.add_argument("--run-root", default="runs")
    fake_load.add_argument("--requests", type=int, default=100)
    fake_load.add_argument("--workers", type=int, default=8)
    fake_load.add_argument("--latency-ms", type=int, default=20)
    fake_load.add_argument("--every-429", type=int, default=0)
    serve = rag_sub.add_parser("serve")
    serve.add_argument("config")
    serve.add_argument("--run-root", default="runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--revisions-json")
    serve.add_argument("--limits-json")
    serve.add_argument("--provider-only")
    serve.add_argument("--allow-paid", action="store_true")
    traces = rag_sub.add_parser("export-traces")
    traces.add_argument("config")
    traces.add_argument("--run-root", default="runs")
    traces.add_argument("--output", required=True)
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
    elif args.rag_command in {"run", "matrix"}:
        if args.limit < 1 or args.workers < 1:
            parser.error("--limit and --workers must be positive")
        if args.judge and not args.generate:
            parser.error("--judge requires --generate")
        datasets = [args.dataset] if args.rag_command == "run" else (args.datasets or [item.id for item in experiment.datasets])
        retrievers = [args.retriever] if args.rag_command == "run" else (args.retrievers or [item.id for item in experiment.retrievers])
        rerankers = [args.reranker] if args.rag_command == "run" else (args.rerankers or [item.id for item in experiment.rerankers])
        for selected, configured, kind in (
            (datasets, experiment.datasets, "dataset"),
            (retrievers, experiment.retrievers, "retriever"),
            (rerankers, experiment.rerankers, "reranker"),
        ):
            if len(selected) != len(set(selected)) or set(selected) - {item.id for item in configured}:
                parser.error(f"Invalid or duplicate {kind} selection")
        cells = len(datasets) * len(retrievers) * len(rerankers)
        forecast = {
            "datasets": datasets, "retrievers": retrievers, "rerankers": rerankers, "cells": cells,
            "split": args.split, "query_limit": args.limit, "workers": args.workers,
            "max_jev_calls": args.limit * experiment.candidate_depth * len(datasets) * len(retrievers) if "jev" in rerankers else 0,
            "max_generation_calls": args.limit * cells if args.generate else 0,
            "max_judge_calls": args.limit * cells if args.judge else 0,
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
        if "jev" in rerankers and "typesafe" not in limits:
            parser.error("--limits-json needs account-specific typesafe limits and prices")
        if args.generate and "openrouter" not in limits:
            parser.error("--limits-json needs account-specific openrouter limits and prices")
        from decimal import Decimal

        from .ledger import Ledger

        upper = Decimal(0)
        for provider, calls, input_bound, output_bound in (
            ("typesafe", forecast["max_jev_calls"], 32000, 64),
            ("openrouter", forecast["max_generation_calls"], 20000, experiment.generator.max_output_tokens),
            ("openrouter", forecast["max_judge_calls"], 20000, 256),
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
        from .rag.telemetry import setup_tracing

        setup_tracing(Path(args.run_root) / experiment.experiment_id)
        outcomes = []
        for dataset in datasets:
            runtime = ExperimentRuntime(experiment, args.run_root, dataset,
                                        revisions, args.provider_only, limits)
            try:
                for retriever in retrievers:
                    for reranker in rerankers:
                        outcome = runtime.run_slice(retriever=retriever, reranker=reranker,
                                                    split=args.split, limit=args.limit,
                                                    generate=args.generate, judge=args.judge,
                                                    workers=args.workers)
                        outcomes.append({"dataset": dataset, "retriever": retriever,
                                         "reranker": reranker, **outcome})
            finally:
                runtime.close()
        print(json.dumps({**forecast, "results": outcomes}, indent=2))
    elif args.rag_command == "report":
        from .rag.reporting import write_report

        run_root = Path(args.run_root) / experiment.experiment_id
        result = write_report(args.results, run_root, args.output)
        print(json.dumps({"output": args.output, "cells": len(result["quality"]),
                          "budget": result["budget"]}, indent=2))
    elif args.rag_command == "compare":
        from .io import write_json
        from .rag.reporting import compare

        result = compare(args.baseline, args.challenger, args.metric)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.rag_command == "load-fake":
        from .rag.load import run_fake_load

        root = Path(args.run_root) / experiment.experiment_id / "fake_load"
        result = run_fake_load(root, requests=args.requests, workers=args.workers,
                               latency_ms=args.latency_ms, every_429=args.every_429)
        print(json.dumps(result, indent=2))
    elif args.rag_command == "serve":
        import os

        import uvicorn
        from dotenv import load_dotenv

        from .rag.service import create_app

        load_dotenv()
        revisions = json.loads(Path(args.revisions_json).read_text()) if args.revisions_json else {}
        limits = json.loads(Path(args.limits_json).read_text()) if args.limits_json else {}
        app = create_app(experiment, args.run_root, api_key=os.getenv("SERVICE_API_KEY", ""),
                         revisions=revisions, limits=limits, provider_only=args.provider_only,
                         allow_paid=args.allow_paid)
        from .rag.telemetry import setup_tracing

        setup_tracing(Path(args.run_root) / experiment.experiment_id)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.rag_command == "export-traces":
        from .rag.reporting import export_trace_parquet

        print(json.dumps(export_trace_parquet(Path(args.run_root) / experiment.experiment_id,
                                              args.output), indent=2))
