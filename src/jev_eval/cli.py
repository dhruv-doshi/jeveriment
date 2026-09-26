"""Commands for the production RAG experiment. No live work at import time."""

import argparse
import json

from .rag.config import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Pluggable RAG reranker evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    rag = sub.add_parser("rag", help="Production RAG experiment")
    rag_sub = rag.add_subparsers(dest="rag_command", required=True)
    for name in ("validate", "plan"):
        command = rag_sub.add_parser(name)
        command.add_argument("config")
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
