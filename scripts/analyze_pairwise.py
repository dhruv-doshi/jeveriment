"""Compare two completed rerankers on the same saved queries."""

import argparse
import json
import os
import re
from pathlib import Path

from jev_eval.config import load_config
from jev_eval.io import write_json
from jev_eval.pipeline import run_dir
from jev_eval.stats import paired_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("CONFIG", "configs/pilot.yaml"))
    parser.add_argument("--baseline", default="qwen")
    parser.add_argument("--challenger", default="jev")
    parser.add_argument("--metric", default="ndcg@10")
    args = parser.parse_args()

    config = load_config(args.config)
    directory = run_dir(config)
    rows = json.loads((directory / "per_query_metrics.json").read_text())
    scores = {args.baseline: {}, args.challenger: {}}
    if args.baseline == args.challenger:
        raise ValueError("Choose two different systems")
    for row in rows:
        system = row["system"]
        if system not in scores or not row["eligible"]:
            continue
        query_id = row["query_id"]
        if query_id in scores[system]:
            raise ValueError(f"Duplicate {system} query: {query_id}")
        scores[system][query_id] = row[args.metric]
    baseline, challenger = scores[args.baseline], scores[args.challenger]
    if not baseline or set(baseline) != set(challenger):
        raise ValueError("Systems need the same nonempty set of eligible queries")

    pairs = {q: [baseline[q], challenger[q]] for q in baseline}
    result = {
        "experiment_id": config.experiment_id,
        "dataset": config.dataset,
        "split": config.split,
        "confirmatory": config.confirmatory,
        "baseline": args.baseline,
        "challenger": args.challenger,
        "metric": args.metric,
        **paired_comparison({config.dataset: pairs}),
    }
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", args.metric)
    output = directory / f"paired_{args.baseline}_vs_{args.challenger}_{label}.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
