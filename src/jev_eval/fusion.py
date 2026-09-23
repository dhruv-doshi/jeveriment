import itertools
import math

from .metrics import evaluate_query, rank


def minmax(scores):
    if not scores or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("Invalid feature values")
    lo, hi = min(scores.values()), max(scores.values())
    return {d: (v - lo) / (hi - lo) if hi > lo else 0.0 for d, v in scores.items()}


def weighted(features, weights):
    if (
        set(features) != set(weights)
        or any(not math.isfinite(w) or w < 0 for w in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-9
    ):
        raise ValueError("Weights must form a simplex over the supplied features")
    normalized = {f: minmax(s) for f, s in features.items()}
    ids = set(next(iter(normalized.values())))
    if any(set(s) != ids for s in normalized.values()):
        raise ValueError("Fusion membership mismatch")
    return {d: sum(weights[f] * s[d] for f, s in normalized.items()) for d in ids}


def simplex_grid(names, steps=10):
    for values in itertools.product(range(steps + 1), repeat=len(names)):
        if sum(values) == steps:
            yield dict(zip(names, (v / steps for v in values)))


def tune(
    features_by_query,
    judgments,
    splits,
    test_query_ids=(),
    threshold=1,
    gain="linear",
    trial_budget=None,
    seed=1729,
):
    ids = set(features_by_query)
    if ids & set(test_query_ids) or any(
        splits.get(q) not in {"dev", "train"} for q in ids
    ):
        raise ValueError(
            "Fusion fitting requires disjoint development/training queries"
        )
    if not ids or set(judgments) != ids:
        raise ValueError("Missing tuning judgments")
    names = sorted(next(iter(features_by_query.values())))
    best, trials = None, []
    grid = list(simplex_grid(names))
    grid_size = len(grid)
    if trial_budget is not None:
        if trial_budget < 1 or trial_budget > grid_size:
            raise ValueError("Trial budget must fit available simplex grid")
        if trial_budget < grid_size:
            import random

            grid = random.Random(seed).sample(grid, trial_budget)
    for weights in grid:
        values = []
        for q, features in features_by_query.items():
            metrics = evaluate_query(
                rank(weighted(features, weights)),
                judgments[q],
                threshold=threshold,
                gain=gain,
            )
            if metrics["eligible"]:
                values.append(metrics["ndcg@10"])
        if not values:
            raise ValueError("No eligible tuning queries")
        value = sum(values) / len(values)
        trials.append({"weights": weights, "mean_ndcg@10": value})
        if best is None or value > best[0]:
            best = value, weights
    return {
        "weights": best[1],
        "objective": best[0],
        "trials": trials,
        "fit_query_ids": sorted(ids),
        "frozen": True,
        "grid_size": grid_size,
        "evaluated_configurations": len(trials),
        "seed": seed,
    }


def tune_matched(
    conditions, judgments, splits, test_query_ids=(), threshold=1, gain="linear"
):
    if len(conditions) < 2:
        raise ValueError("Matched tuning needs at least two conditions")
    query_ids = set(next(iter(conditions.values())))
    if any(set(condition) != query_ids for condition in conditions.values()):
        raise ValueError("Tuning conditions must share query membership")
    budget = min(
        len(list(simplex_grid(sorted(next(iter(condition.values()))))))
        for condition in conditions.values()
    )
    return {
        name: tune(
            features,
            judgments,
            splits,
            test_query_ids,
            threshold,
            gain,
            trial_budget=budget,
        )
        for name, features in conditions.items()
    }
