"""Paired query inference; datasets retain equal weight in macro claims."""

import itertools
import math

import numpy as np


def holm(pvalues):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("Invalid p-value")
    adjusted, floor = {}, 0.0
    ordered = sorted(pvalues, key=lambda k: (pvalues[k], k))
    for i, key in enumerate(ordered):
        floor = max(floor, min(1.0, (len(ordered) - i) * pvalues[key]))
        adjusted[key] = floor
    return adjusted


def paired_comparison(
    datasets, seed=1729, bootstrap=10000, permutations=100000, margin=0.01
):
    """Input dataset -> query ID -> (baseline, challenger); no unpaired rows."""
    if bootstrap < 1 or permutations < 1 or not datasets:
        raise ValueError("Positive resample counts and nonempty datasets required")
    diffs = []
    for name, queries in sorted(datasets.items()):
        if not queries:
            raise ValueError(f"Empty paired dataset: {name}")
        values = np.array(
            [queries[q][1] - queries[q][0] for q in sorted(queries)], dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite paired metrics")
        diffs.append(values)
    rng = np.random.default_rng(seed)
    observed = float(np.mean([d.mean() for d in diffs]))
    boot = np.zeros(bootstrap)
    for d in diffs:
        for start in range(0, bootstrap, 256):
            n = min(256, bootstrap - start)
            boot[start : start + n] += rng.choice(d, (n, len(d)), replace=True).mean(
                axis=1
            ) / len(diffs)
    all_d = np.concatenate(diffs)
    weights = np.concatenate(
        [np.full(len(d), 1 / (len(d) * len(diffs))) for d in diffs]
    )
    weighted = all_d * weights
    if len(all_d) <= 16:
        samples = np.array(
            [
                np.dot(signs, weighted)
                for signs in itertools.product((-1, 1), repeat=len(all_d))
            ]
        )
        p = float(np.mean(np.abs(samples) >= abs(observed) - 1e-12))
        exact = True
    else:
        extremes = 0
        for start in range(0, permutations, 256):
            n = min(256, permutations - start)
            vals = rng.choice([-1, 1], (n, len(all_d))) @ weighted
            extremes += int((np.abs(vals) >= abs(observed) - 1e-12).sum())
        p = (extremes + 1) / (permutations + 1)
        exact = False
    sd = float(all_d.std(ddof=1)) if len(all_d) > 1 else 0.0
    return {
        "macro_delta": observed,
        "query_weighted_delta": float(all_d.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "p_value": p,
        "exact_randomization": exact,
        "datasets_improved": int(sum(d.mean() > 1e-12 for d in diffs)),
        "datasets": len(diffs),
        "query_count": len(all_d),
        "wins": int((all_d > 1e-12).sum()),
        "ties": int((abs(all_d) <= 1e-12).sum()),
        "losses": int((all_d < -1e-12).sum()),
        "practical_improvement_fraction": float((all_d > margin).mean()),
        "paired_standardized_effect_query_weighted": float(all_d.mean() / sd)
        if sd
        else None,
        "scope": "fixed_suite_query_uncertainty",
        "seed": seed,
        "bootstrap_replicates": bootstrap,
    }
