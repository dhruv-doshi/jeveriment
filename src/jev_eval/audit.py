"""Blinded human audit packages. Model scores and qrels never enter exports."""

import random

from .io import digest


def sample_audit(
    dataset, pools, queries, views, split, query_count, pairs_per_query=20, seed=1729
):
    if split not in {"dev", "test"}:
        raise ValueError("Audit requires separate dev or test population")
    if query_count > len(pools):
        raise ValueError("Audit query budget exceeds available queries")
    ids = sorted(
        pools, key=lambda q: (digest({"dataset": dataset, "q": q, "seed": seed}), q)
    )[:query_count]
    blinded, manifest = [], []
    for q in ids:
        candidates = sorted(c["doc_id"] for c in pools[q]["candidates"])
        rng = random.Random(digest({"seed": seed, "dataset": dataset, "query": q}))
        selected = rng.sample(candidates, min(pairs_per_query, len(candidates)))
        for d in selected:
            pair_id = digest(
                {
                    "dataset": dataset,
                    "query": q,
                    "document": d,
                    "split": split,
                    "seed": seed,
                }
            )
            blinded.append(
                {
                    "pair_id": pair_id,
                    "query": queries[q]["text"],
                    "document": views[d]["text"],
                    "assessor_1": None,
                    "assessor_2": None,
                    "adjudicated_label": None,
                    "judgment_status": "unjudged",
                }
            )
            manifest.append(
                {
                    "pair_id": pair_id,
                    "dataset": dataset,
                    "query_id": q,
                    "doc_id": d,
                    "split": split,
                    "inclusion_probability": query_count
                    / len(pools)
                    * len(selected)
                    / len(candidates),
                    "population": "uniform query and candidate within frozen pool",
                }
            )
    return blinded, manifest


def import_judgments(blinded, manifest, probabilities):
    mapping = {r["pair_id"]: r for r in manifest}
    if set(mapping) != set(r["pair_id"] for r in blinded) or len(blinded) != len(
        mapping
    ):
        raise ValueError("Audit pair mapping mismatch")
    result, insufficient = [], []
    for row in blinded:
        pair_id = row["pair_id"]
        a, b = row.get("assessor_1"), row.get("assessor_2")
        valid = (0, 1, "insufficient_information")
        if a not in valid or b not in valid:
            raise ValueError("Two independent assessor labels are required")
        label = row.get("adjudicated_label") if a != b else a
        if label not in valid:
            raise ValueError("Disagreements require adjudication")
        if label == "insufficient_information":
            insufficient.append(pair_id)
            continue
        result.append(
            {
                **mapping[pair_id],
                "label": label,
                "probability": probabilities[pair_id],
                "judgment_status": "judged",
                "label_source": "human_audit",
            }
        )
    return result, insufficient
