"""Corpus-denominator ranking metrics with explicit judgment coverage."""

import math
from pathlib import Path


def rank(scores, expected_ids=None):
    if not scores or any(
        not isinstance(k, str) or not math.isfinite(v) for k, v in scores.items()
    ):
        raise ValueError("Scores require string IDs and finite values")
    if expected_ids is not None and set(scores) != set(expected_ids):
        raise ValueError("Scored IDs differ from frozen pool")
    return sorted(scores, key=lambda d: (-scores[d], d))


def evaluate_query(ordered_ids, qrels, pool_ids=None, threshold=1, gain="linear"):
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Duplicate ranked IDs")
    if gain not in {"linear", "exponential"}:
        raise ValueError("Unknown gain mapping")
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in qrels.values()):
        raise ValueError("Invalid relevance grade")
    judged = {d: g for d, g in qrels.items() if g is not None}
    relevant = {d for d, g in judged.items() if g >= threshold}
    if not relevant:
        return {"eligible": False, "reason": "no_known_relevant_judgments"}
    pool = set(ordered_ids if pool_ids is None else pool_ids)
    if not set(ordered_ids) <= pool:
        raise ValueError("Ranked IDs outside candidate pool")

    def gains(g):
        return g if gain == "linear" else 2**g - 1

    ideal = sorted((gains(g) for g in judged.values()), reverse=True)
    result = {
        "eligible": True,
        "known_relevant": len(relevant),
        "candidate_recall": len(pool & relevant) / len(relevant),
        "explicit_negative_count": sum(g == 0 for g in judged.values()),
    }
    for k in (1, 5, 10, 20, 50, 100, 1000):
        if k > len(ordered_ids) and k not in (1, 5, 10, 20):
            continue
        top = ordered_ids[:k]
        hits = [int(d in relevant) for d in top]
        dcg = sum(gains(judged.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(top))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        result.update(
            {
                f"ndcg@{k}": dcg / idcg if idcg else 0,
                f"recall@{k}": sum(hits) / len(relevant),
                f"precision@{k}": sum(hits) / k,
                f"hit@{k}": int(any(hits)),
                f"judged@{k}": sum(d in judged for d in top) / k,
                f"mrr@{k}": next((1 / (i + 1) for i, h in enumerate(hits) if h), 0),
                f"map@{k}": sum(
                    sum(hits[: i + 1]) / (i + 1) for i, h in enumerate(hits) if h
                )
                / len(relevant),
            }
        )
    result["retention@10"] = (
        len(set(ordered_ids[:10]) & relevant) / len(pool & relevant)
        if pool & relevant
        else None
    )
    oracle = sorted(pool, key=lambda d: (-judged.get(d, 0), d))[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:10]))
    result["oracle_ndcg@10"] = (
        sum(gains(judged.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(oracle))
        / idcg
    )
    return result


def export_trec(path, query_scores, system):
    if any(c.isspace() for c in system):
        raise ValueError("TREC system name must not contain whitespace")
    lines = []
    for q, scores in sorted(query_scores.items()):
        if any(c.isspace() for c in q):
            raise ValueError("TREC query ID must not contain whitespace")
        for i, d in enumerate(rank(scores), 1):
            if any(c.isspace() for c in d):
                raise ValueError("TREC document ID must not contain whitespace")
            # Rank-derived export score preserves canonical tie order in evaluators
            # that otherwise break numeric-score ties by descending document ID.
            lines.append(f"{q} Q0 {d} {i} {len(scores) - i + 1} {system}\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(lines))
