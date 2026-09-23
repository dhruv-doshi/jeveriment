from .metrics import rank


def select(scores, pool_ids, k=10):
    if k < 1 or len(set(pool_ids)) != len(pool_ids):
        raise ValueError("Invalid selection budget or duplicate pool IDs")
    if not pool_ids:
        if scores:
            raise ValueError("Scores outside empty pool")
        return []
    ordered = rank(scores, pool_ids)
    return [
        {
            "doc_id": d,
            "score": scores[d],
            "reason": "deterministic_top_k",
            "status": "complete",
        }
        for d in ordered[:k]
    ]
