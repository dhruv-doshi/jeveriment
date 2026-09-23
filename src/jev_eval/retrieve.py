import numpy as np

from .contracts import Candidate, Pool
from .metrics import rank


def normalized(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("Dense vectors must be finite and nonzero")
    return vectors / norms


def blocked_topk(queries, documents, doc_ids, k=1000, block_size=4096):
    if len(doc_ids) != len(documents) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Document row mapping is invalid")
    if k < 1 or block_size < 1:
        raise ValueError("Invalid retrieval depth/block size")
    results = []
    for query in queries:
        best = {}
        for start in range(0, len(documents), block_size):
            scores = (
                np.asarray(documents[start : start + block_size], dtype=np.float32)
                @ query
            )
            best.update(
                {doc_ids[start + i]: float(score) for i, score in enumerate(scores)}
            )
            best = {d: best[d] for d in rank(best)[:k]}
        results.append(best)
    return results


def rrf(rankings, constant=60):
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative")
    scores = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Duplicate source rank IDs")
        for i, d in enumerate(ranking, 1):
            scores[d] = scores.get(d, 0.0) + 1 / (constant + i)
    return scores


def make_pool(
    query_id, bm25_scores, dense_scores, views, depth=1000, size=50, kind="hybrid"
):
    if set(bm25_scores) != set(dense_scores):
        raise ValueError("Numerical fusion requires both scores for every document")
    b, d = rank(bm25_scores)[:depth], rank(dense_scores)[:depth]
    br, dr = {v: i for i, v in enumerate(b, 1)}, {v: i for i, v in enumerate(d, 1)}
    if kind == "hybrid":
        ids = rank(rrf([b, d]))[:size]
    elif kind == "bm25":
        ids = b[:size]
    elif kind == "dense":
        ids = d[:size]
    elif kind == "union":
        # Union condition is independently sized: B_100 union D_100, at most 200.
        ids = list(dict.fromkeys(b[:100] + d[:100]))
    else:
        raise ValueError("Unknown pool family")
    candidates = [
        Candidate(
            query_id=query_id,
            doc_id=i,
            bm25_rank=br.get(i),
            dense_rank=dr.get(i),
            bm25_score=bm25_scores[i],
            dense_score=dense_scores[i],
            in_bm25=i in br,
            in_dense=i in dr,
            text_view_hash=views[i],
        )
        for i in ids
    ]
    return Pool.build(query_id, candidates)
