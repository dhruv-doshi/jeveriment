import pytest

from jev_eval.metrics import rank
from jev_eval.retrieve import blocked_topk, make_pool, normalized, rrf


def test_blocked_exact_and_ties():
    docs = normalized([[1, 0], [0, 1], [1, 0], [-1, 0]])
    queries = normalized([[2, 1], [1, 0]])
    ids = ["b", "c", "a", "d"]
    result = blocked_topk(queries, docs, ids, k=3, block_size=2)
    for q, got in zip(queries, result):
        expected = dict(zip(ids, map(float, docs @ q)))
        assert list(got) == rank(expected)[:3]
        assert got == pytest.approx({d: expected[d] for d in got})
    assert list(result[1])[:2] == ["a", "b"]


def test_pool_hashes_and_missing_ranks():
    scores = rrf([["a", "b"], ["b", "c"]])
    assert scores["a"] == 1 / 61
    assert scores["b"] == 1 / 62 + 1 / 61
    b, d = {"a": 3, "b": 2, "c": 1}, {"a": 1, "b": 2, "c": 3}
    args = ("q", b, d, {k: k for k in b})
    a = make_pool(*args, depth=2, size=2)
    assert a == make_pool(*args, depth=2, size=2)
    assert len(a.candidates) == 2
    assert len(make_pool(*args, depth=3, kind="union").candidates) == 3
    with pytest.raises(ValueError):
        make_pool("q", b, {"a": 1}, {})
