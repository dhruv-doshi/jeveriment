import math

import ir_measures
import pytest

from jev_eval.metrics import evaluate_query, export_trec, rank


def test_corpus_ideal_and_judgment_status():
    result = evaluate_query(
        ["b", "x", "a"], {"a": 2, "b": 1, "missing": 3, "x": None, "zero": 0}
    )
    expected = (1 + 2 / math.log2(4)) / (3 + 2 / math.log2(3) + 1 / math.log2(4))
    assert result["ndcg@10"] == pytest.approx(expected)
    assert result["recall@10"] == pytest.approx(2 / 3)
    assert result["judged@10"] == 0.2
    assert result["explicit_negative_count"] == 1
    assert result["candidate_recall"] == pytest.approx(2 / 3)
    assert not evaluate_query(["a"], {"a": 0})["eligible"]


def test_threshold_and_missing_pool():
    result = evaluate_query(["a", "b"], {"a": 1, "b": 2}, threshold=2)
    assert result["mrr@10"] == 0.5
    with pytest.raises(ValueError):
        evaluate_query(["a", "a"], {"a": 1})
    with pytest.raises(ValueError):
        rank({"a": 1}, ["a", "b"])
    assert rank({"b": 1, "a": 1}) == ["a", "b"]


def test_official_backend_and_trec(tmp_path):
    qrels = [ir_measures.Qrel("q", d, g) for d, g in {"a": 3, "b": 2, "c": 1}.items()]
    scores = {"b": 3.0, "x": 2.0, "a": 1.0}
    run = [ir_measures.ScoredDoc("q", d, s) for d, s in scores.items()]
    official = ir_measures.calc_aggregate(
        [ir_measures.nDCG @ 10, ir_measures.R @ 10, ir_measures.AP @ 100], qrels, run
    )
    own = evaluate_query(rank(scores), {"a": 3, "b": 2, "c": 1})
    assert own["ndcg@10"] == pytest.approx(official[ir_measures.nDCG @ 10])
    assert own["recall@10"] == pytest.approx(official[ir_measures.R @ 10])
    export_trec(tmp_path / "run.trec", {"q": scores}, "fixture")
    assert len(list(ir_measures.read_trec_run(str(tmp_path / "run.trec")))) == 3
