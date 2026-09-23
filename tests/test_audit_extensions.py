import pytest

from jev_eval.audit import import_judgments, sample_audit
from jev_eval.extensions import evidence_coverage, mmr, rag_audit
from jev_eval.rubrics import build_request


def test_audit_is_blinded_and_requires_two_assessors():
    pools = {"q": {"candidates": [{"doc_id": "d", "bm25_score": 9, "jev": 0.9}]}}
    blinded, manifest = sample_audit(
        "fixture",
        pools,
        {"q": {"text": "query"}},
        {"d": {"text": "document"}},
        "test",
        1,
        1,
    )
    assert set(blinded[0]) == {
        "pair_id",
        "query",
        "document",
        "assessor_1",
        "assessor_2",
        "adjudicated_label",
        "judgment_status",
    }
    with pytest.raises(ValueError):
        import_judgments(blinded, manifest, {blinded[0]["pair_id"]: 0.9})
    blinded[0].update(assessor_1=1, assessor_2=1)
    pairs, _ = import_judgments(blinded, manifest, {blinded[0]["pair_id"]: 0.9})
    assert pairs[0]["label"] == 1


def test_independent_and_contextual_semantics():
    candidates = [
        {"doc_id": "secret_real_id", "text": "text"},
        {"doc_id": "another", "text": "other"},
    ]
    with pytest.raises(ValueError):
        build_request("q", candidates, "question_answering", "independent")
    state, questions, mapping = build_request(
        "q", candidates, "counterargument", "contextual"
    )
    assert "documents" in state and "secret_real_id" not in str(state)
    assert len(questions) == 2 and mapping["candidate_000"] == "secret_real_id"
    assert "candidate_001" in questions["candidate_001"]["instructions"]
    assert "counterargument" in questions["candidate_000"]["instructions"]


def test_mmr_and_alternative_support():
    selected = mmr(
        ["a", "b", "c"], {"a": 1, "b": 0.9, "c": 0.8}, [[1, 0], [1, 0], [0, 1]], 2, 0.5
    )
    assert selected == ["a", "c"]
    assert evidence_coverage(selected, [["a", "b"], ["c"]])["complete_evidence_set"]
    with pytest.raises(ValueError):
        rag_audit([{"audit_source": "jev"}])
