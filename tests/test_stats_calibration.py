import pytest

from jev_eval.calibration import (
    calibration_metrics,
    cluster_brier_interval,
    fit_logistic,
)
from jev_eval.stats import holm, paired_comparison


def pair(q, label, p, split="test"):
    return {
        "dataset": "fixture",
        "query_id": q,
        "doc_id": "d",
        "label": label,
        "probability": p,
        "split": split,
        "judgment_status": "judged",
        "label_source": "human_audit",
    }


def test_paired_domain_weights_and_exact_null():
    result = paired_comparison(
        {"a": {"1": (0, 1)}, "b": {"2": (0, 0), "3": (0, 0), "4": (0, 0)}},
        bootstrap=200,
    )
    assert result["macro_delta"] == 0.5
    assert result["query_weighted_delta"] == 0.25
    null = paired_comparison({"a": {"q": (1, 1)}}, bootstrap=100)
    assert null["p_value"] == 1
    assert null["ci95"] == [0, 0]
    assert holm({"H1": 0.01, "H2": 0.04, "H3": 0.2}) == pytest.approx(
        {"H1": 0.03, "H2": 0.08, "H3": 0.2}
    )


def test_calibration_and_clustering():
    perfect = [pair("a", 0, 0), pair("b", 1, 1)]
    metrics = calibration_metrics(perfect)
    assert metrics["brier"] == 0
    assert metrics["ece"] == 0
    assert cluster_brier_interval(perfect, replicates=30) == [0, 0]
    assert calibration_metrics([pair("a", 0, 1), pair("b", 1, 0)])["brier"] == 1
    with pytest.raises(ValueError):
        calibration_metrics([{**perfect[0], "judgment_status": "unjudged"}])
    with pytest.raises(ValueError):
        fit_logistic([pair("a", 0, 0.1, "dev"), pair("b", 1, 0.9, "dev")], perfect)
