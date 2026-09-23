import pytest

from jev_eval.feature_model import fit_feature_model


def row(q, label, split):
    return {
        "dataset": "fixture",
        "query_id": q,
        "doc_id": "d",
        "label": label,
        "split": split,
        "judgment_status": "judged",
        "features": {"b": float(label), "d": float(label)},
    }


def test_feature_baseline_uses_disjoint_judged_groups():
    train = [row("a", 0, "train"), row("b", 1, "train")]
    dev = [row("c", 0, "dev"), row("d", 1, "dev")]
    _, report = fit_feature_model(train, dev, ["b", "d"])
    assert report["validation_brier"] < 0.25
    with pytest.raises(ValueError, match="complete queries"):
        fit_feature_model(train, train, ["b", "d"])
    with pytest.raises(ValueError, match="explicitly judged"):
        fit_feature_model(
            train, [{**dev[0], "judgment_status": "unjudged"}], ["b", "d"]
        )
