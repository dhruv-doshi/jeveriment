import pytest

from jev_eval.fusion import minmax, tune, tune_matched, weighted
from jev_eval.selection import select


def test_fixed_selection_and_constants():
    assert [r["doc_id"] for r in select({"b": 1, "a": 1}, ["a", "b"], 10)] == ["a", "b"]
    assert minmax({"a": 2, "b": 2}) == {"a": 0, "b": 0}
    assert weighted(
        {"b": {"a": 0, "b": 2}, "d": {"a": 2, "b": 0}}, {"b": 0.5, "d": 0.5}
    ) == {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError):
        select({"a": 1}, ["a", "b"])


def test_no_test_tuning():
    features = {"q": {"b": {"a": 1, "b": 0}, "d": {"a": 0, "b": 1}}}
    with pytest.raises(ValueError):
        tune(features, {"q": {"a": 1}}, {"q": "test"})
    fit = tune(features, {"q": {"a": 1}}, {"q": "dev"})
    assert fit["weights"]["b"] >= 0.5  # canonical ID tie already places a first
    assert len(fit["trials"]) == 11


def test_tuning_comparators_receive_equal_search_budget():
    features = {"q": {"b": {"a": 1, "b": 0}, "d": {"a": 0, "b": 1}}}
    with_jev = {"q": {**features["q"], "j": {"a": 0.8, "b": 0.2}}}
    fits = tune_matched(
        {"bd": features, "bdj": with_jev}, {"q": {"a": 1}}, {"q": "dev"}
    )
    assert (
        fits["bd"]["evaluated_configurations"]
        == fits["bdj"]["evaluated_configurations"]
        == 11
    )
    assert fits["bdj"]["grid_size"] == 66
