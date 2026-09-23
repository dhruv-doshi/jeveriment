from decimal import Decimal

import pytest

from jev_eval.jev import validate_response
from jev_eval.ledger import Ledger


def test_response_mapping_and_distribution():
    q = {"r": {"type": "score", "criteria": ["no", "yes"]}}
    response = {
        "model": "test",
        "answers": {
            "r": {
                "type": "score",
                "score": 0.75,
                "confidence": 0.5,
                "probabilities": {"0": 0.25, "1": 0.75},
            }
        },
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    validate_response(response, q)
    response["answers"]["r"]["score"] = 0.5
    with pytest.raises(ValueError):
        validate_response(response, q)
    response["answers"] = {"wrong": {}}
    with pytest.raises(ValueError):
        validate_response(response, q)


def test_budget_survives_restart(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = Ledger(path)
    ledger.reserve("key", Decimal("0.6"), 10, Decimal("1"), 100, 5)
    ledger.db.close()
    reopened = Ledger(path)
    with pytest.raises(ValueError, match="Monetary"):
        reopened.reserve("key", Decimal("0.6"), 10, Decimal("1"), 100, 5)
    assert reopened.totals()["cost_usd"] == "0.6"


def test_live_observed_score_rounding_inconsistency_remains_rejected():
    questions = {"r": {"type": "score", "criteria": ["none", "partial", "direct"]}}
    response = {
        "model": "fixture",
        "answers": {
            "r": {
                "type": "score",
                "score": 1.63,
                "confidence": 0.44,
                "probabilities": {"0": 0, "1": 0.36, "2": 0.64},
            }
        },
        "usage": {"input_tokens": 320, "output_tokens": 17},
    }
    with pytest.raises(ValueError, match="expected level"):
        validate_response(response, questions)
