import sqlite3
from concurrent.futures import ThreadPoolExecutor
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


def test_parallel_ledger_reservations_and_completion(tmp_path):
    path = tmp_path / "budget.sqlite"

    def request(index):
        ledger = Ledger(path)
        attempt = ledger.reserve(
            str(index), Decimal("0.01"), 100, Decimal("1"), 10000, 100
        )
        ledger.finish(attempt, "success", tmp_path / str(index), Decimal("0.002"), 20)
        ledger.db.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(request, range(40)))
    ledger = Ledger(path)
    totals = ledger.totals()
    assert totals["attempts"] == 40 and totals["tokens"] == 800
    assert Decimal(totals["cost_usd"]) == Decimal("0.08")
    assert Decimal(totals["reported_cost_usd"]) == Decimal("0.08")
    assert Decimal(totals["uncertain_reserved_usd"]) == 0
    assert ledger.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 40


def test_ledger_imports_existing_attempts(tmp_path):
    path = tmp_path / "budget.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE attempts (id INTEGER PRIMARY KEY, request_key TEXT,"
        "started_at TEXT, status TEXT, reserved_usd TEXT, actual_usd TEXT,"
        "reserved_tokens INTEGER, tokens INTEGER, raw_path TEXT, error_code TEXT)"
    )
    db.execute(
        "INSERT INTO attempts (reserved_usd,actual_usd,reserved_tokens,tokens) "
        "VALUES ('0.5','0.02',100,12)"
    )
    db.commit()
    db.close()
    ledger = Ledger(path)
    assert ledger.totals() == {
        "attempts": 1,
        "cost_usd": "0.02",
        "reported_cost_usd": "0.02",
        "uncertain_reserved_usd": "0",
        "tokens": 12,
    }


def test_parallel_reservations_keep_shared_money_cap(tmp_path):
    path = tmp_path / "budget.sqlite"

    def reserve(index):
        ledger = Ledger(path)
        try:
            ledger.reserve(
                str(index), Decimal("0.6"), 10, Decimal("1"), 100, 10
            )
            return "reserved"
        except ValueError as error:
            return str(error)
        finally:
            ledger.db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, range(2)))
    assert outcomes.count("reserved") == 1
    assert any("Monetary cap reached" in outcome for outcome in outcomes)


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
