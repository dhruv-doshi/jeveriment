from decimal import Decimal

import pytest

from jev_eval.ledger import Ledger


def test_two_clients_share_the_spend_cap(tmp_path):
    db = tmp_path / "budget.sqlite"
    first = Ledger(db)
    second = Ledger(db)
    limits = {"max_usd": Decimal("1"), "max_tokens": 1000, "max_requests": 10}
    first.reserve("typesafe:q1", Decimal("0.60"), 100, **limits)
    with pytest.raises(ValueError, match="Monetary cap"):
        second.reserve("openrouter:q1", Decimal("0.60"), 100, **limits)
    assert Decimal(second.totals()["cost_usd"]) == Decimal("0.60")
