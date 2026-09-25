from decimal import Decimal

import httpx
import pytest

from jev_eval.config import Settings
from jev_eval.jev import JevClient


def test_direct_settings_use_typesafe_key_and_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "TYPESAFE_API_KEY",
        "AI_GATEWAY_API_KEY",
        "JEV_MODEL",
        "JEV_API_BASE_URL",
        "JEV_MAX_COST_USD",
        "JEV_INPUT_USD_PER_MILLION_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=direct-fixture\n"
        "JEV_MODEL=jev-1.13.0\n"
        "JEV_MAX_COST_USD=1\n"
    )
    settings = Settings.load()
    assert settings.base_url == "https://api.typesafe.ai"
    assert settings.model == "jev-1.13.0"
    assert settings.api_key.get_secret_value() == "direct-fixture"
    monkeypatch.setenv("JEV_API_BASE_URL", "https://ai-gateway.vercel.sh/typesafe")
    with pytest.raises(ValueError, match="api.typesafe.ai"):
        Settings.load()


def fixture_client(tmp_path, monkeypatch, responses):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("jev_eval.jev.time.sleep", lambda _: None)
    client = JevClient(
        Settings(api_key="synthetic-not-a-real-key", max_cost_usd=Decimal("1")),
        tmp_path / "run",
        max_attempts=3,
    )
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "jev-latest",
                        }
                    ]
                },
            )
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


def response(value=0.8, model="jev-1.13.0"):
    return {
        "model": model,
        "answers": {"r": {"type": "noul", "noul": value}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }


def test_success_resume_and_no_retry_low_probability(tmp_path, monkeypatch):
    client, calls = fixture_client(tmp_path, monkeypatch, [(200, response(0.001))])
    question = {"r": {"type": "noul", "instructions": "fixture"}}
    first = client.evaluate("text", question)
    second = client.evaluate("text", question)
    assert first == second and len(calls) == 2
    assert client.ledger.totals()["attempts"] == 1
    assert Decimal(client.ledger.totals()["reported_cost_usd"]) == Decimal("0.00000042")
    assert len(list((tmp_path / "run" / "raw").glob("*.json"))) == 1
    assert (
        "synthetic-not-a-real-key"
        not in next((tmp_path / "run" / "raw").glob("*.json")).read_text()
    )


def test_error_only_retry_and_model_drift(tmp_path, monkeypatch):
    client, calls = fixture_client(
        tmp_path,
        monkeypatch,
        [
            (429, {"error": "rate limit"}),
            (200, response()),
            (200, response(model="changed")),
        ],
    )
    question = {"r": {"type": "noul", "instructions": "fixture"}}
    client.evaluate("text", question)
    assert client.ledger.totals()["attempts"] == 2
    with pytest.raises(ValueError, match="identity changed"):
        client.evaluate("different", question)
    assert client.ledger.cached("unknown") is None


def test_rate_limit_pauses_shared_limiter_before_retry(tmp_path, monkeypatch):
    client, calls = fixture_client(
        tmp_path,
        monkeypatch,
        [(429, {"error": "rate limit"}), (200, response())],
    )

    class Limiter:
        def __init__(self):
            self.events = []

        def acquire(self):
            self.events.append("acquire")

        def penalize(self, delay):
            self.events.append(("penalize", delay))

        def succeeded(self):
            self.events.append("success")

    client.rate_limiter = Limiter()
    client.http.close()

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "models": [{"name": "jev-latest"}]
                },
            )
        if len([r for r in calls if r.method == "POST"]) == 1:
            return httpx.Response(429, headers={"retry-after": "90"}, json={})
        return httpx.Response(200, json=response())

    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    client.evaluate("text", {"r": {"type": "noul", "instructions": "fixture"}})
    assert client.rate_limiter.events == [
        "acquire",
        ("penalize", 90.0),
        "acquire",
        "success",
    ]


def test_auth_error_fails_fast_and_malformed_not_cached(tmp_path, monkeypatch):
    client, calls = fixture_client(
        tmp_path, monkeypatch, [(403, {"error": "verification"})]
    )
    question = {"r": {"type": "noul", "instructions": "fixture"}}
    with pytest.raises(RuntimeError, match="403"):
        client.evaluate("text", question)
    assert len(calls) == 2
    assert client.ledger.db.execute("SELECT COUNT(*) FROM success").fetchone()[0] == 0


def test_invalid_probability_fails_without_retry(tmp_path, monkeypatch):
    client, calls = fixture_client(tmp_path, monkeypatch, [(200, response(1.1))])
    with pytest.raises(ValueError, match="probability"):
        client.evaluate("text", {"r": {"type": "noul", "instructions": "fixture"}})
    assert len(calls) == 2
    assert client.ledger.db.execute("SELECT COUNT(*) FROM success").fetchone()[0] == 0
