"""Budgeted HTTP clients. Secrets are sent in headers and never written to artifacts."""

import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import httpx

from jev_eval.io import digest, write_json
from jev_eval.ledger import Ledger, now


@dataclass(frozen=True)
class ProviderReply:
    body: dict
    raw_path: str
    elapsed_ms: float
    cache_hit: bool
    provider_inference_ms: float | None = None


class RateGate:
    """Account-wide sliding window for requests/tokens and concurrent leases."""

    def __init__(self, path: str | Path, rpm: int, tpm: int, inflight: int):
        if min(rpm, tpm, inflight) < 1:
            raise ValueError("Live rate limits must be positive account-specific values")
        self.path, self.rpm, self.tpm, self.inflight = str(path), rpm, tpm, inflight
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.executescript("""
        CREATE TABLE IF NOT EXISTS events (at REAL NOT NULL,tokens INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY,expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cooldown (id INTEGER PRIMARY KEY CHECK(id=1),until REAL NOT NULL);
        INSERT OR IGNORE INTO cooldown VALUES(1,0);
        """)
        db.close()

    def acquire(self, tokens: int, max_wait=3600.0):
        if tokens > self.tpm:
            raise ValueError("Single request reservation exceeds token/minute limit")
        deadline = time.monotonic() + max_wait
        lease = digest(f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}:{random.random()}")
        while True:
            wall = time.time()
            with sqlite3.connect(self.path, timeout=30) as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM events WHERE at<?", (wall - 60,))
                db.execute("DELETE FROM leases WHERE expires<?", (wall,))
                count, used = db.execute("SELECT COUNT(*),COALESCE(SUM(tokens),0) FROM events").fetchone()
                active = db.execute("SELECT COUNT(*) FROM leases").fetchone()[0]
                cooldown = db.execute("SELECT until FROM cooldown WHERE id=1").fetchone()[0]
                if count < self.rpm and used + tokens <= self.tpm and active < self.inflight and wall >= cooldown:
                    db.execute("INSERT INTO events VALUES(?,?)", (wall, tokens))
                    db.execute("INSERT INTO leases VALUES(?,?)", (lease, wall + 180))
                    return lease
            if time.monotonic() >= deadline:
                raise TimeoutError("Rate gate wait exceeded configured deadline; resume later")
            time.sleep(0.2 + random.random() * 0.1)

    def release(self, lease):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("DELETE FROM leases WHERE id=?", (lease,))

    def penalize(self, seconds):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("UPDATE cooldown SET until=MAX(until,?) WHERE id=1", (time.time() + min(float(seconds), 60),))


class BudgetedHTTP:
    def __init__(self, *, provider: str, root: str | Path, token: str, base_url: str,
                 budget_usd: Decimal, max_tokens: int, max_requests: int,
                 input_usd_per_million: Decimal, output_usd_per_million: Decimal,
                 rpm: int, tpm: int, inflight: int, transport=None):
        if not token:
            raise ValueError(f"{provider} API key missing")
        self.provider, self.root = provider, Path(root)
        self.ledger_path = self.root / "budget.sqlite"
        self.gate = RateGate(self.root / f"rate_{provider}.sqlite", rpm, tpm, inflight)
        self.budget_usd, self.max_tokens, self.max_requests = budget_usd, max_tokens, max_requests
        self.input_price = input_usd_per_million / Decimal(1_000_000)
        self.output_price = output_usd_per_million / Decimal(1_000_000)
        self.http = httpx.Client(base_url=base_url, timeout=60, transport=transport,
                                 headers={"Authorization": f"Bearer {token}"}, follow_redirects=False)

    def post(self, endpoint: str, payload: dict, *, max_input_tokens: int,
             max_output_tokens: int, validate, attempts=4) -> ProviderReply:
        request_key = digest({"provider": self.provider, "endpoint": endpoint, "payload": payload})
        ledger = Ledger(self.ledger_path)
        cached = ledger.cached(request_key)
        if cached is not None:
            validate(cached)
            return ProviderReply(cached, "cached", 0.0, True)
        reserved_tokens = max_input_tokens + max_output_tokens
        reserved_usd = self.input_price * max_input_tokens + self.output_price * max_output_tokens
        if reserved_tokens > self.gate.tpm:
            raise ValueError("Request exceeds configured provider token/minute limit")
        for retry in range(attempts):
            lease = self.gate.acquire(reserved_tokens)
            attempt_id = None
            try:
                attempt_id = ledger.reserve(request_key, reserved_usd, reserved_tokens,
                                            self.budget_usd, self.max_tokens, self.max_requests)
                raw_path = self.root / "raw" / self.provider / f"{attempt_id:08d}-{request_key}.json"
                started = time.monotonic()
                try:
                    response = self.http.post(endpoint, json=payload)
                    elapsed = (time.monotonic() - started) * 1000
                    try:
                        body = response.json()
                    except ValueError:
                        body = {"error": "non_json_response"}
                    write_json(raw_path, {"request": payload, "response": body,
                                          "status_code": response.status_code, "elapsed_ms": elapsed,
                                          "timestamp": now(), "request_id": response.headers.get("x-request-id"),
                                          "server_timing": response.headers.get("server-timing")})
                except httpx.TransportError as exc:
                    elapsed = (time.monotonic() - started) * 1000
                    write_json(raw_path, {"request": payload, "error": type(exc).__name__,
                                          "elapsed_ms": elapsed, "timestamp": now()})
                    ledger.finish(attempt_id, "transport_error", raw_path, error=type(exc).__name__)
                    if retry + 1 == attempts:
                        raise RuntimeError("Transport retries exhausted; charge reservation retained") from exc
                    time.sleep(min(2**retry + random.random(), 16))
                    continue
                if response.status_code != 200:
                    ledger.finish(attempt_id, "http_error", raw_path, error=str(response.status_code))
                    if response.status_code in {429, 500, 502, 503, 504, 529} and retry + 1 < attempts:
                        try:
                            delay = float(response.headers.get("retry-after", 2**retry))
                        except ValueError:
                            delay = 2**retry
                        self.gate.penalize(delay)
                        time.sleep(min(delay + random.random(), 60))
                        continue
                    raise RuntimeError(f"{self.provider} HTTP {response.status_code}; see {raw_path}")
                try:
                    validate(body)
                except (ValueError, KeyError, TypeError) as exc:
                    ledger.finish(attempt_id, "invalid_response", raw_path, error=type(exc).__name__)
                    raise
                usage = body.get("usage", {})
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
                if type(input_tokens) is not int or type(output_tokens) is not int:
                    ledger.finish(attempt_id, "missing_usage", raw_path, error="missing_usage")
                    raise ValueError("Provider response missing billed token usage")
                cost = self.input_price * input_tokens + self.output_price * output_tokens
                ledger.finish(attempt_id, "success", raw_path, cost=cost, tokens=input_tokens + output_tokens)
                ledger.success(request_key, raw_path, body.get("model", "unknown"))
                if Decimal(ledger.totals()["cost_usd"]) > self.budget_usd:
                    raise ValueError("Provider actual charge exceeded reserved global cap; stop run")
                return ProviderReply(body, str(raw_path), elapsed, False)
            finally:
                self.gate.release(lease)
        raise RuntimeError("Retries exhausted")


def validate_jev(body):
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise ValueError("JEV response model missing")
    answer = body.get("answers", {}).get("relevance", {})
    score = answer.get("noul")
    if answer.get("type") != "noul" or type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError("Invalid JEV Noul response")


def validate_openrouter(body):
    if not body.get("model") or not body.get("choices"):
        raise ValueError("OpenRouter model or choices missing")
    content = body["choices"][0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("OpenRouter content missing")
    parsed = json.loads(content)
    if not isinstance(parsed.get("answer"), str) or not isinstance(parsed.get("citations"), list):
        raise ValueError("OpenRouter structured answer invalid")
