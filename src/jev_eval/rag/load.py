"""Local fault-injection load profile; never contacts a paid provider."""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import httpx

from .clients import BudgetedHTTP
from .reporting import distribution


def run_fake_load(root, *, requests=100, workers=8, latency_ms=20, every_429=0):
    if min(requests, workers) < 1 or latency_ms < 0 or every_429 < 0:
        raise ValueError("Invalid fake load settings")
    lock = threading.Lock()
    calls = 0
    root = Path(root) / uuid.uuid4().hex

    def respond(request):
        nonlocal calls
        with lock:
            calls += 1
            attempt = calls
        time.sleep(latency_ms / 1000)
        if every_429 and attempt % every_429 == 0:
            return httpx.Response(429, json={"error": "injected_throttle"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"model": "fake", "usage": {"input_tokens": 1, "output_tokens": 1}, "value": attempt})

    client = BudgetedHTTP(provider="fake", root=root, token="local-fixture", base_url="https://fake.invalid",
                          budget_usd=Decimal("0"), max_tokens=requests * 10, max_requests=requests * 4,
                          input_usd_per_million=Decimal("0"), output_usd_per_million=Decimal("0"),
                          rpm=max(1000, requests * 4), tpm=max(10000, requests * 10), inflight=workers,
                          transport=httpx.MockTransport(respond))

    def one(index):
        started = time.monotonic()
        try:
            def validate(body):
                if body.get("model") != "fake":
                    raise ValueError("Unexpected fake model")

            reply = client.post("/fixture", {"index": index}, max_input_tokens=2,
                                max_output_tokens=2, validate=validate)
            return {"ok": True, "elapsed_ms": (time.monotonic() - started) * 1000,
                    "queue_wait_ms": reply.queue_wait_ms}
        except Exception as exc:
            return {"ok": False, "elapsed_ms": (time.monotonic() - started) * 1000,
                    "error": type(exc).__name__}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, index) for index in range(requests)]
        results = [future.result() for future in as_completed(futures)]
    duration = time.monotonic() - started
    successes = sum(item["ok"] for item in results)
    return {"artifact_root": str(root), "requests": requests, "attempts": calls, "workers": workers,
            "successes": successes, "failures": requests - successes,
            "retry_amplification": calls / requests,
            "throughput_qps": successes / duration if duration else None,
            "end_to_end_ms": distribution(item["elapsed_ms"] for item in results),
            "queue_ms": distribution(item.get("queue_wait_ms") for item in results),
            "fault_injection": {"latency_ms": latency_ms, "every_429": every_429}}
