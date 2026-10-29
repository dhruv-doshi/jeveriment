from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from jev_eval.config import Settings
from jev_eval.efficiency import jev_job, packed_request, query_row, summarize
from jev_eval.io import read_json, write_json
from jev_eval.jev import JevClient
from jev_eval.tradeoffs import archived_path, percentiles, replay


def archive():
    return {
        "config": {
            "dataset": "fixture",
            "experiment_id": "fixture",
            "binary_threshold": 1,
            "gain": "linear",
            "task": "question_answering",
        },
        "pools": {
            "q": SimpleNamespace(
                candidates=[SimpleNamespace(doc_id=d) for d in ("b", "a")]
            )
        },
        "sources": {"rrf": {"q": {"a": 2, "b": 1}}},
        "scores": {s: {"q": {"a": 0, "b": 1}} for s in ("qwen", "bge", "jev")},
        "qrels": {"q": {"b": 1, "outside_pool": 1}},
        "queries": {"q": {"text": "query"}},
        "views": {"a": {"text": "A"}, "b": {"text": "B"}},
    }


def test_depth_replay_uses_incoming_order_and_full_corpus_denominator():
    rows = replay(archive(), [1, 2])
    jev = {r["depth"]: r for r in rows if r["system"] == "jev"}
    assert jev[1]["candidate_recall"] == 0
    assert jev[1]["ndcg@10"] == 0
    assert jev[2]["candidate_recall"] == 0.5
    assert 0 < jev[2]["ndcg@10"] < 1


def test_packed_questions_do_not_share_document_text():
    state, questions, mapping = packed_request(
        "query",
        [{"doc_id": "a", "text": "A"}, {"doc_id": "b", "text": "B"}],
        "question_answering",
    )
    assert state == {"query": "query", "task": "question_answering"}
    assert questions["candidate_000"]["instructions"]["document"]["text"] == "A"
    assert questions["candidate_001"]["instructions"]["document"]["text"] == "B"
    assert mapping == {"candidate_000": "a", "candidate_001": "b"}


def test_archive_path_cannot_escape(tmp_path):
    assert (
        archived_path(tmp_path, "/root/jeveriment/runs/x/raw/a.json")
        == tmp_path / "runs/x/raw/a.json"
    )
    with pytest.raises(ValueError, match="escapes"):
        archived_path(tmp_path, "runs/../../secret")


def test_isolated_ledger_keeps_original_budget_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(api_key="fake", max_cost_usd=Decimal("1"))
    client = JevClient(
        settings,
        tmp_path / "benchmark",
        ledger_path=tmp_path / "benchmark/budget.sqlite",
    )
    assert not (tmp_path / "runs").exists()
    client.http.close()
    client.ledger.db.close()


@pytest.mark.parametrize("arm", ["jev_sequential", "jev_concurrent", "jev_packed"])
def test_live_runner_calls_every_pair_and_resume_does_not_bill_again(
    tmp_path, monkeypatch, arm
):
    seen = []

    class FakeClient(JevClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.http.close()

            def respond(request):
                import json

                if request.method == "GET":
                    return httpx.Response(200, json={"models": []})
                payload = json.loads(request.content)
                seen.append(payload)
                return httpx.Response(
                    200,
                    json={
                        "model": "jev-1.13.0",
                        "answers": {
                            k: {"type": "noul", "noul": 0.7}
                            for k in payload["questions"]
                        },
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    },
                )

            self.http = httpx.Client(transport=httpx.MockTransport(respond))

    monkeypatch.setattr("jev_eval.efficiency.JevClient", FakeClient)
    monkeypatch.setattr(
        "jev_eval.efficiency.SharedRateLimiter",
        lambda _: SimpleNamespace(acquire=lambda: None, succeeded=lambda: None),
    )
    data = {"dataset": "fixture", "query_ids": ["q"], "candidates": {"q": ["a", "b"]}}
    manifest = {
        "identity": "frozen",
        "workers": 2,
        "pack_size": 2,
        "requests_per_second": 5,
    }
    settings = Settings(api_key="fake", max_cost_usd=Decimal("1"))
    jev_job(archive(), data, 2, arm, manifest, tmp_path, settings, Decimal("1"))
    row = read_json(tmp_path / f"queries/fixture-2-{arm}-q.json")
    assert row["status"] == "complete"
    assert set(row["scores"]) == {"a", "b"}
    assert row["attempts"] == (1 if arm == "jev_packed" else 2)
    assert row["input_tokens"] == 100 * row["attempts"]
    assert row["query_latency_seconds"] > 0
    before = len(seen)
    # A direct job call warms connections, but completed query requests are skipped.
    jev_job(archive(), data, 2, arm, manifest, tmp_path, settings, Decimal("1"))
    assert len(seen) - before == (2 if arm == "jev_concurrent" else 1)


def test_failure_not_silently_counted_as_success_or_in_paired_quality(tmp_path):
    write_json(
        tmp_path / "manifest.json",
        {
            "seed": 1729,
            "datasets": [{"dataset": "fixture"}],
            "depths": [2],
            "query_count_per_dataset": 1,
        },
    )
    row = query_row(
        archive(), "q", ["a", "b"], {"a": 1}, "jev_packed", 2, 7, ["timeout"]
    )
    write_json(tmp_path / "queries/failed.json", row)
    result = next(r for r in summarize(tmp_path) if r["arm"] == "jev_packed")
    assert result["completed"] == 0 and result["failed"] == 1
    assert result["ndcg@10_completed_only"] is None
    assert result["serial_query_throughput_per_second"] == 0
    assert read_json(tmp_path / "paired.json") == []


def test_nearest_rank_p95_small_sample():
    assert percentiles(range(1, 31))["p95"] == 29
