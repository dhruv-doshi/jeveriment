import gzip
import json
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np
import pytest

from jev_eval.rag.answer import generate_answer, pack_context
from jev_eval.rag.clients import BudgetedHTTP, validate_openrouter
from jev_eval.rag.contracts import GoldRecord, QueryRecord
from jev_eval.rag.datasets import adapter_for
from jev_eval.rag.evaluation import evaluate_answer, evaluate_ranking
from jev_eval.rag.experiment import ExperimentRuntime, prepare_dataset
from jev_eval.rag.judge import judge_answer
from jev_eval.rag.load import run_fake_load
from jev_eval.rag.reporting import compare, export_trace_parquet, write_report
from jev_eval.rag.reranking import retrieval_order, score_with_jev, score_with_model
from jev_eval.rag.retrieval import ChunkIndex, DenseIndex, freeze_pool, rrf


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_five_official_shape_adapters(tmp_path):
    root = tmp_path / "sources"
    _write(root / "mirage/benchmark.json", json.dumps({"mmlu": [{"question": "Why?", "options": {"A": "one", "B": "two"}, "answer": "A"}]}))
    _write(root / "mirage/medrag_corpus.jsonl", '{"id":"m1","title":"med","content":"medical evidence"}\n')
    _write(root / "clerc/queries/test.single-removed.direct.tsv", "q1\tWhat does the case say?\n")
    _write(root / "clerc/qrels/qrels-passage.test.direct.tsv", "q1\t0\tp1\t1\n")
    path = root / "clerc/collection/collection.passage.tsv.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as stream:
        stream.write("p1\tlegal passage\n")
    _write(root / "hotpotqa/hotpot_dev_fullwiki_v1.json", json.dumps([{"_id": "h1", "question": "Who?", "answer": "Alice", "supporting_facts": [["Page", 0]], "context": [["Distractor", ["given retrieval result"]]]}]))
    _write(root / "hotpotqa/wiki.jsonl", '{"title":"Page","sentences":["Alice was here."]}\n')
    qasper = {"id": "paper1", "full_text": [{"section_name": "Results", "paragraphs": ["We found a good result."]}], "qas": [{"question_id": "a1", "question": "What result?", "answers": [{"answer": {"unanswerable": False, "extractive_spans": ["good result"], "yes_no": False, "free_form_answer": "", "evidence": ["We found a good result."]}}]}]}
    _write(root / "qasper/qasper.jsonl", json.dumps(qasper) + "\n")
    _write(root / "kilt_fever/kilt_knowledgesource.json", '{"wikipedia_id":42,"wikipedia_title":"Page","text":["KILT evidence"]}\n')
    _write(root / "kilt_fever/fever-dev-kilt.jsonl", '{"id":"f1","input":"A claim","output":[{"answer":"SUPPORTS","provenance":[{"wikipedia_id":42,"start_paragraph_id":0}]}]}\n')
    expected = {"mirage": "m1:0", "clerc": "p1:0", "hotpotqa": "Page:0", "qasper": "paper1:0:0", "kilt_fever": "42:0"}
    for dataset, cid in expected.items():
        adapter = adapter_for(dataset, root)
        assert adapter.audit()["ready"]
        chunk_ids = [chunk.chunk_id for chunk in adapter.iter_chunks()]
        assert cid in chunk_ids
        if dataset != "qasper":
            assert chunk_ids == [cid]
        query, gold = next(adapter.iter_queries())
        assert query.text and gold.query_id == query.query_id
        if dataset == "mirage":
            assert not gold.qrels and "A: one" not in query.text
        if dataset == "hotpotqa":
            assert cid in gold.qrels and "given retrieval result" not in next(adapter.iter_chunks()).text
    qasper["split"] = "validation"
    _write(root / "qasper/qasper.jsonl", json.dumps(qasper) + "\n")
    assert next(adapter_for("qasper", root).iter_queries())[0].split == "reporting"


def test_retrieval_rerank_context_and_native_metrics(tmp_path):
    index = ChunkIndex(tmp_path / "chunks.sqlite")
    from jev_eval.io import digest
    from jev_eval.rag.contracts import ChunkRecord, GoldRecord

    index.add([ChunkRecord(dataset_id="qasper", document_id="p", chunk_id=cid, text=text, text_hash=digest(text), corpus_scope=scope)
               for cid, text, scope in [("p:0", "Alice discovered treatment", "p"), ("p:1", "irrelevant page", "p"), ("other:0", "Alice treatment", "other")]])
    query = QueryRecord(dataset_id="qasper", query_id="q", text="Alice treatment", split="reporting", task="scientific_paper_question_answering", corpus_scope="p")
    found = index.search_bm25(query, 10)
    assert found and all(cid.startswith("p:") for cid, _ in found)
    pool = freeze_pool(query, "bm25", found, index, 10)

    class FakeModel:
        def score(self, question, texts):
            return [float("treatment" in text) for text in texts]

    baseline = retrieval_order(pool)
    reranked = score_with_model(query, pool, index, FakeModel(), "fake")
    assert set(reranked.ordered_ids) == set(baseline.ordered_ids)
    context = pack_context(query, reranked, index, budget_tokens=100)
    assert "other:0" not in context.text
    gold = GoldRecord(dataset_id="qasper", query_id="q", answers=("Alice",), qrels={"p:0": 1})
    assert evaluate_ranking(reranked, gold)["applicable"]
    assert context.selected_ids


def test_dense_index_scope_revision_and_hybrid(tmp_path):
    from jev_eval.io import digest
    from jev_eval.rag.contracts import ChunkRecord

    index = ChunkIndex(tmp_path / "chunks.sqlite")
    rows = [("a", "heart disease evidence", "paper-a"),
            ("b", "unrelated treatment", "paper-a"),
            ("c", "heart disease evidence", "paper-b")]
    index.add([ChunkRecord(dataset_id="qasper", document_id=scope, chunk_id=cid,
                           text=text, text_hash=digest(text), corpus_scope=scope)
               for cid, text, scope in rows])

    class FakeEncoder:
        name = "fixture"
        revision = "rev-1"

        def encode(self, texts, query=False):
            return np.asarray([[1.0, 0.0] if "heart" in text else [0.0, 1.0]
                               for text in texts], dtype=np.float32)

    path = tmp_path / "dense.npy"
    DenseIndex.build(index, path, FakeEncoder(), batch_size=2)
    query = QueryRecord(dataset_id="qasper", query_id="q", text="heart", split="reporting",
                        task="scientific_paper_question_answering", corpus_scope="paper-a")
    dense = DenseIndex(index, path, FakeEncoder()).search(query, 2)
    lexical = index.search_bm25(query, 2)
    assert dense[0][0] == "a"
    assert {cid for cid, _ in dense} == {"a", "b"}
    assert rrf([[cid for cid, _ in lexical], [cid for cid, _ in dense]])[0][0] == "a"
    wrong = FakeEncoder()
    wrong.revision = "rev-2"
    with pytest.raises(ValueError, match="different embedding model"):
        DenseIndex(index, path, wrong)


def test_parallel_jev_scores_exact_frozen_pool(tmp_path):
    from jev_eval.io import digest
    from jev_eval.rag.contracts import ChunkRecord

    index = ChunkIndex(tmp_path / "chunks.sqlite")
    index.add([ChunkRecord(dataset_id="qasper", document_id="paper", chunk_id=f"c{i}",
                           text=f"evidence {i}", text_hash=digest(f"evidence {i}"), corpus_scope="paper")
               for i in range(4)])
    query = QueryRecord(dataset_id="qasper", query_id="q", text="evidence", split="reporting",
                        task="scientific_paper_question_answering", corpus_scope="paper")
    pool = freeze_pool(query, "bm25", [(f"c{i}", 4 - i) for i in range(4)], index, 4)

    def respond(request):
        data = json.loads(request.content)
        cid = data["state"]["document"]["id"]
        resolved = "different-model" if data["model"] == "mismatch" else data["model"]
        return httpx.Response(200, json={"model": resolved,
                                         "answers": {"relevance": {"type": "noul", "noul": int(cid[-1]) / 3}},
                                         "usage": {"input_tokens": 40, "output_tokens": 3}})

    client = BudgetedHTTP(provider="typesafe", root=tmp_path / "run", token="fixture-secret",
                          base_url="https://fixture.invalid", budget_usd=Decimal("1"),
                          max_tokens=10000, max_requests=10,
                          input_usd_per_million=Decimal("0.1"), output_usd_per_million=Decimal("0.1"),
                          rpm=100, tpm=10000, inflight=2, transport=httpx.MockTransport(respond))
    first = score_with_jev(query, pool, index, client, model="fixture-jev", workers=4)
    second = score_with_jev(query, pool, index, client, model="fixture-jev", workers=4)
    assert first.ordered_ids == ("c3", "c2", "c1", "c0")
    assert second.ordered_ids == first.ordered_ids
    assert len(first.raw_response_paths) == 4
    assert client._ledger().totals()["attempts"] == 4
    with pytest.raises(ValueError, match="differs from requested"):
        score_with_jev(query, pool, index, client, model="mismatch", workers=2)


def test_budgeted_generator_records_request_and_checks_citations(tmp_path):
    from jev_eval.rag.contracts import ContextSelection

    query = QueryRecord(dataset_id="mirage", query_id="q", text="Which answer?", split="reporting", task="medical_multiple_choice", answer_options={"A": "correct", "B": "wrong"})
    context = ContextSelection(query_id="q", selected_ids=("c1",), dropped_ids=(), text="[c1] correct", estimated_tokens=4, budget_tokens=100)
    seen = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"id": "g1", "total_cost": 0.00005,
                                                      "latency": 1250, "generation_time": 1200,
                                                      "provider_name": "fixture-provider"}})
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "g1", "model": "moonshotai/kimi-k2.5", "choices": [{"message": {"content": json.dumps({"answer": "A", "citations": ["c1"]})}}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}})

    client = BudgetedHTTP(provider="openrouter", root=tmp_path / "run", token="test-secret",
                          base_url="https://openrouter.ai/api/v1", budget_usd=Decimal("1"),
                          max_tokens=100000, max_requests=10,
                          input_usd_per_million=Decimal("0.45"), output_usd_per_million=Decimal("2.25"),
                          rpm=100, tpm=100000, inflight=2, transport=httpx.MockTransport(respond))
    result = generate_answer(query, context, client, "moonshotai/kimi-k2.5", "fixture-provider")
    assert result.answer == "A" and result.citations == ("c1",)
    assert result.provider_metadata_hash
    assert result.provider_name == "fixture-provider"
    assert result.provider_reported_generation_time_ms == 1200
    assert result.raw_response_path and result.api_latency_ms is not None
    assert seen[0]["provider"]["allow_fallbacks"] is False
    assert seen[0]["max_completion_tokens"] == 768
    assert seen[0]["reasoning"] == {"enabled": False}
    assert "Options:" in seen[0]["messages"][1]["content"]
    assert "test-secret" not in "".join(p.read_text() for p in (tmp_path / "run/raw").rglob("*.json"))
    gold = GoldRecord(dataset_id="mirage", query_id="q", answers=("A",))
    assert evaluate_answer(result, gold, context)["answer_exact_match"] == 1
    assert client.ledger_path.exists()


def test_openrouter_usage_above_reservation_stops_and_records_charge(tmp_path):
    def respond(_request):
        return httpx.Response(200, json={
            "model": "moonshotai/kimi-k2.5",
            "choices": [{"message": {"content": '{"answer":"Blue","citations":[]}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 300, "cost": 0.0006795},
        })

    client = BudgetedHTTP(provider="openrouter", root=tmp_path, token="test-secret",
                          base_url="https://openrouter.ai/api/v1", budget_usd=Decimal("1"),
                          max_tokens=10000, max_requests=10,
                          input_usd_per_million=Decimal("0.45"), output_usd_per_million=Decimal("2.25"),
                          rpm=100, tpm=10000, inflight=1, transport=httpx.MockTransport(respond))
    with pytest.raises(ValueError, match="exceeded reserved token bound"):
        client.post("/chat/completions", {"model": "moonshotai/kimi-k2.5"},
                    max_input_tokens=100, max_output_tokens=128, validate=validate_openrouter)
    assert Decimal(client._ledger().totals()["cost_usd"]) == Decimal("0.0006795")


def test_openrouter_generation_metadata_retries_eventual_404(tmp_path, monkeypatch):
    calls = []

    def respond(_request):
        calls.append(1)
        return (httpx.Response(404, json={"error": "not found"}) if len(calls) == 1
                else httpx.Response(200, json={"data": {"id": "g1", "provider_name": "fixture-provider"}}))

    monkeypatch.setattr("jev_eval.rag.clients.time.sleep", lambda _seconds: None)
    client = BudgetedHTTP(provider="openrouter", root=tmp_path, token="test-secret",
                          base_url="https://openrouter.ai/api/v1", budget_usd=Decimal("1"),
                          max_tokens=10000, max_requests=10,
                          input_usd_per_million=Decimal("0.45"), output_usd_per_million=Decimal("2.25"),
                          rpm=100, tpm=10000, inflight=1, transport=httpx.MockTransport(respond))
    assert client.generation_metadata("g1", None)["provider_name"] == "fixture-provider"
    assert client.generation_metadata("g1", None)["provider_name"] == "fixture-provider"
    assert len(calls) == 2


def test_optional_judge_is_metered_and_replayable(tmp_path):
    from jev_eval.rag.contracts import ContextSelection, GenerationResult

    query = QueryRecord(dataset_id="qasper", query_id="q", text="What happened?", split="reporting", task="scientific_paper_question_answering")
    context = ContextSelection(query_id="q", selected_ids=("p:0",), dropped_ids=(), text="[p:0] A happened", estimated_tokens=5, budget_tokens=100)
    answer = GenerationResult(query_id="q", model_requested="m", model_resolved="m", answer="A", citations=("p:0",), raw_response_hash="h")
    gold = GoldRecord(dataset_id="qasper", query_id="q", answers=("A",))
    calls = []

    def respond(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"id": "judge1", "total_cost": 0.0001}})
        content = {"faithfulness": 0.9, "answer_relevancy": 1.0, "answer_correctness": 0.8, "rationale": "Supported"}
        return httpx.Response(200, json={"id": "judge1", "model": "m", "choices": [{"message": {"content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 10}})

    client = BudgetedHTTP(provider="openrouter", root=tmp_path / "judge", token="fixture",
                          base_url="https://openrouter.ai/api/v1", budget_usd=Decimal("1"),
                          max_tokens=100000, max_requests=10, input_usd_per_million=Decimal("0.45"),
                          output_usd_per_million=Decimal("2.25"), rpm=100, tpm=100000,
                          inflight=2, transport=httpx.MockTransport(respond))
    first = judge_answer(query, context, answer, gold, client, "m", "fixture-provider")
    second = judge_answer(query, context, answer, gold, client, "m", "fixture-provider")
    assert first.faithfulness == second.faithfulness == 0.9
    assert calls == ["POST", "GET"]


def test_missing_sources_fail_closed(tmp_path):
    adapter = adapter_for("hotpotqa", tmp_path)
    assert not adapter.audit()["ready"]
    with pytest.raises(FileNotFoundError):
        list(adapter.iter_queries())


def test_parallel_run_and_replay_without_provider_calls(tmp_path, monkeypatch, capsys):
    from jev_eval.rag.config import load_experiment

    spec = load_experiment("configs/rag/experiment.yaml")
    source = tmp_path / "sources/mirage"
    rows = [{"question": f"Which evidence {i}?", "options": {"A": "good", "B": "bad"}, "answer": "A"} for i in range(12)]
    _write(source / "benchmark.json", json.dumps({"mmlu": rows}))
    _write(source / "medrag_corpus.jsonl", '{"id":"m1","content":"good medical evidence"}\n{"id":"m2","content":"bad medical evidence"}\n')
    manifest = prepare_dataset(spec, "mirage", tmp_path / "sources", tmp_path / "runs")
    assert manifest["queries"] == 12 and manifest["chunks"] == 2
    runtime = ExperimentRuntime(spec, tmp_path / "runs", "mirage")
    first = runtime.run_slice(retriever="bm25", reranker="retrieval_order", split="reporting", limit=3, workers=3)
    second = runtime.run_slice(retriever="bm25", reranker="retrieval_order", split="reporting", limit=3, workers=3)
    assert first["results_hash"] != second["results_hash"]  # cache flags change, scientific scores do not
    first_rows = json.loads(Path(first["path"]).read_text())
    second_rows = json.loads(Path(second["path"]).read_text())
    assert all(not row["cache"]["rerank"] for row in first_rows)
    assert all(row["cache"]["rerank"] for row in second_rows)
    assert [row["ranking"] for row in first_rows] == [row["ranking"] for row in second_rows]
    report = write_report([first["path"]], tmp_path / "runs" / spec.experiment_id, tmp_path / "report.json")
    assert report["quality"]["mirage/bm25/retrieval_order"]["attempted"] == 3
    assert report["stage_latency_ms"]["retrieve"]["count"] == 3
    exported = export_trace_parquet(tmp_path / "runs" / spec.experiment_id, tmp_path / "parquet")
    assert exported["stages"] >= 3 and (tmp_path / "parquet/stages.parquet").exists()
    from jev_eval.cli import main

    monkeypatch.setattr("jev_eval.rag.telemetry.setup_tracing", lambda root: None)
    monkeypatch.setattr("sys.argv", ["jev-eval", "rag", "matrix", "configs/rag/experiment.yaml",
                                     "--datasets", "mirage", "--retrievers", "bm25",
                                     "--rerankers", "retrieval_order", "--run-root", str(tmp_path / "runs"),
                                     "--split", "reporting", "--limit", "2", "--workers", "2", "--execute"])
    main()
    executed = json.loads(capsys.readouterr().out)
    assert executed["cells"] == 1 and executed["results"][0]["queries"] == 2
    assert executed["results"][0]["workers"] == 2


def test_fake_load_retries_and_records_tail_latency(tmp_path):
    result = run_fake_load(tmp_path, requests=12, workers=4, latency_ms=1, every_429=4)
    assert result["successes"] == 12
    assert result["attempts"] > result["requests"]
    assert result["end_to_end_ms"]["count"] == 12
    assert result["end_to_end_ms"]["p99_low_sample_warning"]


def test_paired_comparison_rejects_different_candidate_pools(tmp_path):
    baseline = tmp_path / "baseline.json"
    challenger = tmp_path / "challenger.json"
    row = {"dataset": "qasper", "retriever": "bm25", "reranker": "baseline",
           "query_id": "q1", "pool_hash": "same", "ranking": {"ndcg@10": 0.5}}
    baseline.write_text(json.dumps([row]))
    challenger.write_text(json.dumps([{**row, "reranker": "challenger", "ranking": {"ndcg@10": 0.8}}]))
    result = compare([baseline], [challenger], "ranking.ndcg@10", bootstrap=100, permutations=100)
    assert result["paired"]["macro_delta"] == pytest.approx(0.3)
    challenger.write_text(json.dumps([{**row, "pool_hash": "different"}]))
    with pytest.raises(ValueError, match="Candidate pool mismatch"):
        compare([baseline], [challenger], "ranking.ndcg@10", bootstrap=10, permutations=10)
