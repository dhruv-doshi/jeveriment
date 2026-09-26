import gzip
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from jev_eval.rag.answer import generate_answer, pack_context
from jev_eval.rag.clients import BudgetedHTTP
from jev_eval.rag.contracts import GoldRecord, QueryRecord
from jev_eval.rag.datasets import adapter_for
from jev_eval.rag.evaluation import evaluate_answer, evaluate_ranking
from jev_eval.rag.experiment import ExperimentRuntime, prepare_dataset
from jev_eval.rag.reranking import retrieval_order, score_with_model
from jev_eval.rag.retrieval import ChunkIndex, freeze_pool


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
        assert [chunk.chunk_id for chunk in adapter.iter_chunks()] == [cid]
        query, gold = next(adapter.iter_queries())
        assert query.text and gold.query_id == query.query_id
        if dataset == "mirage":
            assert not gold.qrels and "A: one" not in query.text
        if dataset == "hotpotqa":
            assert cid in gold.qrels and "given retrieval result" not in next(adapter.iter_chunks()).text


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


def test_budgeted_generator_records_request_and_checks_citations(tmp_path):
    from jev_eval.rag.contracts import ContextSelection

    query = QueryRecord(dataset_id="mirage", query_id="q", text="Which answer?", split="reporting", task="medical_multiple_choice", answer_options={"A": "correct", "B": "wrong"})
    context = ContextSelection(query_id="q", selected_ids=("c1",), dropped_ids=(), text="[c1] correct", estimated_tokens=4, budget_tokens=100)
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "g1", "model": "moonshotai/kimi-k2.5", "choices": [{"message": {"content": json.dumps({"answer": "A", "citations": ["c1"]})}}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}})

    client = BudgetedHTTP(provider="openrouter", root=tmp_path / "run", token="test-secret",
                          base_url="https://openrouter.ai/api/v1", budget_usd=Decimal("1"),
                          max_tokens=100000, max_requests=10,
                          input_usd_per_million=Decimal("0.45"), output_usd_per_million=Decimal("2.25"),
                          rpm=100, tpm=100000, inflight=2, transport=httpx.MockTransport(respond))
    result = generate_answer(query, context, client, "moonshotai/kimi-k2.5", "fixture-provider")
    assert result.answer == "A" and result.citations == ("c1",)
    assert seen[0]["provider"]["allow_fallbacks"] is False
    assert "Options:" in seen[0]["messages"][1]["content"]
    assert "test-secret" not in "".join(p.read_text() for p in (tmp_path / "run/raw").rglob("*.json"))
    gold = GoldRecord(dataset_id="mirage", query_id="q", answers=("A",))
    assert evaluate_answer(result, gold, context)["answer_exact_match"] == 1
    assert client.ledger_path.exists()


def test_missing_sources_fail_closed(tmp_path):
    adapter = adapter_for("hotpotqa", tmp_path)
    assert not adapter.audit()["ready"]
    with pytest.raises(FileNotFoundError):
        list(adapter.iter_queries())


def test_parallel_run_and_replay_without_provider_calls(tmp_path):
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
