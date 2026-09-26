import json

from fastapi.testclient import TestClient

from jev_eval.rag.config import load_experiment
from jev_eval.rag.experiment import prepare_dataset
from jev_eval.rag.service import create_app


def test_internal_service_auth_and_unlabeled_query(tmp_path):
    spec = load_experiment("configs/rag/experiment.yaml")
    source = tmp_path / "sources" / "mirage"
    source.mkdir(parents=True)
    (source / "benchmark.json").write_text(json.dumps({"mmlu": [{"question": "What medical evidence?", "options": {"A": "good"}, "answer": "A"}]}))
    (source / "medrag_corpus.jsonl").write_text('{"id":"m1","content":"good medical evidence"}\n')
    prepare_dataset(spec, "mirage", tmp_path / "sources", tmp_path / "runs")
    app = create_app(spec, tmp_path / "runs", api_key="fixture-secret")
    client = TestClient(app)
    request = {"dataset": "mirage", "question": "medical evidence", "task": "medical_multiple_choice",
               "retriever": "bm25", "reranker": "retrieval_order", "generate": False}
    assert client.post("/v1/answer", json=request).status_code == 401
    response = client.post("/v1/answer", json=request, headers={"x-api-key": "fixture-secret"})
    assert response.status_code == 200
    assert response.json()["ranking"]["reason"] == "service_query_has_no_labels"
    assert client.post("/v1/answer", json={**request, "generate": True}, headers={"x-api-key": "fixture-secret"}).status_code == 403
