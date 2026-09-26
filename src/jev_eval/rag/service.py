"""Internal FastAPI service over the same recorded RAG execution path."""

import threading
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from jev_eval.io import digest

from .config import ExperimentSpec
from .contracts import QueryRecord
from .experiment import ExperimentRuntime


class AnswerRequest(BaseModel):
    dataset: str
    question: str = Field(min_length=1)
    task: str
    retriever: str
    reranker: str
    corpus_scope: str | None = None
    answer_options: dict[str, str] = Field(default_factory=dict)
    generate: bool = True


def create_app(spec: ExperimentSpec, run_root: str | Path, *, api_key: str,
               revisions=None, limits=None, provider_only=None, allow_paid=False):
    if not api_key:
        raise ValueError("Service API key required")
    app = FastAPI(title="JEV RAG experiment", version="0.1")
    runtimes = {}
    lock = threading.Lock()

    def runtime_for(dataset):
        if dataset not in {item.id for item in spec.datasets}:
            raise HTTPException(400, "Unknown dataset")
        with lock:
            if dataset not in runtimes:
                runtimes[dataset] = ExperimentRuntime(spec, run_root, dataset,
                                                      revisions, provider_only, limits)
            return runtimes[dataset]

    @app.get("/health")
    def health():
        return {"status": "ok", "experiment_id": spec.experiment_id,
                "paid_calls_enabled": allow_paid}

    @app.post("/v1/answer")
    def answer(body: AnswerRequest, x_api_key: str | None = Header(default=None)):
        if x_api_key != api_key:
            raise HTTPException(401, "Unauthorized")
        if not allow_paid and (body.generate or body.reranker == "jev"):
            raise HTTPException(403, "Paid adapters disabled for this service")
        runtime = runtime_for(body.dataset)
        if body.retriever not in {item.id for item in spec.retrievers}:
            raise HTTPException(400, "Unknown retriever")
        if body.reranker not in {item.id for item in spec.rerankers}:
            raise HTTPException(400, "Unknown reranker")
        dataset_spec = next(item for item in spec.datasets if item.id == body.dataset)
        if body.task != dataset_spec.task:
            raise HTTPException(400, "Task does not match dataset")
        query_id = digest({"dataset": body.dataset, "question": body.question,
                           "task": body.task, "scope": body.corpus_scope,
                           "options": body.answer_options})[:24]
        query = QueryRecord(dataset_id=body.dataset, query_id=f"service:{query_id}",
                            text=body.question, split="service", task=body.task,
                            corpus_scope=body.corpus_scope, answer_options=body.answer_options)
        try:
            return runtime.run_one(query, body.retriever, body.reranker, body.generate)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
