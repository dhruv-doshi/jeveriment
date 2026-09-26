"""Generic cross-plugin data records with stable IDs and provenance."""

from typing import Literal

from pydantic import Field, model_validator

from jev_eval.io import digest

from .config import StrictModel


class QueryRecord(StrictModel):
    dataset_id: str
    query_id: str
    text: str
    split: str
    task: str
    corpus_scope: str | None = None
    answer_options: dict[str, str] = Field(default_factory=dict)


class GoldRecord(StrictModel):
    dataset_id: str
    query_id: str
    answers: tuple[str, ...] = ()
    qrels: dict[str, int] = Field(default_factory=dict)
    evidence_sets: tuple[tuple[str, ...], ...] = ()
    metric_availability: dict[str, str] = Field(default_factory=dict)


class ChunkRecord(StrictModel):
    dataset_id: str
    document_id: str
    chunk_id: str
    text: str
    text_hash: str
    source_offset: int | None = Field(default=None, ge=0)
    title: str | None = None
    source_uri: str | None = None
    corpus_scope: str | None = None

    @model_validator(mode="after")
    def validate_hash(self):
        if self.text_hash != digest(self.text):
            raise ValueError("Chunk text hash mismatch")
        return self


class CandidateRecord(StrictModel):
    query_id: str
    chunk_id: str
    rank: int = Field(ge=1)
    score: float
    text_hash: str
    source_document_id: str


class CandidatePool(StrictModel):
    query_id: str
    retriever_id: str
    candidates: tuple[CandidateRecord, ...]
    ordered_hash: str

    @model_validator(mode="after")
    def validate_pool(self):
        ids = [c.chunk_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate chunk ID")
        if any(c.query_id != self.query_id or c.rank != i for i, c in enumerate(self.candidates, 1)):
            raise ValueError("Mixed query or nonconsecutive ranks")
        if self.ordered_hash != digest([c.model_dump() for c in self.candidates]):
            raise ValueError("Candidate pool hash mismatch")
        return self

    @classmethod
    def build(cls, query_id: str, retriever_id: str, candidates: list[CandidateRecord]):
        return cls(
            query_id=query_id,
            retriever_id=retriever_id,
            candidates=tuple(candidates),
            ordered_hash=digest([c.model_dump() for c in candidates]),
        )


class StageRecord(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    stage: str
    plugin_id: str
    query_id: str | None = None
    config_hash: str
    upstream_hashes: tuple[str, ...]
    input_hash: str
    output_hash: str
    status: Literal["success", "failed", "skipped"]
    mode: Literal["live", "replay", "fixture"]
    elapsed_ms: float = Field(ge=0)
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    trace_id: str | None = None
    error: str | None = None


class RerankResult(StrictModel):
    query_id: str
    pool_hash: str
    reranker_id: str
    ordered_ids: tuple[str, ...]
    scores: dict[str, float]
    raw_response_hashes: dict[str, str] = Field(default_factory=dict)
    raw_response_paths: dict[str, str] = Field(default_factory=dict)
    missing_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_ranking(self):
        if len(set(self.ordered_ids)) != len(self.ordered_ids):
            raise ValueError("Duplicate reranked IDs")
        if set(self.scores) | set(self.missing_ids) != set(self.ordered_ids):
            raise ValueError("Reranker IDs do not match pool")
        if self.raw_response_paths and set(self.raw_response_paths) != set(self.scores):
            raise ValueError("Raw response paths do not cover scored IDs")
        return self


class ContextSelection(StrictModel):
    query_id: str
    selected_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    text: str
    estimated_tokens: int = Field(ge=0)
    budget_tokens: int = Field(gt=0)


class GenerationResult(StrictModel):
    query_id: str
    model_requested: str
    model_resolved: str
    answer: str
    citations: tuple[str, ...] = ()
    usage: dict[str, int] = Field(default_factory=dict)
    provider_generation_id: str | None = None
    provider_timing_ms: float | None = None
    provider_metadata_hash: str | None = None
    raw_response_hash: str


class JudgeResult(StrictModel):
    query_id: str
    model_requested: str
    model_resolved: str
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    answer_correctness: float | None = Field(default=None, ge=0, le=1)
    rationale: str
    raw_response_hash: str
    provider_metadata_hash: str | None = None
