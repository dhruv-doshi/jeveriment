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


class ChunkRecord(StrictModel):
    dataset_id: str
    document_id: str
    chunk_id: str
    text: str
    text_hash: str
    source_offset: int | None = Field(default=None, ge=0)

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
    error: str | None = None
