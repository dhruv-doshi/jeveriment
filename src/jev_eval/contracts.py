from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from .io import digest


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Document(Contract):
    doc_id: StrictStr
    title: StrictStr = ""
    text: StrictStr
    content_sha256: StrictStr

    @model_validator(mode="after")
    def check_hash(self):
        if self.content_sha256 != digest({"title": self.title, "text": self.text}):
            raise ValueError("Document content hash mismatch")
        return self


class Query(Contract):
    query_id: StrictStr
    text: StrictStr
    split: Literal["train", "dev", "test"]


class Qrel(Contract):
    query_id: StrictStr
    doc_id: StrictStr
    grade: float | None = None
    judgment_status: Literal["judged", "unjudged", "insufficient_information"]

    @model_validator(mode="after")
    def check_status(self):
        if (self.judgment_status == "judged") != (self.grade is not None):
            raise ValueError("Only judged pairs have numeric grades")
        if self.grade is not None and self.grade < 0:
            raise ValueError("Judged relevance grades must be nonnegative")
        return self


class Candidate(Contract):
    query_id: StrictStr
    doc_id: StrictStr
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_score: float
    dense_score: float
    in_bm25: bool
    in_dense: bool
    text_view_hash: StrictStr

    @model_validator(mode="after")
    def source_membership(self):
        if self.in_bm25 != (self.bm25_rank is not None) or self.in_dense != (
            self.dense_rank is not None
        ):
            raise ValueError("Source ranks and membership flags disagree")
        return self


class Pool(Contract):
    query_id: StrictStr
    candidates: tuple[Candidate, ...]
    membership_hash: StrictStr
    ordered_input_hash: StrictStr

    @model_validator(mode="after")
    def check(self):
        ids = [c.doc_id for c in self.candidates]
        if len(set(ids)) != len(ids) or any(
            c.query_id != self.query_id for c in self.candidates
        ):
            raise ValueError("Duplicate IDs or mixed queries in pool")
        if self.membership_hash != digest(sorted(ids)):
            raise ValueError("Pool membership hash mismatch")
        if self.ordered_input_hash != digest([c.model_dump() for c in self.candidates]):
            raise ValueError("Pool ordered input hash mismatch")
        return self

    @classmethod
    def build(cls, query_id, candidates):
        return cls(
            query_id=query_id,
            candidates=tuple(candidates),
            membership_hash=digest(sorted(c.doc_id for c in candidates)),
            ordered_input_hash=digest([c.model_dump() for c in candidates]),
        )
