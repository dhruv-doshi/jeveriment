import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator

from .contracts import Contract


class Settings(Contract):
    api_key: SecretStr
    max_cost_usd: Decimal
    base_url: str = "https://ai-gateway.vercel.sh/typesafe"
    model: str = "typesafe-ai/jev"

    @classmethod
    def load(cls):
        load_dotenv(Path.cwd() / ".env", override=False)
        key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
        if not key:
            raise ValueError("AI_GATEWAY_API_KEY is missing")
        try:
            budget = Decimal(os.environ.get("JEV_MAX_COST_USD", ""))
        except InvalidOperation:
            raise ValueError("JEV_MAX_COST_USD must be a nonnegative amount") from None
        if not budget.is_finite() or budget < 0:
            raise ValueError("JEV_MAX_COST_USD must be finite and nonnegative")
        base = os.environ.get(
            "JEV_API_BASE_URL", cls.model_fields["base_url"].default
        ).rstrip("/")
        if base != "https://ai-gateway.vercel.sh/typesafe":
            raise ValueError(
                "This adapter only sends credentials to the configured Vercel host"
            )
        return cls(
            api_key=key,
            max_cost_usd=budget,
            base_url=base,
            model=os.environ.get("JEV_MODEL", "typesafe-ai/jev"),
        )


class Experiment(Contract):
    experiment_id: str
    dataset: str = "scifact"
    split: str = "train"
    sample_queries: int | None = Field(default=30, gt=0)
    seed: int = 1729
    confirmatory: bool = False
    corpus_revision: str | None = None
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dense_revision: str | None = None
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_revision: str | None = None
    jev_revision: str | None = None
    depth: int = Field(default=1000, gt=0)
    pool_size: int = Field(default=50, gt=0)
    selection_k: int = Field(default=10, gt=0)
    max_document_tokens: int = Field(default=384, gt=0)
    max_input_tokens: int = Field(default=512, gt=0)
    batch_size: int = Field(default=4, gt=0)
    device: str = "auto"
    gain: str = "linear"
    binary_threshold: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=5, ge=1, le=5)
    token_budget: int = Field(default=2000000, gt=0)
    request_budget: int = Field(default=1800, gt=0)
    tuning_enabled: bool = False
    task: str = "scientific_claim_evidence"
    exclude_self_match: bool = False
    jev_mode: str = "independent"
    contextual_batch_size: int = Field(default=5, ge=1, le=50)
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)

    @model_validator(mode="after")
    def gates(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.experiment_id):
            raise ValueError("experiment_id must be a plain name, not a path")
        if self.jev_mode not in {
            "independent",
            "contextual",
            "structured",
            "contextual_structured",
        }:
            raise ValueError("Unsupported Jev mode")
        if self.task not in {
            "scientific_claim_evidence",
            "question_answering",
            "counterargument",
            "duplicate_question",
        }:
            raise ValueError("Unsupported task rubric")
        if self.split not in {"train", "dev", "test"} or self.gain not in {
            "linear",
            "exponential",
        }:
            raise ValueError("Unsupported split or gain mapping")
        if self.selection_k > self.pool_size or self.pool_size > self.depth:
            raise ValueError("Require selection_k <= pool_size <= depth")
        if self.confirmatory and not all(
            [
                self.corpus_revision,
                self.dense_revision,
                self.reranker_revision,
                self.jev_revision,
            ]
        ):
            raise ValueError(
                "Confirmatory runs require immutable corpus/model revisions"
            )
        if self.confirmatory and any(
            not re.fullmatch(r"[a-f0-9]{40,64}", rev)
            for rev in (
                self.corpus_revision,
                self.dense_revision,
                self.reranker_revision,
            )
        ):
            raise ValueError(
                "Confirmatory corpus/local-model revisions must be content/commit hashes"
            )
        if self.confirmatory and self.jev_revision in {
            "typesafe-ai/jev",
            "jev-latest",
            "latest",
            "main",
        }:
            raise ValueError("Mutable Jev aliases cannot serve as immutable revisions")
        if self.tuning_enabled and self.split == "test":
            raise ValueError("Test split must never be used for tuning")
        return self


def load_config(path):
    return Experiment.model_validate(yaml.safe_load(Path(path).read_text()))
