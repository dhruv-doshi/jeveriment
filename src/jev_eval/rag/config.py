"""Frozen experiment specification. Loading a config cannot run a model."""

import re
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PluginSpec(StrictModel):
    id: str
    kind: str
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)


class DatasetSpec(StrictModel):
    id: str
    source: str
    revision: str | None = None
    task: str


class GeneratorSpec(StrictModel):
    provider: str
    model: str
    evidence_tokens: int = Field(default=4096, gt=0)
    max_output_tokens: int = Field(default=768, gt=0)


class ExperimentSpec(StrictModel):
    experiment_id: str
    seed: int = 1729
    budget_usd: Decimal = Field(default=Decimal("50"), ge=0)
    datasets: tuple[DatasetSpec, ...]
    retrievers: tuple[PluginSpec, ...]
    rerankers: tuple[PluginSpec, ...]
    generator: GeneratorSpec
    candidate_depth: int = Field(default=100, gt=0)
    max_saved_depth: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def validate_design(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.experiment_id):
            raise ValueError("experiment_id must be a plain name")
        if self.candidate_depth > self.max_saved_depth:
            raise ValueError("candidate_depth cannot exceed max_saved_depth")
        for collection in (self.datasets, self.retrievers, self.rerankers):
            ids = [item.id for item in collection]
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("Each plugin collection needs unique IDs")
        if any(r.kind != "retriever" for r in self.retrievers):
            raise ValueError("Retriever kinds must be 'retriever'")
        if any(r.kind != "reranker" for r in self.rerankers):
            raise ValueError("Reranker kinds must be 'reranker'")
        return self


def load_experiment(path: str | Path) -> ExperimentSpec:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Experiment config must be a mapping")
    return ExperimentSpec.model_validate(value)
