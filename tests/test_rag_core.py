from decimal import Decimal

import pytest

from jev_eval.io import digest
from jev_eval.rag.artifacts import ArtifactStore
from jev_eval.rag.config import ExperimentSpec, load_experiment
from jev_eval.rag.contracts import CandidatePool, CandidateRecord, ChunkRecord
from jev_eval.rag.runner import execute_stage


def test_experiment_spec_and_matrix():
    config = load_experiment("configs/rag/experiment.yaml")
    assert isinstance(config, ExperimentSpec)
    assert len(config.datasets) == 5
    assert len(config.retrievers) * len(config.rerankers) * len(config.datasets) == 75
    assert config.budget_usd == Decimal("50")
    assert config.generator.model == "moonshotai/kimi-k2.5"


def test_chunk_and_candidate_pool_hashes():
    chunk = ChunkRecord(
        dataset_id="fixture",
        document_id="d1",
        chunk_id="d1:0",
        text="Evidence text",
        text_hash=digest("Evidence text"),
    )
    with pytest.raises(ValueError, match="hash"):
        ChunkRecord.model_validate({**chunk.model_dump(), "text_hash": "wrong"})


def test_candidate_pool_rejects_duplicate_and_mixed_queries():
    first = CandidateRecord(
        query_id="q",
        chunk_id="d1:0",
        rank=1,
        score=0.5,
        text_hash=digest("text"),
        source_document_id="d1",
    )
    pool = CandidatePool.build("q", "bm25", [first])
    assert pool.ordered_hash == digest([first.model_dump()])
    with pytest.raises(ValueError, match="Duplicate"):
        CandidatePool.build("q", "bm25", [first, first.model_copy(update={"rank": 2})])
    with pytest.raises(ValueError, match="Mixed"):
        CandidatePool.build("q", "bm25", [first.model_copy(update={"query_id": "other"})])


def test_replay_and_targeted_invalidation(tmp_path):
    store = ArtifactStore(tmp_path)
    calls = []

    def double(value):
        calls.append(value)
        return {"doubled": value["number"] * 2}

    kwargs = {
        "run_id": "run1",
        "stage": "rerank",
        "plugin_id": "jev_formula",
        "config": {"method": "sort"},
        "upstream_hashes": (),
        "input_value": {"number": 3},
        "call": double,
    }
    first, record, cached = execute_stage(store, **kwargs)
    assert first == {"doubled": 6} and not cached
    second, reused, cached = execute_stage(store, **kwargs)
    assert second == first and cached and reused == record and len(calls) == 1
    changed = {**kwargs, "config": {"method": "reverse"}}
    _, _, cached = execute_stage(store, **changed)
    assert not cached and len(calls) == 2
    artifact = tmp_path / "blobs" / f"{record.output_hash}.json"
    artifact.write_text('{"doubled":7}')
    with pytest.raises(ValueError, match="hash"):
        store.get(record.output_hash)
