"""Runs the real data/BM25/pool/evaluator pipeline on 20 authored documents.

Neural adapters are replaced with deterministic fixture vectors/scores. No model
downloads, real corpus reads, external API calls, or benchmark results occur.
"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jev_eval import pipeline
from jev_eval.config import Experiment
from jev_eval.data import ingest_beir
from jev_eval.io import digest, read_json, write_json


class FixtureDense:
    calls = 0

    def __init__(self, *args):
        pass

    def encode(self, texts, query=False):
        FixtureDense.calls += len(texts)
        vectors = np.array(
            [
                [1.0 + t.lower().count("tides"), 1.0 + t.lower().count("cake")]
                for t in texts
            ],
            dtype=np.float32,
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FixtureReranker:
    calls = 0

    def __init__(self, *args):
        pass

    def score(self, query, texts):
        FixtureReranker.calls += len(texts)
        return [float(t.lower().count("tides")) for t in texts]


def test_full_synthetic_pipeline_resume_and_failure_reporting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "DenseModel", FixtureDense)
    monkeypatch.setattr(pipeline, "LocalReranker", FixtureReranker)
    source = tmp_path / "fixture"
    (source / "qrels").mkdir(parents=True)
    docs = [
        {
            "_id": f"{i:03}",
            "title": f"Document {i}",
            "text": "Tides are caused by gravity." if i < 5 else "Cake contains flour.",
        }
        for i in range(20)
    ]
    (source / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (source / "queries.jsonl").write_text(
        "".join(
            json.dumps({"_id": f"q{i}", "text": "What causes tides?"}) + "\n"
            for i in range(8)
        )
    )
    (source / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"q{i}\t000\t2\nq{i}\t001\t1\nq{i}\t019\t0\n" for i in range(8))
    )
    config = Experiment(
        experiment_id="synthetic",
        dataset="fixture",
        sample_queries=8,
        depth=20,
        pool_size=10,
        selection_k=5,
    )
    import yaml

    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()))
    dest = pipeline.run_dir(config)
    ingest_beir(source, dest, "train", 8)
    views = {
        d["_id"]: {
            "text": d["title"] + "\n" + d["text"],
            "hash": digest(d["title"] + "\n" + d["text"]),
        }
        for d in docs
    }
    write_json(dest / "text_views.json", views)
    write_json(dest / "input_config.json", config.model_dump())
    write_json(
        dest / "resolved_config.json",
        {
            **config.model_dump(),
            "dense_revision": "fixture",
            "reranker_revision": "fixture",
        },
    )
    write_json(
        dest / "model_manifest.json", {config.reranker_model: {"revision": "fixture"}}
    )
    result = pipeline.retrieve(config_path)
    assert result == {"queries": 8, "corpus": 20}
    from jev_eval import scan

    class FixtureJev:
        calls = 0
        active = 0
        max_active = 0
        lock = threading.Lock()

        def __init__(self, *args):
            pass

        def sentinel(self):
            pass

        def evaluate(self, state, questions, provenance=None):
            with FixtureJev.lock:
                FixtureJev.calls += 1
                FixtureJev.active += 1
                FixtureJev.max_active = max(
                    FixtureJev.max_active, FixtureJev.active
                )
            try:
                time.sleep(0.002)
                score = 0.9 if "Tides" in state["document"]["text"] else 0.1
                return {"answers": {"candidate_000": {"noul": score}}}
            finally:
                with FixtureJev.lock:
                    FixtureJev.active -= 1

    monkeypatch.setattr(scan, "JevClient", FixtureJev)
    monkeypatch.setattr(
        scan.Settings, "load", classmethod(lambda cls: SimpleNamespace(model="jev"))
    )
    (tmp_path / "manifests").mkdir()
    write_json(
        tmp_path / "manifests" / "capabilities_typesafe.json",
        {"status": "noul_functional_checks_passed"},
    )
    write_json(dest / "jev_scan_scores.json", {"vercel_checkpoint": True})
    with pytest.raises(ValueError, match="approve-pairs 40"):
        scan.scan(config_path, 2, 39, 100, 100000)
    assert FixtureJev.calls == 0
    assert scan.scan(config_path, 2, 40, 100, 100000)["pairs"] == 40
    assert scan.scan(config_path, 2, 40, 100, 100000)["pairs"] == 40
    assert FixtureJev.calls == 40
    assert FixtureJev.max_active > 1
    assert scan.scan(config_path, 3, 60, 100, 100000)["pairs"] == 60
    assert FixtureJev.calls == 60
    assert read_json(dest / "jev_scan_scores.json") == {"vercel_checkpoint": True}
    comparison = scan.compare(config_path)
    assert comparison["jev_typesafe_full_scan"]["eligible_queries"] == 3
    assert comparison["dense_cosine"]["eligible_queries"] == 3
    assert (dest / "retrieval_typesafe_summary.json").exists()
    pools = read_json(dest / "pools.json")
    from jev_eval import decisions

    class FixtureSetJev:
        calls = 0

        def __init__(self, *args):
            pass

        def sentinel(self):
            pass

        def evaluate(self, state, questions, provenance=None):
            FixtureSetJev.calls += 1
            return {"answers": {"sufficient": {"noul": 0.9}}}

    monkeypatch.setattr(decisions, "JevClient", FixtureSetJev)
    write_json(
        dest / "scores_jev.json",
        {
            "status": "complete",
            "api_source": "typesafe_direct",
            "requested_model": "jev",
            "scores": {
                q: {
                    c["doc_id"]: (0.9 if int(c["doc_id"]) < 5 else 0.1)
                    for c in pool["candidates"]
                }
                for q, pool in pools.items()
            },
        },
    )
    assert decisions.benchmark(config_path, 0.5, 100, 100000)["sets"] == 8
    assert decisions.benchmark(config_path, 0.5, 100, 100000)["sets"] == 8
    assert FixtureSetJev.calls == 8
    decision_report = read_json(dest / "decision_benchmark.json")
    assert decision_report["pair_metrics"]["jev_threshold"]["recall"] == 1.0
    assert decision_report["set_proxy_metrics"]["jev_set"]["recall"] == 1.0
    original_hashes = {q: p["ordered_input_hash"] for q, p in pools.items()}
    assert pipeline.rerank(config_path, "qwen")["status"] == "complete"
    calls = FixtureReranker.calls
    pipeline.rerank(config_path, "qwen")
    assert calls == FixtureReranker.calls
    summary = pipeline.evaluate(config_path)
    assert summary["qwen"]["eligible_queries"] == 8
    assert summary["qwen"]["recall@10"] == 1.0
    assert (dest / "candidates.parquet").exists() and (dest / "scores.parquet").exists()
    saved = read_json(dest / "scores_qwen.json")
    saved["scores"].pop(next(iter(saved["scores"])))
    saved["status"] = "incomplete"
    write_json(dest / "scores_qwen.json", saved)
    summary = pipeline.evaluate(config_path)
    assert summary["qwen"]["failure_fraction"] == 1 / 8
    assert summary["qwen_operational_fallback"]["eligible_queries"] == 8
    assert {
        q: p["ordered_input_hash"] for q, p in read_json(dest / "pools.json").items()
    } == original_hashes
