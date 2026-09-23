import json
from types import SimpleNamespace

import pytest
import yaml

from jev_eval.config import Experiment
from jev_eval.data import ingest_beir
from jev_eval.io import read_json
from jev_eval.pipeline import prepare


class FixtureTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()

    def __call__(self, a, b=None, **kwargs):
        return {"input_ids": self.encode(a + (b or ""))}


def test_prepare_freezes_revisions_and_detects_config_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *a, **k: SimpleNamespace(
            sha="a" * 40, siblings=[], card_data=SimpleNamespace(license="fixture")
        ),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: FixtureTokenizer()
    )
    source = tmp_path / "source"
    (source / "qrels").mkdir(parents=True)
    (source / "corpus.jsonl").write_text(
        json.dumps({"_id": "001", "title": "Title", "text": "Evidence"}) + "\n"
    )
    (source / "queries.jsonl").write_text(
        json.dumps({"_id": "q", "text": "Claim"}) + "\n"
    )
    (source / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq\t001\t1\n"
    )
    config = Experiment(
        experiment_id="fixture",
        dataset="fixture",
        sample_queries=1,
        max_input_tokens=1024,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()))
    dest = prepare(path, source)
    assert read_json(dest / "resolved_config.json")["dense_revision"] == "a" * 40
    assert "001" in read_json(dest / "text_views.json")
    assert prepare(path, source) == dest
    path.write_text(yaml.safe_dump({**config.model_dump(), "seed": 99}))
    with pytest.raises(ValueError, match="changed"):
        prepare(path, source)


def test_failed_ingestion_does_not_leave_partial_corpus(tmp_path):
    source = tmp_path / "bad_source"
    (source / "qrels").mkdir(parents=True)
    (source / "corpus.jsonl").write_text(
        json.dumps({"_id": "001", "text": "Evidence"}) + "\n"
    )
    (source / "queries.jsonl").write_text(
        json.dumps({"_id": "q", "text": "Claim"}) + "\n"
    )
    (source / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq\tmissing\t1\n"
    )
    with pytest.raises(ValueError, match="missing corpus"):
        ingest_beir(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared" / "corpus.sqlite").exists()
    assert list((tmp_path / "prepared").iterdir()) == []
