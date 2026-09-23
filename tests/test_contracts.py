import pytest

from jev_eval.config import Experiment
from jev_eval.contracts import Document, Qrel
from jev_eval.io import digest


def test_ids_and_hashes():
    doc = Document(
        doc_id="001",
        title="T",
        text="body",
        content_sha256=digest({"title": "T", "text": "body"}),
    )
    assert Document.model_validate_json(doc.model_dump_json()).doc_id == "001"
    with pytest.raises(ValueError):
        Document(**{**doc.model_dump(), "doc_id": 1})
    with pytest.raises(ValueError):
        Document(**{**doc.model_dump(), "text": "changed"})


def test_judgments_and_confirmatory_gates():
    assert Qrel(query_id="q", doc_id="d", grade=0, judgment_status="judged").grade == 0
    with pytest.raises(ValueError):
        Qrel(query_id="q", doc_id="d", grade=0, judgment_status="unjudged")
    with pytest.raises(ValueError):
        Experiment(experiment_id="x", confirmatory=True)
    with pytest.raises(ValueError):
        Experiment(experiment_id="x", split="test", tuning_enabled=True)
