import pytest

from jev_eval.expansion import import_external_scores
from jev_eval.generation import generate_rag
from jev_eval.io import digest, read_json, write_json


class FixtureGenerator:
    calls = 0

    def count(self, text):
        return len(text.split())

    def generate(self, prompt):
        self.calls += 1
        return {
            "text": "Fixture answer [d]",
            "input_tokens": len(prompt.split()),
            "output_tokens": 4,
        }


def test_rag_evidence_provenance_and_resume(tmp_path):
    write_json(tmp_path / "queries.json", {"q": {"text": "Question"}})
    write_json(
        tmp_path / "text_views.json",
        {"d": {"text": "Evidence", "hash": digest("Evidence")}},
    )
    write_json(tmp_path / "selection_rrf.json", {"q": [{"doc_id": "d"}]})
    config = {"model": "fixture", "revision": "a" * 40}
    generator = FixtureGenerator()
    generate_rag(tmp_path, config, ["rrf"], generator)
    generate_rag(tmp_path, config, ["rrf"], generator)
    assert generator.calls == 1
    row = next(iter(read_json(tmp_path / "rag_generations.json").values()))
    assert (
        row["evidence_ids"] == ["d"]
        and row["audit_status"] == "pending_independent_human_audit"
    )


def test_external_scores_cannot_change_pool(tmp_path):
    write_json(
        tmp_path / "pools.json",
        {"q": {"ordered_input_hash": "frozen", "candidates": [{"doc_id": "d"}]}},
    )
    artifact = {
        "metadata": {
            "model": "fixture",
            "revision": "abc",
            "provider": "remote",
            "input_view_policy": "shared",
            "resources": {"seconds": 1},
        },
        "scores": {"q": {"d": 1.0}},
        "ordered_pool_hashes": {"q": "changed"},
    }
    write_json(tmp_path / "artifact.json", artifact)
    with pytest.raises(ValueError, match="different pools"):
        import_external_scores(tmp_path, "fixture", tmp_path / "artifact.json")
