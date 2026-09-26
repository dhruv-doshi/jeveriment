from jev_eval.rag.artifacts import ArtifactStore
from jev_eval.rag.runner import execute_stage
from jev_eval.rag.telemetry import setup_tracing


def test_stage_span_is_persisted_without_input_payload(tmp_path):
    setup_tracing(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    _, record, _ = execute_stage(store, run_id="run", stage="retrieve", plugin_id="fixture",
                                 config={}, upstream_hashes=(), input_value={"private": "payload"},
                                 call=lambda _: {"score": 1})
    assert record.trace_id
    spans = list((tmp_path / "spans").glob("*.json"))
    assert len(spans) == 1
    assert "payload" not in spans[0].read_text()
