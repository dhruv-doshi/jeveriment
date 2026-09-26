"""Minimal stage executor that reuses compatible immutable outputs."""

import fcntl
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timezone
from time import monotonic

from jev_eval.io import digest

from .artifacts import ArtifactStore
from .contracts import StageRecord


def execute_stage(
    store: ArtifactStore,
    *,
    run_id: str,
    stage: str,
    plugin_id: str,
    config: dict,
    upstream_hashes: tuple[str, ...],
    input_value,
    call: Callable,
    query_id: str | None = None,
    mode: str = "fixture",
):
    key = store.stage_key(run_id, stage, plugin_id, config, upstream_hashes, input_value)
    lock_path = store.root / "locks" / f"{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = store.cached(key)
        if cached is not None:
            return store.get(cached.output_hash), cached, True
        input_hash = store.put(input_value)
        started = monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            from opentelemetry import trace

            span_scope = trace.get_tracer("jev_eval.rag").start_as_current_span(f"rag.{stage}")
        except ImportError:
            span_scope = nullcontext(None)
        with span_scope as span:
            if span is not None:
                span.set_attribute("rag.stage", stage)
                span.set_attribute("rag.plugin", plugin_id)
                span.set_attribute("rag.run_id", run_id)
            try:
                output = call(input_value)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                store.put(error)
                from jev_eval.io import write_json

                write_json(store.root / "failures" / f"{key}.json", error)
                if span is not None:
                    span.record_exception(exc)
                raise
            context = span.get_span_context() if span is not None else None
            trace_id = f"{context.trace_id:032x}" if context and context.is_valid else None
        output_hash = store.put(output)
        record = StageRecord(
            run_id=run_id,
            stage=stage,
            plugin_id=plugin_id,
            query_id=query_id,
            config_hash=digest(config),
            upstream_hashes=upstream_hashes,
            input_hash=input_hash,
            output_hash=output_hash,
            status="success",
            mode=mode,
            elapsed_ms=(monotonic() - started) * 1000,
            started_at_utc=started_at,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            trace_id=trace_id,
        )
        store.record(key, record)
        return output, record, False
