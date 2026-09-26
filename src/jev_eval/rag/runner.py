"""Minimal stage executor that reuses compatible immutable outputs."""

import fcntl
from collections.abc import Callable
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
        try:
            output = call(input_value)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            store.put(error)
            from jev_eval.io import write_json

            write_json(store.root / "failures" / f"{key}.json", error)
            raise
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
        )
        store.record(key, record)
        return output, record, False
