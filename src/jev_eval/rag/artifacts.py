"""Atomic content-addressed stage artifacts for selective replay."""

from pathlib import Path

from jev_eval.io import digest, read_json, write_json

from .contracts import StageRecord


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, value) -> str:
        key = digest(value)
        path = self.root / "blobs" / f"{key}.json"
        if path.exists():
            if read_json(path) != value:
                raise ValueError("Artifact hash collision or corruption")
        else:
            write_json(path, value)
        return key

    def get(self, key: str):
        value = read_json(self.root / "blobs" / f"{key}.json")
        if digest(value) != key:
            raise ValueError("Artifact content hash mismatch")
        return value

    @staticmethod
    def stage_key(run_id: str, stage: str, plugin_id: str, config, upstream: tuple[str, ...], input_value) -> str:
        return digest(
            {
                "run_id": run_id,
                "stage": stage,
                "plugin_id": plugin_id,
                "config": config,
                "upstream": upstream,
                "input": input_value,
            }
        )

    def record(self, key: str, record: StageRecord) -> None:
        path = self.root / "stages" / f"{key}.json"
        if path.exists():
            if StageRecord.model_validate(read_json(path)) != record:
                raise ValueError("Stage key already used for different output")
        else:
            write_json(path, record.model_dump())

    def cached(self, key: str) -> StageRecord | None:
        path = self.root / "stages" / f"{key}.json"
        if not path.exists():
            return None
        record = StageRecord.model_validate(read_json(path))
        if record.status == "success":
            self.get(record.output_hash)
            return record
        return None
