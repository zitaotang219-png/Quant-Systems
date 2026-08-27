"""Append-only experiment lifecycle accounting independent of research results."""

from __future__ import annotations

import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from utils.experiment_manifest import canonicalize, config_hash


NON_MATERIAL_KEYS = {"output_dir", "run_dir", "artifact_locations", "timestamp", "timestamp_utc", "directory"}


def research_configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    """Fingerprint material research inputs while excluding output-location metadata."""
    return config_hash(_materialize(configuration))


def _materialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items() if str(key) not in NON_MATERIAL_KEYS}
    if isinstance(value, (list, tuple)):
        return [_materialize(item) for item in value]
    return value


class ExperimentLedger:
    """Writes immutable lifecycle events to a project-level JSONL ledger."""

    def __init__(self, root: str | Path = "reports") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "experiments.jsonl"

    def start(self, *, configuration: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "event": "started",
            "experiment_id": str(uuid.uuid4()),
            "search_run_id": str(uuid.uuid4()),
            "configuration_fingerprint": research_configuration_fingerprint(configuration),
            "created_at_utc": _now(),
            "status": "started",
            "configuration": canonicalize(_materialize(configuration)),
            "metadata": canonicalize(metadata),
        }
        self._append(record)
        return record

    def complete(self, started: Mapping[str, Any], *, results: Mapping[str, Any]) -> dict[str, Any]:
        return self._transition(started, "completed", {"results": canonicalize(results), "completed_at_utc": _now()})

    def fail(self, started: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
        return self._transition(started, "failed", {"failed_at_utc": _now(), "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})

    def _transition(self, started: Mapping[str, Any], status: str, extra: Mapping[str, Any]) -> dict[str, Any]:
        record = {"event": status, "experiment_id": started["experiment_id"], "search_run_id": started["search_run_id"], "configuration_fingerprint": started["configuration_fingerprint"], "status": status, **extra}
        self._append(record)
        return record

    def _append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(canonicalize(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
