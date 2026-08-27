"""Durable accounting for every attempted fitness-optimization trial."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from utils.experiment_manifest import canonicalize


class FitnessTrialLedger:
    """Append-only project ledger; trial records are durable before the next trial."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.root / "fitness_optimization_trials.jsonl"
        self.runs_path = self.root / "fitness_optimization_runs.jsonl"

    def start_run(
        self,
        *,
        parent_experiment_id: str,
        configuration: Mapping[str, Any],
        artifact_locations: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = {
            "event": "started",
            "status": "started",
            "optimization_run_id": str(uuid.uuid4()),
            "parent_experiment_id": str(parent_experiment_id),
            "created_at_utc": _now(),
            "configuration": canonicalize(configuration),
            "artifact_locations": canonicalize(artifact_locations),
        }
        self._append(self.runs_path, record)
        return record

    def record_trial(
        self,
        *,
        optimization_run_id: str,
        parent_experiment_id: str,
        trial_index: int,
        fitness_configuration: Mapping[str, Any],
        candidate_pool_identity: Mapping[str, Any],
        seed: int,
        status: str,
        objective_value: float,
        validation_metrics: Mapping[str, Any],
        selected_factor_count: int,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed"}:
            raise ValueError(f"Unsupported trial status: {status}")
        now = _now()
        record = {
            "event": "trial_finished",
            "optimization_run_id": str(optimization_run_id),
            "trial_id": str(uuid.uuid4()),
            "parent_experiment_id": str(parent_experiment_id),
            "trial_index": int(trial_index),
            "fitness_configuration": canonicalize(fitness_configuration),
            "candidate_pool_identity": canonicalize(candidate_pool_identity),
            "seed": int(seed),
            "status": status,
            "objective_value": float(objective_value),
            "validation_metrics": canonicalize(validation_metrics),
            "selected_factor_count": int(selected_factor_count),
            "failure_reason": failure_reason,
            "created_at_utc": now,
            "completed_at_utc": now,
        }
        self._append(self.trials_path, record)
        return record

    def complete_run(self, started: Mapping[str, Any], *, summary: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "event": "completed",
            "status": "completed",
            "optimization_run_id": started["optimization_run_id"],
            "parent_experiment_id": started["parent_experiment_id"],
            "completed_at_utc": _now(),
            "summary": canonicalize(summary),
        }
        self._append(self.runs_path, record)
        return record

    def fail_run(self, started: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
        record = {
            "event": "failed",
            "status": "failed",
            "optimization_run_id": started["optimization_run_id"],
            "parent_experiment_id": started["parent_experiment_id"],
            "failed_at_utc": _now(),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        self._append(self.runs_path, record)
        return record

    def trials_for_run(self, optimization_run_id: str) -> list[dict[str, Any]]:
        if not self.trials_path.exists():
            return []
        return [
            record
            for line in self.trials_path.read_text(encoding="utf-8").splitlines()
            if line and (record := json.loads(line)).get("optimization_run_id") == optimization_run_id
        ]

    @staticmethod
    def _append(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(canonicalize(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
