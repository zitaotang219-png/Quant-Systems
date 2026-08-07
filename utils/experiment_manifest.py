from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA_VERSION = 1


def canonicalize(value: Any) -> Any:
    """Convert config values into a stable, JSON-serializable representation."""

    if is_dataclass(value):
        return canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: _canonical_json(item))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return canonicalize(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported config value for deterministic hashing: {type(value).__name__}")


def config_hash(config: Any) -> str:
    """Return a deterministic SHA-256 hash for a configuration value."""

    return hashlib.sha256(_canonical_json(canonicalize(config)).encode("utf-8")).hexdigest()


def create_experiment_directory(requested_path: str | Path) -> Path:
    """Create an output directory without reusing a previous experiment."""

    requested = Path(requested_path)
    try:
        requested.mkdir(parents=True, exist_ok=False)
        return requested
    except FileExistsError:
        pass

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for suffix in range(1000):
        discriminator = f"_{suffix:03d}" if suffix else ""
        candidate = requested.with_name(f"{requested.name}_{timestamp}{discriminator}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a unique experiment directory for {requested}")


class ExperimentManifest:
    """Build and persist the audit record for one research experiment."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        random_seed: int,
        cli_command: str | None = None,
        configuration_path: str | Path | None = None,
        repo_root: str | Path | None = None,
        input_paths: Iterable[str | Path | None] = (),
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent.parent).resolve()
        self.path = self.run_dir / "experiment_manifest.json"
        config_snapshot_path = str((self.run_dir / "config.yaml").resolve())
        source_config_path = _resolved_path(configuration_path)
        self.payload: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "git": _git_information(self.repo_root),
            "runtime": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "os": platform.platform(),
                "packages": _package_versions(),
            },
            "experiment": {
                "name": self.run_dir.name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "random_seed": int(random_seed),
                "cli_command": cli_command or subprocess.list2cmdline(sys.argv),
                "configuration_path": source_config_path or config_snapshot_path,
                "source_configuration_path": source_config_path,
                "configuration_snapshot": config_snapshot_path,
                "status": "running",
            },
            "data": {
                "inputs": [_input_record(path) for path in input_paths if path],
                "dataset_metadata": canonicalize(dataset_metadata or {}),
            },
            "research": {
                "generated_factor_count": 0,
                "selected_factor_count": 0,
                "factor_registry_version": None,
            },
            "config_hash": None,
        }

    def set_config(self, config: Any) -> str:
        normalized = canonicalize(config)
        digest = config_hash(normalized)
        self.payload["config_hash"] = digest
        # JSON is valid YAML 1.2 and avoids adding a runtime dependency.
        (self.run_dir / "config.yaml").write_text(
            json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return digest

    def set_dataset_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.payload["data"]["dataset_metadata"] = canonicalize(metadata)

    def set_research_results(
        self,
        *,
        generated_factor_count: int,
        selected_factors: Iterable[str],
    ) -> None:
        expressions = [str(expression) for expression in selected_factors]
        version_payload = {
            "config_hash": self.payload.get("config_hash"),
            "selected_factors": expressions,
        }
        self.payload["research"] = {
            "generated_factor_count": int(generated_factor_count),
            "selected_factor_count": len(expressions),
            "factor_registry_version": config_hash(version_payload),
        }

    def save(self, *, status: str | None = None) -> Path:
        if status is not None:
            self.payload["experiment"]["status"] = str(status)
        temporary_path = self.path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(canonicalize(self.payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.path)
        return self.path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _resolved_path(path: str | Path | None) -> str | None:
    return str(Path(path).expanduser().resolve()) if path else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(raw_path: str | Path) -> dict[str, Any]:
    path = Path(raw_path).expanduser().resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        record.update({"type": "missing", "sha256": None})
        return record
    if path.is_file():
        record.update({"type": "file", "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        return record

    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix())
    children = []
    digest = hashlib.sha256()
    for item in files:
        relative_path = item.relative_to(path).as_posix()
        file_hash = _sha256_file(item)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        children.append({"path": relative_path, "size_bytes": item.stat().st_size, "sha256": file_hash})
    record.update({"type": "directory", "file_count": len(children), "sha256": digest.hexdigest(), "files": children})
    return record


def _git_information(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "commit_hash": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _package_versions() -> dict[str, str]:
    discovered: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            discovered.setdefault(str(name).lower(), set()).add(distribution.version)
    packages = {
        name: ",".join(sorted(versions))
        for name, versions in discovered.items()
    }
    # In environments with duplicate metadata, the imported module is the
    # authoritative version that participated in this experiment.
    for package_name, module_name in (("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn")):
        module = sys.modules.get(module_name)
        version = getattr(module, "__version__", None) if module is not None else None
        if version:
            packages[package_name] = str(version)
    return dict(sorted(packages.items()))
