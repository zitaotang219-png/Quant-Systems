from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AlphaMiningConfig, SelectedFactor
from .dsl import parse_expression


class FactorRegistry:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def save(
        self,
        selected_factors: list[SelectedFactor],
        config: AlphaMiningConfig,
        panel: pd.DataFrame,
        search_statistics: dict[str, int | float] | None = None,
    ) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = self.base_dir / config.registry.pkl_name
        csv_path = self.base_dir / config.registry.csv_name
        metadata_path = self.base_dir / config.registry.metadata_name

        payload = []
        for factor in selected_factors:
            payload.append(
                {
                    "expression": factor.expression,
                    "node": factor.node,
                    "expression_text": factor.expression,
                    "direction": factor.direction,
                    "fitness": factor.fitness,
                    "metrics": factor.metrics,
                    "complexity": factor.complexity,
                    "finite_ratio": factor.finite_ratio,
                }
            )

        with pkl_path.open("wb") as handle:
            pickle.dump(payload, handle)

        pd.DataFrame([factor.summary_row() for factor in selected_factors]).to_csv(csv_path, index=False)

        metadata = {
            "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
            "data_range": {
                "date_min": str(pd.to_datetime(panel["date"]).min()),
                "date_max": str(pd.to_datetime(panel["date"]).max()),
                "row_count": int(len(panel)),
                "symbol_count": int(panel["symbol"].nunique()) if "symbol" in panel.columns else 0,
            },
            "data_fingerprint": _panel_fingerprint(panel),
            "code_fingerprint": _code_fingerprint(),
            "config": config.to_dict(),
            "factors": [factor.summary_row() for factor in selected_factors],
            "search_statistics": dict(search_statistics or {}),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    def load(self, config: AlphaMiningConfig) -> list[SelectedFactor]:
        pkl_path = self.base_dir / config.registry.pkl_name
        if not pkl_path.exists():
            return []
        with pkl_path.open("rb") as handle:
            payload: list[dict[str, Any]] = pickle.load(handle)
        return [
            SelectedFactor(
                expression=item["expression"],
                node=_load_node(item),
                direction=int(item["direction"]),
                fitness=float(item["fitness"]),
                metrics=dict(item["metrics"]),
                complexity=int(item["complexity"]),
                finite_ratio=float(item["finite_ratio"]),
            )
            for item in payload
        ]

    def load_metadata(self, config: AlphaMiningConfig) -> dict[str, Any]:
        metadata_path = self.base_dir / config.registry.metadata_name
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_node(item: dict[str, Any]):
    expression = str(item.get("expression_text") or item.get("expression") or "")
    if expression:
        try:
            return parse_expression(expression)
        except Exception:
            pass
    if "node" in item:
        return item["node"]
    raise ValueError("Registry factor payload is missing both expression text and serialized node.")


def _panel_fingerprint(panel: pd.DataFrame) -> str:
    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _code_fingerprint() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    tracked_files = [
        repo_root / "alpha_mining" / "config.py",
        repo_root / "alpha_mining" / "dsl.py",
        repo_root / "alpha_mining" / "evaluator.py",
        repo_root / "alpha_mining" / "gp_generator.py",
        repo_root / "alpha_mining" / "pipeline.py",
        repo_root / "execution_crypto" / "paper.py",
        repo_root / "backtest" / "engine.py",
    ]
    return {
        str(path.relative_to(repo_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked_files
        if path.exists()
    }
