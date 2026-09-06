"""Lightweight factor research evaluation, independent of portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from backtest.trading_convention import DEFAULT_TRADING_CONVENTION, TradingConvention

from .config import FitnessConfig
from .dsl import FactorNode
from .evaluation_types import EvaluationResult, VERY_BAD_FITNESS


@dataclass(frozen=True)
class WindowResearchContext:
    """Prepared point-in-time data and split identity for one research window."""

    name: str
    panel: pd.DataFrame = field(compare=False, repr=False)
    split_map: dict[pd.Timestamp, str] = field(compare=False, repr=False)
    fingerprint: str


@dataclass
class FactorResearchEvaluator:
    """Evaluate predictive information without constructing a strategy portfolio."""

    trading_convention: TradingConvention = DEFAULT_TRADING_CONVENTION
    min_finite_ratio: float = 0.6
    min_std: float = 1e-6
    min_abs_rank_ic: float = 0.005
    max_signal_turnover: float = 2.5
    fitness_config: FitnessConfig = field(default_factory=FitnessConfig)
    cache: dict[tuple[str, str, str], EvaluationResult] = field(default_factory=dict)
    evaluation_calls: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _contexts_by_panel_id: dict[int, WindowResearchContext] = field(default_factory=dict, repr=False)

    def prepare_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        prepared = panel if "future_return" in panel.columns else _prepare_research_panel(panel, self.trading_convention)
        self._context_for_panel(prepared)
        return prepared

    def create_context(
        self,
        panel: pd.DataFrame,
        *,
        name: str,
        split_map: dict[pd.Timestamp, str] | None = None,
        split_label: str = "validation",
    ) -> WindowResearchContext:
        prepared = self.prepare_panel(panel)
        resolved_split_map = (
            split_map
            if split_map is not None
            else {
                pd.Timestamp(date): split_label
                for date in pd.to_datetime(prepared["date"], utc=False).dropna().unique()
            }
        )
        context = WindowResearchContext(
            name=str(name),
            panel=prepared,
            split_map={pd.Timestamp(date): str(label) for date, label in resolved_split_map.items()},
            fingerprint=_panel_fingerprint(prepared),
        )
        self._contexts_by_panel_id[id(prepared)] = context
        return context

    def fast_filter(self, node: FactorNode, panel: pd.DataFrame) -> EvaluationResult:
        return self.evaluate_context(node, self._context_for_panel(self.prepare_panel(panel)), mode="fast")

    def evaluate(self, node: FactorNode, panel: pd.DataFrame) -> EvaluationResult:
        context = self.create_context(panel, name="default_research", split_map=_split_dates(panel["date"]))
        return self.evaluate_context(node, context, mode="research")

    def evaluate_with_splits(
        self,
        node: FactorNode,
        panel: pd.DataFrame,
        split_map: dict[pd.Timestamp, str],
        *,
        context_name: str = "research",
    ) -> EvaluationResult:
        context = self.create_context(panel, name=context_name, split_map=split_map)
        return self.evaluate_context(node, context, mode="research")

    def evaluate_context(
        self,
        node: FactorNode,
        context: WindowResearchContext,
        *,
        mode: str,
    ) -> EvaluationResult:
        if mode not in {"fast", "research"}:
            raise ValueError(f"Unknown factor research mode: {mode}")
        expression = node.describe()
        key = (expression, _context_identity(context), self._specification_key(mode))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.evaluation_calls[key] = self.evaluation_calls.get(key, 0) + 1

        prepared = context.panel
        raw_values = node.evaluate(prepared).astype(float)
        finite_ratio = float(np.isfinite(raw_values.to_numpy(dtype=float, na_value=np.nan)).mean())
        if finite_ratio < self.min_finite_ratio:
            return self._store(key, self._reject(node, raw_values, finite_ratio, "finite_ratio"))

        value_std = float(np.nanstd(raw_values.to_numpy(dtype=float, na_value=np.nan)))
        if not np.isfinite(value_std) or value_std < self.min_std:
            return self._store(key, self._reject(node, raw_values, finite_ratio, "variance"))

        daily_ic = _daily_cross_sectional_rank_ic(prepared["date"], raw_values, prepared["future_return"])
        raw_rank_ic = _safe_mean(daily_ic["rank_ic"])
        if mode == "fast":
            if abs(raw_rank_ic) < self.min_abs_rank_ic:
                return self._store(key, self._reject(node, raw_values, finite_ratio, "low_fast_rank_ic"))
            metrics = {
                "finite_ratio": finite_ratio,
                "value_std": value_std,
                "rank_ic_mean": raw_rank_ic,
                "validation_rank_ic_mean": 0.0,
                "temporal_ic_stability": 0.0,
                "stability": 0.0,
                "signal_turnover": 0.0,
                "turnover": 0.0,
                "complexity": node.complexity(),
                "ic_observation_count": int(daily_ic["rank_ic"].notna().sum()),
            }
            return self._store(
                key,
                EvaluationResult(
                    node=node,
                    values=raw_values,
                    finite_ratio=finite_ratio,
                    fitness=(
                        self.fitness_config.fast_rank_ic_weight * abs(raw_rank_ic)
                        - self.fitness_config.complexity_penalty * node.complexity()
                    ),
                    direction=1,
                    metrics=metrics,
                ),
            )

        split_metrics = _summarize_ic_splits(daily_ic, context.split_map)
        raw_validation_rank_ic = split_metrics["validation"]["rank_ic_mean"]
        if abs(raw_validation_rank_ic) < self.min_abs_rank_ic:
            return self._store(key, self._reject(node, raw_values, finite_ratio, "low_validation_rank_ic"))

        direction = -1 if raw_validation_rank_ic < 0.0 else 1
        values = raw_values * float(direction)
        oriented_ic = daily_ic.copy()
        oriented_ic["rank_ic"] = oriented_ic["rank_ic"] * direction
        split_metrics = _summarize_ic_splits(oriented_ic, context.split_map)
        validation_ic = _validation_ic_series(oriented_ic, context.split_map)
        stability = _temporal_ic_stability(validation_ic)
        signal_turnover = _signal_turnover(prepared, values, context.split_map)
        if signal_turnover > self.max_signal_turnover:
            return self._store(key, self._reject(node, values, finite_ratio, "high_signal_turnover"))

        metrics = {
            "finite_ratio": finite_ratio,
            "value_std": value_std,
            "rank_ic_mean": _safe_mean(oriented_ic["rank_ic"]),
            "validation_rank_ic_mean": split_metrics["validation"]["rank_ic_mean"],
            "temporal_ic_stability": stability,
            "stability": stability,
            "signal_turnover": signal_turnover,
            "turnover": signal_turnover,
            "complexity": node.complexity(),
            "direction": direction,
            "ic_observation_count": int(validation_ic.notna().sum()),
        }
        return self._store(
            key,
            EvaluationResult(
                node=node,
                values=values,
                finite_ratio=finite_ratio,
                fitness=compute_research_fitness(metrics, node.complexity(), self.fitness_config),
                direction=direction,
                metrics=metrics,
                split_metrics=split_metrics,
            ),
        )

    def _context_for_panel(self, panel: pd.DataFrame) -> WindowResearchContext:
        cached = self._contexts_by_panel_id.get(id(panel))
        if cached is not None:
            return cached
        fingerprint = _panel_fingerprint(panel)
        context = WindowResearchContext(
            name=f"panel:{fingerprint}",
            panel=panel,
            split_map={
                pd.Timestamp(date): "validation"
                for date in pd.to_datetime(panel["date"], utc=False).dropna().unique()
            },
            fingerprint=fingerprint,
        )
        self._contexts_by_panel_id[id(panel)] = context
        return context

    def _specification_key(self, mode: str) -> str:
        config = self.fitness_config
        return "|".join(str(value) for value in (
            mode,
            self.min_finite_ratio,
            self.min_std,
            self.min_abs_rank_ic,
            self.max_signal_turnover,
            config.fast_rank_ic_weight,
            config.validation_ic_weight,
            config.stability_weight,
            config.turnover_penalty,
            config.complexity_penalty,
            tuple(sorted(self.trading_convention.to_dict().items())),
        ))

    def _store(self, key: tuple[str, str, str], result: EvaluationResult) -> EvaluationResult:
        self.cache[key] = result
        return result

    def _reject(self, node: FactorNode, values: pd.Series, finite_ratio: float, reason: str) -> EvaluationResult:
        return EvaluationResult(
            node=node,
            values=values,
            finite_ratio=finite_ratio,
            fitness=VERY_BAD_FITNESS,
            direction=1,
            metrics={
                "finite_ratio": finite_ratio,
                "rank_ic_mean": 0.0,
                "validation_rank_ic_mean": 0.0,
                "temporal_ic_stability": 0.0,
                "stability": 0.0,
                "signal_turnover": 0.0,
                "turnover": 0.0,
                "complexity": node.complexity(),
                "reject_reason": reason,
            },
        )


def compute_research_fitness(metrics: dict[str, Any], complexity: int, config: FitnessConfig) -> float:
    """Research fitness intentionally excludes all strategy-PnL quantities."""
    return float(
        config.validation_ic_weight * float(metrics.get("validation_rank_ic_mean", 0.0))
        + config.stability_weight * float(metrics.get("temporal_ic_stability", metrics.get("stability", 0.0)))
        - config.turnover_penalty * float(metrics.get("signal_turnover", metrics.get("turnover", 0.0)))
        - config.complexity_penalty * float(complexity)
    )


def _prepare_research_panel(panel: pd.DataFrame, trading_convention: TradingConvention) -> pd.DataFrame:
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing required columns: {missing}")
    prepared = panel.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], utc=False)
    prepared = prepared.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["future_return"] = trading_convention.forward_return(prepared)
    return prepared


def _daily_cross_sectional_rank_ic(dates: pd.Series, values: pd.Series, returns: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates, utc=False),
        "value": pd.to_numeric(values, errors="coerce"),
        "future_return": pd.to_numeric(returns, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["date", "rank_ic", "observations"])
    frame["value_rank"] = frame.groupby("date", sort=False)["value"].rank(method="average")
    frame["return_rank"] = frame.groupby("date", sort=False)["future_return"].rank(method="average")
    frame["value_centered"] = frame["value_rank"] - frame.groupby("date", sort=False)["value_rank"].transform("mean")
    frame["return_centered"] = frame["return_rank"] - frame.groupby("date", sort=False)["return_rank"].transform("mean")
    frame["cross_product"] = frame["value_centered"] * frame["return_centered"]
    frame["value_square"] = frame["value_centered"] ** 2
    frame["return_square"] = frame["return_centered"] ** 2
    grouped = frame.groupby("date", sort=False)
    numerator = grouped["cross_product"].sum()
    denominator = np.sqrt(grouped["value_square"].sum() * grouped["return_square"].sum())
    observations = grouped.size()
    rank_ic = (numerator / denominator.where(denominator > 0.0)).where(observations >= 3)
    return pd.DataFrame({"date": rank_ic.index, "rank_ic": rank_ic.to_numpy(dtype=float), "observations": observations.to_numpy(dtype=int)})


def _summarize_ic_splits(daily_ic: pd.DataFrame, split_map: dict[pd.Timestamp, str]) -> dict[str, dict[str, float]]:
    frame = daily_ic.copy()
    frame["split"] = pd.to_datetime(frame["date"], utc=False).map(split_map).fillna("validation")
    return {
        split: {
            "rank_ic_mean": _safe_mean(frame.loc[frame["split"] == split, "rank_ic"]),
            "ic_observation_count": int(frame.loc[frame["split"] == split, "rank_ic"].notna().sum()),
        }
        for split in ("train", "validation")
    }


def _validation_ic_series(daily_ic: pd.DataFrame, split_map: dict[pd.Timestamp, str]) -> pd.Series:
    labels = pd.to_datetime(daily_ic["date"], utc=False).map(split_map).fillna("validation")
    return daily_ic.loc[labels == "validation", "rank_ic"]


def _temporal_ic_stability(rank_ic: pd.Series) -> float:
    clean = rank_ic.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(np.sign(clean).mean())


def _signal_turnover(panel: pd.DataFrame, values: pd.Series, split_map: dict[pd.Timestamp, str]) -> float:
    frame = pd.DataFrame({
        "date": pd.to_datetime(panel["date"], utc=False),
        "symbol": panel["symbol"].astype(str),
        "signal": pd.to_numeric(values, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan)
    labels = frame["date"].map(split_map).fillna("validation")
    frame = frame.loc[labels == "validation"].dropna(subset=["signal"])
    if frame.empty:
        return 0.0
    percentile = frame.groupby("date", sort=False)["signal"].rank(method="average", pct=True)
    frame["normalized_signal"] = (2.0 * percentile) - 1.0
    changes = (
        frame.sort_values(["symbol", "date"], kind="mergesort")
        .groupby("symbol", sort=False)["normalized_signal"]
        .diff()
        .abs()
        .dropna()
    )
    return float(changes.mean()) if not changes.empty else 0.0


def _split_dates(dates: pd.Series) -> dict[pd.Timestamp, str]:
    unique_dates = np.sort(pd.to_datetime(pd.Series(dates).dropna().unique(), utc=False))
    train_end = min(max(1, int(len(unique_dates) * 0.75)), max(len(unique_dates) - 1, 1))
    return {
        pd.Timestamp(date): "train" if index < train_end else "validation"
        for index, date in enumerate(unique_dates)
    }


def _safe_mean(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else 0.0


def _panel_fingerprint(panel: pd.DataFrame) -> str:
    digest = pd.util.hash_pandas_object(panel, index=True).values.tobytes()
    return hashlib.sha256(digest).hexdigest()


def _context_identity(context: WindowResearchContext) -> str:
    split_payload = "|".join(
        f"{pd.Timestamp(date).isoformat()}={label}"
        for date, label in sorted(context.split_map.items(), key=lambda item: pd.Timestamp(item[0]))
    )
    split_hash = hashlib.sha256(split_payload.encode("utf-8")).hexdigest()
    return f"{context.name}|{context.fingerprint}|{split_hash}"
