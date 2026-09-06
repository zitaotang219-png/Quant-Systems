"""Lightweight factor research evaluation, independent of portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import perf_counter
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
    identity: str
    dates: pd.Series = field(compare=False, repr=False)
    date_codes: np.ndarray = field(compare=False, repr=False)
    unique_dates: np.ndarray = field(compare=False, repr=False)
    symbol_codes: np.ndarray = field(compare=False, repr=False)
    train_mask: np.ndarray = field(compare=False, repr=False)
    validation_mask: np.ndarray = field(compare=False, repr=False)
    target_values: np.ndarray = field(compare=False, repr=False)
    target_valid_mask: np.ndarray = field(compare=False, repr=False)
    target_rank: np.ndarray = field(compare=False, repr=False)


@dataclass
class ResearchEvaluationInstrumentation:
    total_expression_evaluations: int = 0
    unique_expression_evaluations: int = 0
    evaluation_cache_hits: int = 0
    value_cache_hits: int = 0
    rank_ic_cache_hits: int = 0
    rank_ic_evaluations: int = 0
    fast_screen_evaluations: int = 0
    factor_research_evaluations: int = 0
    avoided_duplicate_evaluations: int = 0
    symbolic_expression_seconds: float = 0.0
    rank_ic_seconds: float = 0.0
    research_evaluation_seconds: float = 0.0
    portfolio_simulation_calls: int = 0

    def snapshot(self) -> dict[str, int | float]:
        return {
            "total_expression_evaluations": self.total_expression_evaluations,
            "unique_expression_evaluations": self.unique_expression_evaluations,
            "cache_hits": self.evaluation_cache_hits + self.value_cache_hits + self.rank_ic_cache_hits,
            "evaluation_cache_hits": self.evaluation_cache_hits,
            "value_cache_hits": self.value_cache_hits,
            "rank_ic_cache_hits": self.rank_ic_cache_hits,
            "rank_ic_evaluations": self.rank_ic_evaluations,
            "fast_screen_evaluations": self.fast_screen_evaluations,
            "factor_research_evaluations": self.factor_research_evaluations,
            "avoided_duplicate_evaluations": self.avoided_duplicate_evaluations,
            "symbolic_expression_seconds": self.symbolic_expression_seconds,
            "rank_ic_seconds": self.rank_ic_seconds,
            "research_evaluation_seconds": self.research_evaluation_seconds,
            "portfolio_simulation_calls": self.portfolio_simulation_calls,
        }


@dataclass(frozen=True)
class _PanelResearchMetadata:
    fingerprint: str
    dates: pd.Series = field(compare=False, repr=False)
    date_codes: np.ndarray = field(compare=False, repr=False)
    unique_dates: np.ndarray = field(compare=False, repr=False)
    symbol_codes: np.ndarray = field(compare=False, repr=False)
    target_values: np.ndarray = field(compare=False, repr=False)
    target_valid_mask: np.ndarray = field(compare=False, repr=False)
    target_rank: np.ndarray = field(compare=False, repr=False)


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
    instrumentation: ResearchEvaluationInstrumentation = field(default_factory=ResearchEvaluationInstrumentation)
    _prepared_by_source_id: dict[int, pd.DataFrame] = field(default_factory=dict, repr=False)
    _metadata_by_panel_id: dict[int, _PanelResearchMetadata] = field(default_factory=dict, repr=False)
    _metadata_by_fingerprint: dict[str, _PanelResearchMetadata] = field(default_factory=dict, repr=False)
    _contexts_by_identity: dict[str, WindowResearchContext] = field(default_factory=dict, repr=False)
    _contexts_by_panel_id: dict[int, WindowResearchContext] = field(default_factory=dict, repr=False)
    _value_cache_by_fingerprint: dict[str, dict[str, pd.Series]] = field(default_factory=dict, repr=False)
    _daily_ic_cache: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict, repr=False)

    def prepare_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        cached = self._prepared_by_source_id.get(id(panel))
        if cached is not None:
            return cached
        prepared = panel if "future_return" in panel.columns else _prepare_research_panel(panel, self.trading_convention)
        self._prepared_by_source_id[id(panel)] = prepared
        self._prepared_by_source_id[id(prepared)] = prepared
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
        normalized_split_map = {pd.Timestamp(date): str(label) for date, label in resolved_split_map.items()}
        metadata = self._metadata_for_panel(prepared)
        identity = _context_identity(str(name), metadata.fingerprint, normalized_split_map)
        cached_context = self._contexts_by_identity.get(identity)
        if cached_context is not None:
            return cached_context
        split_labels = metadata.dates.map(normalized_split_map).fillna("validation")
        train_mask = split_labels.eq("train").to_numpy(dtype=bool)
        validation_mask = split_labels.eq("validation").to_numpy(dtype=bool)
        context = WindowResearchContext(
            name=str(name),
            panel=prepared,
            split_map=normalized_split_map,
            fingerprint=metadata.fingerprint,
            identity=identity,
            dates=metadata.dates,
            date_codes=metadata.date_codes,
            unique_dates=metadata.unique_dates,
            symbol_codes=metadata.symbol_codes,
            train_mask=train_mask,
            validation_mask=validation_mask,
            target_values=metadata.target_values,
            target_valid_mask=metadata.target_valid_mask,
            target_rank=metadata.target_rank,
        )
        self._contexts_by_identity[identity] = context
        if context.name == f"panel:{metadata.fingerprint}":
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
        started = perf_counter()
        if mode not in {"fast", "research"}:
            raise ValueError(f"Unknown factor research mode: {mode}")
        self.instrumentation.total_expression_evaluations += 1
        expression = node.describe()
        key = (expression, context.identity, self._specification_key(mode))
        cached = self.cache.get(key)
        if cached is not None:
            self.instrumentation.evaluation_cache_hits += 1
            self.instrumentation.avoided_duplicate_evaluations += 1
            self._record_research_time(mode, started)
            return cached
        self.evaluation_calls[key] = self.evaluation_calls.get(key, 0) + 1
        if mode == "fast":
            self.instrumentation.fast_screen_evaluations += 1
        else:
            self.instrumentation.factor_research_evaluations += 1

        prepared = context.panel
        value_cache = self._value_cache_by_fingerprint.setdefault(context.fingerprint, {})
        raw_values = value_cache.get(expression)
        if raw_values is not None:
            self.instrumentation.value_cache_hits += 1
            self.instrumentation.avoided_duplicate_evaluations += 1
        else:
            symbolic_started = perf_counter()
            raw_values = node.evaluate(prepared).astype(float)
            self.instrumentation.symbolic_expression_seconds += perf_counter() - symbolic_started
            value_cache[expression] = raw_values
            self.instrumentation.unique_expression_evaluations += 1
        finite_ratio = float(np.isfinite(raw_values.to_numpy(dtype=float, na_value=np.nan)).mean())
        if finite_ratio < self.min_finite_ratio:
            return self._finish(
                key,
                self._reject(node, raw_values, finite_ratio, "finite_ratio"),
                mode,
                started,
            )

        value_std = float(np.nanstd(raw_values.to_numpy(dtype=float, na_value=np.nan)))
        if not np.isfinite(value_std) or value_std < self.min_std:
            return self._finish(key, self._reject(node, raw_values, finite_ratio, "variance"), mode, started)

        ic_key = (expression, context.fingerprint)
        daily_ic = self._daily_ic_cache.get(ic_key)
        if daily_ic is None:
            ic_started = perf_counter()
            daily_ic = _daily_cross_sectional_rank_ic_context(context, raw_values)
            self.instrumentation.rank_ic_seconds += perf_counter() - ic_started
            self.instrumentation.rank_ic_evaluations += 1
            self._daily_ic_cache[ic_key] = daily_ic
        else:
            self.instrumentation.rank_ic_cache_hits += 1
        raw_rank_ic = _safe_mean(daily_ic["rank_ic"])
        if mode == "fast":
            if abs(raw_rank_ic) < self.min_abs_rank_ic:
                return self._finish(
                    key,
                    self._reject(node, raw_values, finite_ratio, "low_fast_rank_ic"),
                    mode,
                    started,
                )
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
            return self._finish(
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
                mode,
                started,
            )

        split_metrics = _summarize_ic_splits(daily_ic, context.split_map)
        raw_validation_rank_ic = split_metrics["validation"]["rank_ic_mean"]
        if abs(raw_validation_rank_ic) < self.min_abs_rank_ic:
            return self._finish(
                key,
                self._reject(node, raw_values, finite_ratio, "low_validation_rank_ic"),
                mode,
                started,
            )

        direction = -1 if raw_validation_rank_ic < 0.0 else 1
        values = raw_values * float(direction)
        oriented_ic = daily_ic.copy()
        oriented_ic["rank_ic"] = oriented_ic["rank_ic"] * direction
        split_metrics = _summarize_ic_splits(oriented_ic, context.split_map)
        validation_ic = _validation_ic_series(oriented_ic, context.split_map)
        stability = _temporal_ic_stability(validation_ic)
        signal_turnover = _signal_turnover_context(context, values)
        if signal_turnover > self.max_signal_turnover:
            return self._finish(
                key,
                self._reject(node, values, finite_ratio, "high_signal_turnover"),
                mode,
                started,
            )

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
        return self._finish(
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
            mode,
            started,
        )

    def _context_for_panel(self, panel: pd.DataFrame) -> WindowResearchContext:
        cached = self._contexts_by_panel_id.get(id(panel))
        if cached is not None:
            return cached
        metadata = self._metadata_for_panel(panel)
        return self.create_context(panel, name=f"panel:{metadata.fingerprint}", split_label="validation")

    def _metadata_for_panel(self, panel: pd.DataFrame) -> _PanelResearchMetadata:
        cached = self._metadata_by_panel_id.get(id(panel))
        if cached is not None:
            return cached
        fingerprint = _panel_fingerprint(panel)
        cached = self._metadata_by_fingerprint.get(fingerprint)
        if cached is not None:
            self._metadata_by_panel_id[id(panel)] = cached
            return cached
        dates = pd.to_datetime(panel["date"], utc=False)
        date_codes, unique_dates = pd.factorize(dates, sort=False)
        symbol_codes, _ = pd.factorize(panel["symbol"].astype(str), sort=False)
        target_values = pd.to_numeric(panel["future_return"], errors="coerce").to_numpy(dtype=float)
        target_valid_mask = np.isfinite(target_values) & (date_codes >= 0)
        target_rank = _rank_within_codes(target_values, date_codes, target_valid_mask)
        metadata = _PanelResearchMetadata(
            fingerprint=fingerprint,
            dates=pd.Series(dates.to_numpy(), index=panel.index),
            date_codes=np.asarray(date_codes, dtype=np.int64),
            unique_dates=np.asarray(unique_dates),
            symbol_codes=np.asarray(symbol_codes, dtype=np.int64),
            target_values=target_values,
            target_valid_mask=target_valid_mask,
            target_rank=target_rank,
        )
        self._metadata_by_panel_id[id(panel)] = metadata
        self._metadata_by_fingerprint[fingerprint] = metadata
        return metadata

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

    def _finish(
        self,
        key: tuple[str, str, str],
        result: EvaluationResult,
        mode: str,
        started: float,
    ) -> EvaluationResult:
        self.cache[key] = result
        self._record_research_time(mode, started)
        return result

    def _record_research_time(self, mode: str, started: float) -> None:
        if mode == "research":
            self.instrumentation.research_evaluation_seconds += perf_counter() - started

    def instrumentation_snapshot(self) -> dict[str, int | float]:
        return self.instrumentation.snapshot()

    def cached_evaluation(
        self,
        node: FactorNode,
        context: WindowResearchContext,
        *,
        mode: str,
    ) -> EvaluationResult | None:
        """Return an existing result without recording a new evaluation request."""
        key = (node.describe(), context.identity, self._specification_key(mode))
        cached = self.cache.get(key)
        if cached is not None:
            self.instrumentation.evaluation_cache_hits += 1
            self.instrumentation.avoided_duplicate_evaluations += 1
        return cached

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


def _daily_cross_sectional_rank_ic_context(
    context: WindowResearchContext,
    values: pd.Series,
) -> pd.DataFrame:
    """Rank IC using precomputed target/group metadata from one research window."""
    candidate = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    pair_valid = np.isfinite(candidate) & context.target_valid_mask
    if np.array_equal(pair_valid, context.target_valid_mask):
        target_rank = context.target_rank
    else:
        target_rank = _rank_within_codes(context.target_values, context.date_codes, pair_valid)
    candidate_rank = _rank_within_codes(candidate, context.date_codes, pair_valid)
    rank_valid = pair_valid & np.isfinite(candidate_rank) & np.isfinite(target_rank)
    group_count = len(context.unique_dates)
    if not rank_valid.any() or group_count == 0:
        return pd.DataFrame(columns=["date", "rank_ic", "observations"])

    codes = context.date_codes[rank_valid]
    left = candidate_rank[rank_valid]
    right = target_rank[rank_valid]
    observations = np.bincount(codes, minlength=group_count).astype(int)
    left_mean = np.divide(
        np.bincount(codes, weights=left, minlength=group_count),
        observations,
        out=np.zeros(group_count, dtype=float),
        where=observations > 0,
    )
    right_mean = np.divide(
        np.bincount(codes, weights=right, minlength=group_count),
        observations,
        out=np.zeros(group_count, dtype=float),
        where=observations > 0,
    )
    left_centered = left - left_mean[codes]
    right_centered = right - right_mean[codes]
    numerator = np.bincount(codes, weights=left_centered * right_centered, minlength=group_count)
    left_square = np.bincount(codes, weights=left_centered**2, minlength=group_count)
    right_square = np.bincount(codes, weights=right_centered**2, minlength=group_count)
    denominator = np.sqrt(left_square * right_square)
    rank_ic = np.divide(
        numerator,
        denominator,
        out=np.full(group_count, np.nan, dtype=float),
        where=(denominator > 0.0) & (observations >= 3),
    )
    present = observations > 0
    return pd.DataFrame({
        "date": pd.to_datetime(context.unique_dates[present], utc=False),
        "rank_ic": rank_ic[present],
        "observations": observations[present],
    })


def _rank_within_codes(values: np.ndarray, codes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ranked = (
        pd.Series(np.where(valid, values, np.nan))
        .groupby(pd.Series(codes), sort=False)
        .rank(method="average")
    )
    return ranked.to_numpy(dtype=float)


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


def _signal_turnover_context(context: WindowResearchContext, values: pd.Series) -> float:
    """Signal turnover without rebuilding or sorting a per-factor panel DataFrame."""
    candidate = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    selected = context.validation_mask & np.isfinite(candidate)
    if not selected.any():
        return 0.0
    positions = np.flatnonzero(selected)
    date_codes = context.date_codes[positions]
    symbol_codes = context.symbol_codes[positions]
    signals = candidate[positions]
    percentile = pd.Series(signals).groupby(pd.Series(date_codes), sort=False).rank(method="average", pct=True)
    normalized = (2.0 * percentile) - 1.0
    ordered = pd.DataFrame({
        "symbol_code": symbol_codes,
        "date": context.dates.iloc[positions].to_numpy(),
        "normalized_signal": normalized.to_numpy(dtype=float),
    }).sort_values(["symbol_code", "date"], kind="mergesort")
    changes = ordered.groupby("symbol_code", sort=False)["normalized_signal"].diff().abs().dropna()
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


def _context_identity(name: str, fingerprint: str, split_map: dict[pd.Timestamp, str]) -> str:
    split_payload = "|".join(
        f"{pd.Timestamp(date).isoformat()}={label}"
        for date, label in sorted(split_map.items(), key=lambda item: pd.Timestamp(item[0]))
    )
    split_hash = hashlib.sha256(split_payload.encode("utf-8")).hexdigest()
    return f"{name}|{fingerprint}|{split_hash}"
