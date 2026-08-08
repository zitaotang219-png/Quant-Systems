from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.trading_convention import DEFAULT_TRADING_CONVENTION, TradingConvention

from .config import FitnessConfig
from .dsl import FactorNode
from .portfolio_construction import build_weight_frame
from .regime import build_regime_frame

VERY_BAD_FITNESS = -1_000_000_000.0
MIN_FINITE_RATIO = 0.6
MIN_STD = 1e-6


def compute_factor_fitness(
    metrics: dict[str, Any],
    complexity: int,
    fitness_config: FitnessConfig,
) -> float:
    return float(
        fitness_config.validation_ic_weight * float(metrics.get("validation_rank_ic_mean", 0.0))
        + fitness_config.sharpe_weight * float(metrics.get("sharpe", 0.0))
        + fitness_config.cumulative_return_weight * float(metrics.get("cumulative_return", 0.0))
        + fitness_config.excess_return_weight * float(metrics.get("excess_return_vs_equal_weight", 0.0))
        + fitness_config.stability_weight * float(metrics.get("stability", 0.0))
        + fitness_config.bear_return_weight * float(metrics.get("bear_cumulative_return", 0.0))
        + fitness_config.bear_sharpe_weight * float(metrics.get("bear_sharpe", 0.0))
        - fitness_config.turnover_penalty * float(metrics.get("turnover", 0.0))
        - fitness_config.drawdown_penalty * float(metrics.get("max_drawdown", 0.0))
        - fitness_config.complexity_penalty * float(complexity)
    )


@dataclass
class EvaluationResult:
    node: FactorNode
    values: pd.Series
    finite_ratio: float
    fitness: float
    direction: int
    metrics: dict[str, Any] = field(default_factory=dict)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cumulative_return: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    split_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class FactorEvaluator:
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    trading_convention: TradingConvention = DEFAULT_TRADING_CONVENTION
    min_finite_ratio: float = MIN_FINITE_RATIO
    min_std: float = MIN_STD
    min_abs_rank_ic: float = 0.005
    max_turnover: float = 5.0
    max_allowed_drawdown: float = 0.5
    ic_sign_tolerance: float = 0.002
    annualization: int = 252
    weighting_scheme: str = "continuous"
    position_limit: float = 1.0
    turnover_limit: float = 5.0
    gross_leverage: float = 1.0
    signal_vol_window: int = 20
    signal_clip: float = 3.0
    smoothing: float = 0.5
    market_neutral: bool = True
    benchmark_follow_enabled: bool = False
    benchmark_follow_btc_symbol: str = "BTCUSDT"
    benchmark_follow_btc_weight: float = 0.5
    regime_benchmark_blend: dict[str, float] = field(default_factory=dict)
    regime_benchmark_direction: dict[str, float] = field(default_factory=dict)
    regime_config: Any = None
    reject_non_profitable: bool = True
    reject_unstable_return_path: bool = True
    fitness_config: FitnessConfig = field(default_factory=FitnessConfig)
    cache: dict[str, EvaluationResult] = field(default_factory=dict)

    def fast_filter(self, node: FactorNode, panel: pd.DataFrame) -> EvaluationResult:
        return self._evaluate(node=node, panel=panel, deep=False)

    def evaluate(self, node: FactorNode, panel: pd.DataFrame) -> EvaluationResult:
        return self._evaluate(node=node, panel=panel, deep=True)

    def _evaluate(self, node: FactorNode, panel: pd.DataFrame, deep: bool) -> EvaluationResult:
        prepared = _prepare_panel(
            panel,
            trading_convention=self.trading_convention,
        )
        key = f"{_panel_fingerprint(prepared)}|{node.describe()}|deep={int(deep)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        raw_values = node.evaluate(prepared).astype(float)
        finite_ratio = float(np.isfinite(raw_values.to_numpy(dtype=float, na_value=np.nan)).mean())
        if finite_ratio < self.min_finite_ratio:
            result = self._reject(node, raw_values, finite_ratio, "finite_ratio")
            self.cache[key] = result
            return result

        value_std = float(np.nanstd(raw_values.to_numpy(dtype=float, na_value=np.nan)))
        if not np.isfinite(value_std) or value_std < self.min_std:
            result = self._reject(node, raw_values, finite_ratio, "variance")
            self.cache[key] = result
            return result

        if not deep:
            ic_frame = _compute_cross_sectional_ic(prepared["date"], raw_values, prepared["future_return"])
            rank_ic_mean = _safe_mean(ic_frame["rank_ic"])
            if abs(rank_ic_mean) < self.min_abs_rank_ic:
                result = self._reject(node, raw_values, finite_ratio, "low_fast_rank_ic")
                self.cache[key] = result
                return result
            metrics = {
                "ic_mean": _safe_mean(ic_frame["pearson_ic"]),
                "rank_ic_mean": rank_ic_mean,
                "validation_rank_ic_mean": 0.0,
                "sharpe": 0.0,
                "turnover": 0.0,
                "max_drawdown": 0.0,
                "value_std": value_std,
            }
            result = EvaluationResult(
                node=node,
                values=raw_values,
                finite_ratio=finite_ratio,
                fitness=(
                    (self.fitness_config.fast_rank_ic_weight * abs(metrics["rank_ic_mean"]))
                    - (self.fitness_config.complexity_penalty * node.complexity())
                ),
                direction=1,
                metrics=metrics,
            )
            self.cache[key] = result
            return result

        result = self.evaluate_prepared_with_splits(
            node=node,
            prepared=prepared,
            raw_values=raw_values,
            finite_ratio=finite_ratio,
            split_map=_split_dates(prepared["date"]),
        )
        self.cache[key] = result
        return result

    def evaluate_with_splits(
        self,
        node: FactorNode,
        panel: pd.DataFrame,
        split_map: dict[pd.Timestamp, str],
    ) -> EvaluationResult:
        prepared = _prepare_panel(
            panel,
            trading_convention=self.trading_convention,
        )
        raw_values = node.evaluate(prepared).astype(float)
        finite_ratio = float(np.isfinite(raw_values.to_numpy(dtype=float, na_value=np.nan)).mean())
        if finite_ratio < self.min_finite_ratio:
            return self._reject(node, raw_values, finite_ratio, "finite_ratio")

        value_std = float(np.nanstd(raw_values.to_numpy(dtype=float, na_value=np.nan)))
        if not np.isfinite(value_std) or value_std < self.min_std:
            return self._reject(node, raw_values, finite_ratio, "variance")

        return self.evaluate_prepared_with_splits(
            node=node,
            prepared=prepared,
            raw_values=raw_values,
            finite_ratio=finite_ratio,
            split_map=split_map,
            value_std=value_std,
        )

    def evaluate_prepared_with_splits(
        self,
        node: FactorNode,
        prepared: pd.DataFrame,
        raw_values: pd.Series,
        finite_ratio: float,
        split_map: dict[pd.Timestamp, str],
        value_std: float | None = None,
    ) -> EvaluationResult:
        effective_value_std = value_std
        if effective_value_std is None:
            effective_value_std = float(np.nanstd(raw_values.to_numpy(dtype=float, na_value=np.nan)))

        ic_frame = _compute_cross_sectional_ic(prepared["date"], raw_values, prepared["future_return"])
        split_metrics = _summarize_splits(ic_frame, split_map)

        raw_validation_rank_ic = split_metrics["validation"]["rank_ic_mean"]
        if abs(raw_validation_rank_ic) < self.min_abs_rank_ic:
            return self._reject(node, raw_values, finite_ratio, "low_validation_rank_ic")

        direction = -1 if raw_validation_rank_ic < 0.0 else 1
        values = raw_values * float(direction)
        oriented_ic_frame = ic_frame.copy()
        oriented_ic_frame["pearson_ic"] = oriented_ic_frame["pearson_ic"] * direction
        oriented_ic_frame["rank_ic"] = oriented_ic_frame["rank_ic"] * direction
        split_metrics = _summarize_splits(oriented_ic_frame, split_map)

        portfolio = self._simulate_long_short_portfolio(prepared, values)
        portfolio_split_metrics = _summarize_portfolio_splits(portfolio, split_map, self.annualization)
        validation_split = portfolio_split_metrics.get("validation", {})
        sharpe = float(validation_split.get("sharpe", 0.0))
        max_drawdown = float(validation_split.get("max_drawdown", 0.0))
        normalized_turnover = float(validation_split.get("turnover", 0.0))

        if normalized_turnover > self.max_turnover:
            return self._reject(node, values, finite_ratio, "high_turnover")

        if max_drawdown > self.max_allowed_drawdown:
            return self._reject(node, values, finite_ratio, "high_drawdown")

        metrics = {
            "ic_mean": _safe_mean(oriented_ic_frame["pearson_ic"]),
            "rank_ic_mean": _safe_mean(oriented_ic_frame["rank_ic"]),
            "validation_rank_ic_mean": split_metrics["validation"]["rank_ic_mean"],
            "sharpe": sharpe,
            "turnover": normalized_turnover,
            "max_drawdown": max_drawdown,
            "cumulative_return": float(validation_split.get("cumulative_return", 0.0)),
            "excess_return_vs_equal_weight": float(validation_split.get("excess_return_vs_equal_weight", 0.0)),
            "stability": float(validation_split.get("stability", 0.0)),
            "bear_cumulative_return": float(validation_split.get("bear_cumulative_return", 0.0)),
            "bear_sharpe": float(validation_split.get("bear_sharpe", 0.0)),
            "value_std": effective_value_std,
            "direction": direction,
        }

        if self.reject_non_profitable and metrics["cumulative_return"] < -0.02 and metrics["sharpe"] <= 0.0:
            return self._reject(node, values, finite_ratio, "non_profitable_validation")

        if self.reject_unstable_return_path and metrics["stability"] < -0.05:
            return self._reject(node, values, finite_ratio, "unstable_return_path")

        return EvaluationResult(
            node=node,
            values=values,
            finite_ratio=finite_ratio,
            fitness=compute_factor_fitness(metrics, node.complexity(), self.fitness_config),
            direction=direction,
            metrics=metrics,
            daily_returns=portfolio["daily_return"],
            cumulative_return=portfolio["cumulative_return"],
            split_metrics=split_metrics,
        )

    def _reject(self, node: FactorNode, values: pd.Series, finite_ratio: float, reason: str) -> EvaluationResult:
        metrics = {
            "ic_mean": 0.0,
            "rank_ic_mean": 0.0,
            "validation_rank_ic_mean": 0.0,
            "sharpe": 0.0,
            "turnover": 0.0,
            "max_drawdown": 1.0,
            "cumulative_return": 0.0,
            "excess_return_vs_equal_weight": 0.0,
            "stability": 0.0,
            "bear_cumulative_return": 0.0,
            "bear_sharpe": 0.0,
            "reject_reason": reason,
        }
        return EvaluationResult(
            node=node,
            values=values,
            finite_ratio=finite_ratio,
            fitness=VERY_BAD_FITNESS,
            direction=1,
            metrics=metrics,
        )

    def _simulate_long_short_portfolio(self, panel: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "date": panel["date"],
                "symbol": panel["symbol"],
                "factor": values.astype(float),
                "future_return": panel["future_return"].astype(float),
            },
            index=panel.index,
        )

        regime_by_date = None
        if self.regime_config is not None and getattr(self.regime_config, "enabled", False):
            regime_frame = build_regime_frame(panel, self.regime_config)
            if not regime_frame.empty:
                regime_by_date = regime_frame.set_index("date")["regime"]

        weights = build_weight_frame(
            panel=panel,
            score_series=values,
            weighting_scheme=self.weighting_scheme,
            long_quantile=self.long_quantile,
            short_quantile=self.short_quantile,
            position_limit=self.position_limit,
            gross_leverage=self.gross_leverage,
            signal_vol_window=self.signal_vol_window,
            signal_clip=self.signal_clip,
            smoothing=self.smoothing,
            market_neutral=self.market_neutral,
            turnover_limit=self.turnover_limit,
            regime_by_date=regime_by_date,
            benchmark_follow_enabled=self.benchmark_follow_enabled,
            benchmark_follow_btc_symbol=self.benchmark_follow_btc_symbol,
            benchmark_follow_btc_weight=self.benchmark_follow_btc_weight,
            regime_benchmark_blend=self.regime_benchmark_blend,
            regime_benchmark_direction=self.regime_benchmark_direction,
        )
        weights["gross_weight"] = weights["weight"].abs()
        weights = weights.merge(
            frame[["date", "symbol", "future_return"]],
            on=["date", "symbol"],
            how="left",
        )
        weights["pnl_component"] = weights["weight"] * weights["future_return"]

        daily = (
            weights.groupby("date", sort=False)
            .agg(
                gross_exposure=("gross_weight", "sum"),
                gross_return=("pnl_component", "sum"),
                equal_weight_return=("future_return", "mean"),
            )
            .reset_index()
        )

        turnover = (
            weights.sort_values(["symbol", "date"], kind="mergesort")
            .groupby("symbol", sort=False)["weight"]
            .diff()
            .abs()
        )
        weights["weight_change"] = turnover.fillna(weights["weight"].abs())
        turnover_by_day = weights.groupby("date", sort=False)["weight_change"].sum().reindex(daily["date"]).fillna(0.0)

        total_cost_bps = self.transaction_cost_bps + self.slippage_bps
        daily["turnover"] = turnover_by_day.to_numpy(dtype=float)
        daily["transaction_cost"] = daily["turnover"] * (total_cost_bps / 10000.0)
        daily["daily_return"] = (daily["gross_return"] - daily["transaction_cost"]).fillna(0.0)
        daily["excess_daily_return"] = (daily["daily_return"] - daily["equal_weight_return"]).fillna(0.0)
        daily["cumulative_return"] = (1.0 + daily["daily_return"]).cumprod() - 1.0
        daily["equal_weight_cumulative_return"] = (1.0 + daily["equal_weight_return"]).cumprod() - 1.0
        daily["excess_cumulative_return"] = (1.0 + daily["excess_daily_return"]).cumprod() - 1.0
        if regime_by_date is not None and not regime_by_date.empty:
            daily["regime"] = pd.to_datetime(daily["date"], utc=False).map(regime_by_date).fillna("bull_low_vol")
        return daily


def _prepare_panel(
    panel: pd.DataFrame,
    trading_convention: TradingConvention = DEFAULT_TRADING_CONVENTION,
) -> pd.DataFrame:
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


def _compute_cross_sectional_ic(dates: pd.Series, values: pd.Series, future_returns: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, utc=False),
            "value": values.astype(float),
            "future_return": future_returns.astype(float),
        }
    )

    def _ic_for_date(day: pd.DataFrame) -> pd.Series:
        clean = day[["value", "future_return"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            return pd.Series({"pearson_ic": np.nan, "rank_ic": np.nan})
        pearson_ic = clean["value"].corr(clean["future_return"], method="pearson")
        rank_ic = clean["value"].corr(clean["future_return"], method="spearman")
        return pd.Series({"pearson_ic": pearson_ic, "rank_ic": rank_ic})

    return frame.groupby("date", sort=False).apply(_ic_for_date).reset_index()


def _split_dates(dates: pd.Series) -> dict[pd.Timestamp, str]:
    unique_dates = np.sort(pd.to_datetime(pd.Series(dates).dropna().unique(), utc=False))
    total = len(unique_dates)
    if total == 0:
        return {}

    train_end = max(1, int(total * 0.75))
    train_end = min(train_end, max(total - 1, 1))

    label_map: dict[pd.Timestamp, str] = {}
    for index, date in enumerate(unique_dates):
        if index < train_end:
            label_map[pd.Timestamp(date)] = "train"
        else:
            label_map[pd.Timestamp(date)] = "validation"
    return label_map


def _summarize_splits(ic_frame: pd.DataFrame, split_map: dict[pd.Timestamp, str]) -> dict[str, dict[str, float]]:
    frame = ic_frame.copy()
    frame["split"] = pd.to_datetime(frame["date"], utc=False).map(split_map).fillna("validation")
    summary: dict[str, dict[str, float]] = {}
    for split_name in ("train", "validation"):
        subset = frame.loc[frame["split"] == split_name]
        summary[split_name] = {
            "ic_mean": _safe_mean(subset["pearson_ic"]) if not subset.empty else 0.0,
            "rank_ic_mean": _safe_mean(subset["rank_ic"]) if not subset.empty else 0.0,
        }
    return summary


def _summarize_portfolio_splits(
    portfolio: pd.DataFrame,
    split_map: dict[pd.Timestamp, str],
    annualization: int,
) -> dict[str, dict[str, float]]:
    if portfolio.empty:
        empty = {
            "sharpe": 0.0,
            "turnover": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
            "excess_return_vs_equal_weight": 0.0,
            "stability": 0.0,
            "bear_cumulative_return": 0.0,
            "bear_sharpe": 0.0,
        }
        return {"validation": dict(empty)}

    frame = portfolio.copy()
    frame["split"] = pd.to_datetime(frame["date"], utc=False).map(split_map)
    summaries: dict[str, dict[str, float]] = {}
    for split_name in ("validation",):
        subset = frame.loc[frame["split"] == split_name].copy()
        if subset.empty:
            summaries[split_name] = {
                "sharpe": 0.0,
                "turnover": 0.0,
                "max_drawdown": 0.0,
                "cumulative_return": 0.0,
                "excess_return_vs_equal_weight": 0.0,
                "stability": 0.0,
                "bear_cumulative_return": 0.0,
                "bear_sharpe": 0.0,
            }
            continue
        summaries[split_name] = _portfolio_summary(subset, annualization)
    return summaries


def _portfolio_summary(portfolio: pd.DataFrame, annualization: int) -> dict[str, float]:
    if portfolio.empty:
        return {
            "sharpe": 0.0,
            "turnover": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
            "excess_return_vs_equal_weight": 0.0,
            "stability": 0.0,
            "bear_cumulative_return": 0.0,
            "bear_sharpe": 0.0,
        }
    daily_return = portfolio["daily_return"].astype(float)
    turnover = float(portfolio["turnover"].mean()) if "turnover" in portfolio.columns else 0.0
    cumulative_return = (1.0 + daily_return.fillna(0.0)).cumprod() - 1.0
    excess_daily_return = portfolio["excess_daily_return"].astype(float) if "excess_daily_return" in portfolio.columns else pd.Series(0.0, index=portfolio.index)
    excess_cumulative_return = (1.0 + excess_daily_return.fillna(0.0)).cumprod() - 1.0
    bear_mask = portfolio.get("regime", pd.Series("", index=portfolio.index)).astype(str).str.startswith("bear")
    bear_returns = daily_return.loc[bear_mask]
    bear_cumulative_return = float((1.0 + bear_returns.fillna(0.0)).prod() - 1.0) if not bear_returns.empty else 0.0
    return {
        "sharpe": _annualized_sharpe(daily_return, annualization),
        "turnover": 0.0 if pd.isna(turnover) else turnover,
        "max_drawdown": _max_drawdown(cumulative_return),
        "cumulative_return": float(cumulative_return.iloc[-1]) if not cumulative_return.empty else 0.0,
        "excess_return_vs_equal_weight": float(excess_cumulative_return.iloc[-1]) if not excess_cumulative_return.empty else 0.0,
        "stability": _path_stability(daily_return),
        "bear_cumulative_return": bear_cumulative_return,
        "bear_sharpe": _annualized_sharpe(bear_returns, annualization) if not bear_returns.empty else 0.0,
    }


def _annualized_sharpe(daily_returns: pd.Series, annualization: int) -> float:
    clean = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    volatility = clean.std()
    if volatility == 0 or pd.isna(volatility):
        return 0.0
    return float((clean.mean() / volatility) * math.sqrt(annualization))


def _max_drawdown(cumulative_return: pd.Series) -> float:
    clean = cumulative_return.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    equity_curve = 1.0 + clean
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return float(abs(drawdown.min())) if not drawdown.empty else 0.0


def _path_stability(daily_returns: pd.Series) -> float:
    clean = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    total_move = float(clean.abs().sum())
    if total_move <= 1e-12:
        return 0.0
    total_return = float((1.0 + clean).prod() - 1.0)
    return float(np.clip(total_return / total_move, -1.0, 1.0))


def _safe_mean(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(clean.mean())


def _panel_fingerprint(panel: pd.DataFrame) -> str:
    if panel.empty:
        return "empty|rows=0|symbols=0"
    date_values = pd.to_datetime(panel["date"], utc=False)
    date_min = pd.Timestamp(date_values.min()).isoformat()
    date_max = pd.Timestamp(date_values.max()).isoformat()
    row_count = len(panel)
    symbol_count = int(panel["symbol"].nunique())
    return f"{date_min}|{date_max}|rows={row_count}|symbols={symbol_count}"
