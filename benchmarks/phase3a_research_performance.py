"""Focused non-holdout benchmark for the Phase 3A factor-research hot path."""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from alpha_mining.dsl import _by_symbol, parse_expression
from alpha_mining.research_evaluator import (
    FactorResearchEvaluator,
    _daily_cross_sectional_rank_ic,
    _daily_cross_sectional_rank_ic_context,
)


DEFAULT_RESEARCH_END = "2024-10-31"
NESTED_EXPRESSION = "ts_rank(rolling_mean(rank(correlation(close, volume, 20)), 10), 20)"


def _measure(function: Callable[[], object], repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        timings.append(perf_counter() - started)
    return float(median(timings))


def _legacy_ts_rank(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    def rolling_rank(series: pd.Series) -> pd.Series:
        def rank_last(values: pd.Series) -> float:
            valid = values.dropna()
            return np.nan if valid.empty else float(valid.rank(pct=True).iloc[-1])

        return series.rolling(window=window, min_periods=window).apply(rank_last, raw=False)

    return _by_symbol(panel, panel["volume"].astype(float), rolling_rank)


def run(panel_path: str, start: str, end: str, repeats: int) -> dict[str, object]:
    if pd.Timestamp(end) > pd.Timestamp(DEFAULT_RESEARCH_END):
        raise ValueError(f"Benchmark end must be at or before the non-holdout boundary {DEFAULT_RESEARCH_END}.")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], utc=False)
    panel = panel.loc[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0, max_signal_turnover=99.0)
    prepared = evaluator.prepare_panel(panel)
    context = evaluator.create_context(prepared, name="performance_benchmark:validation")

    ts_rank_node = parse_expression("ts_rank(volume, 20)")
    legacy_values = _legacy_ts_rank(prepared)
    optimized_values = ts_rank_node.evaluate(prepared)
    if not np.allclose(legacy_values, optimized_values, equal_nan=True, rtol=0.0, atol=0.0):
        raise AssertionError("Native ts_rank does not match the reference implementation.")

    nested_node = parse_expression(NESTED_EXPRESSION)
    nested_values = nested_node.evaluate(prepared)
    legacy_ic = _daily_cross_sectional_rank_ic(prepared["date"], nested_values, prepared["future_return"])
    optimized_ic = _daily_cross_sectional_rank_ic_context(context, nested_values)
    left = legacy_ic.sort_values("date").reset_index(drop=True)["rank_ic"].to_numpy(dtype=float)
    right = optimized_ic.sort_values("date").reset_index(drop=True)["rank_ic"].to_numpy(dtype=float)
    if not np.allclose(left, right, equal_nan=True, rtol=1e-12, atol=1e-12):
        raise AssertionError("Context Rank IC does not match the reference implementation.")

    legacy_ts_rank_seconds = _measure(lambda: _legacy_ts_rank(prepared), repeats)
    native_ts_rank_seconds = _measure(
        lambda: ts_rank_node.evaluate(prepared),
        repeats,
    )
    nested_expression_seconds = _measure(
        lambda: nested_node.evaluate(prepared),
        repeats,
    )
    legacy_rank_ic_seconds = _measure(
        lambda: _daily_cross_sectional_rank_ic(prepared["date"], nested_values, prepared["future_return"]),
        repeats,
    )
    context_rank_ic_seconds = _measure(
        lambda: _daily_cross_sectional_rank_ic_context(context, nested_values),
        repeats,
    )

    started = perf_counter()
    first = evaluator.evaluate_context(nested_node, context, mode="research")
    first_research_seconds = perf_counter() - started
    started = perf_counter()
    cached = evaluator.evaluate_context(nested_node, context, mode="research")
    cached_research_seconds = perf_counter() - started
    if first.metrics != cached.metrics:
        raise AssertionError("Cached research result changed deterministic metrics.")

    reuse_node = parse_expression("ts_rank(volume, 20)")
    evaluator.evaluate_context(reuse_node, context, mode="fast")
    evaluator.evaluate_context(reuse_node, context, mode="research")
    evaluator.evaluate_context(reuse_node, context, mode="research")

    return {
        "panel": {
            "path": panel_path,
            "start": start,
            "end": end,
            "rows": int(len(prepared)),
            "symbols": int(prepared["symbol"].nunique()),
            "dates": int(prepared["date"].nunique()),
            "holdout_materialized": False,
        },
        "median_seconds": {
            "legacy_python_ts_rank": legacy_ts_rank_seconds,
            "native_ts_rank": native_ts_rank_seconds,
            "nested_expression": nested_expression_seconds,
            "legacy_rank_ic": legacy_rank_ic_seconds,
            "context_rank_ic": context_rank_ic_seconds,
            "first_factor_research": first_research_seconds,
            "cached_factor_research": cached_research_seconds,
        },
        "speedup": {
            "ts_rank": legacy_ts_rank_seconds / native_ts_rank_seconds,
            "rank_ic": legacy_rank_ic_seconds / context_rank_ic_seconds,
            "cached_research": first_research_seconds / cached_research_seconds,
        },
        "instrumentation": evaluator.instrumentation_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-csv", default="crypto_data/binance_crypto30_daily/panel.csv")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=DEFAULT_RESEARCH_END)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.panel_csv, args.start, args.end, args.repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
