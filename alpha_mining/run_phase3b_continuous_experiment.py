"""Compare frozen Phase 3B baseline under flat-session and continuous crypto execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_mining.phase3b import FrozenFactorSet, build_phase3b_baseline_config, load_frozen_factor_set
from alpha_mining.pipeline import backtest_selected_factors
from alpha_mining.portfolio_construction import cross_sectional_rank_normalize
from alpha_mining.research_evaluator import _daily_cross_sectional_rank_ic
from alpha_mining.run_phase3b_experiment import (
    DEVELOPMENT_CUTOFF,
    EXPERIMENT_START,
    FINAL_HOLDOUT_START,
    INITIAL_CAPITAL,
    load_permitted_raw_panel,
    load_source_config,
)
from data_crypto.loader import prepare_point_in_time_crypto_panel
from features_crypto.engineer import build_crypto_panel_features


BASELINE_COMMISSION_BPS = 8.0
BASELINE_SLIPPAGE_BPS = 8.0
HORIZONS = (1, 3, 5, 10)


def close_to_next_open_diagnostics(panel: pd.DataFrame) -> dict[str, float | int]:
    frame = panel.sort_values(["symbol", "date"], kind="mergesort")
    previous_close = frame.groupby("symbol", sort=False)["close"].shift(1)
    previous_date = frame.groupby("symbol", sort=False)["date"].shift(1)
    consecutive = (frame["date"] - previous_date == pd.Timedelta(days=1)) & previous_close.notna()
    gaps = (frame.loc[consecutive, "open"] / previous_close.loc[consecutive]) - 1.0
    absolute = gaps.abs()
    return {
        "observations": int(len(gaps)),
        "signed_mean": float(gaps.mean()),
        "signed_median": float(gaps.median()),
        "signed_std": float(gaps.std()),
        "absolute_median": float(absolute.median()),
        "absolute_p95": float(absolute.quantile(0.95)),
        "absolute_p99": float(absolute.quantile(0.99)),
        "absolute_max": float(absolute.max()),
    }


def compute_frozen_rank_ic_decay(panel: pd.DataFrame, frozen: FrozenFactorSet) -> pd.DataFrame:
    """Descriptive oriented IC for open t+1 through close t+h, with fixed horizons."""
    frozen.verify_unchanged()
    frame = panel.sort_values(["symbol", "date"], kind="mergesort").copy()
    by_symbol = frame.groupby("symbol", sort=False)
    next_open = by_symbol["open"].shift(-1)
    oriented_values: dict[str, pd.Series] = {}
    for factor in frozen.factors:
        raw = factor.node.evaluate(frame).astype(float) * int(factor.direction)
        oriented_values[factor.expression] = raw

    normalized = {
        expression: cross_sectional_rank_normalize(frame["date"], values)
        for expression, values in oriented_values.items()
    }
    composite = sum(normalized.values(), start=pd.Series(0.0, index=frame.index)) / float(len(normalized))
    series_to_report = oriented_values | {"equal_factor_composite": composite}

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        exit_close = by_symbol["close"].shift(-horizon)
        forward_return = (exit_close / next_open) - 1.0
        for expression, values in series_to_report.items():
            daily = _daily_cross_sectional_rank_ic(frame["date"], values, forward_return)
            clean = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
            rows.append(
                {
                    "expression": expression,
                    "horizon_days": horizon,
                    "return_interval": f"open_t+1_to_close_t+{horizon}",
                    "mean_rank_ic": float(clean.mean()) if not clean.empty else 0.0,
                    "median_rank_ic": float(clean.median()) if not clean.empty else 0.0,
                    "rank_ic_std": float(clean.std()) if len(clean) > 1 else 0.0,
                    "positive_ic_fraction": float((clean > 0.0).mean()) if not clean.empty else 0.0,
                    "date_observations": int(len(clean)),
                }
            )
    frozen.verify_unchanged()
    return pd.DataFrame(rows)


def _target_turnover(weights: pd.DataFrame, signal_dates: pd.Series) -> pd.Series:
    frame = weights.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    wide = frame.pivot(index="date", columns="symbol", values="weight").fillna(0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = wide.iloc[0].abs().sum()
    dates = pd.to_datetime(signal_dates, utc=False)
    return turnover.reindex(dates).fillna(0.0)


def summarize_execution(
    *,
    name: str,
    timeseries: pd.DataFrame,
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    reconciliation: pd.DataFrame,
    annualization: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    net = pd.to_numeric(timeseries["net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(timeseries["gross_return"], errors="coerce").fillna(0.0)
    gross_return = float((1.0 + gross).prod() - 1.0)
    net_return = float((1.0 + net).prod() - 1.0)
    realized_turnover = pd.to_numeric(timeseries["turnover"], errors="coerce").fillna(0.0)
    cumulative_turnover = float(realized_turnover.sum())
    gross_pnl = float(pd.to_numeric(timeseries["gross_pnl"], errors="coerce").fillna(0.0).sum())
    executed_notional = float(pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0).sum())
    target_turnover = _target_turnover(weights, timeseries["signal_date"])

    grouped = weights.groupby("date", sort=False)["weight"]
    return {
        "execution_mode": name,
        "gross_return": gross_return,
        "net_return": net_return,
        "mean_realized_turnover": float(realized_turnover.mean()),
        "annualized_turnover": float(realized_turnover.mean() * annualization),
        "cumulative_realized_turnover": cumulative_turnover,
        "mean_target_weight_turnover": float(target_turnover.mean()),
        "cumulative_target_weight_turnover": float(target_turnover.sum()),
        "transaction_cost_drag": gross_return - net_return,
        "total_transaction_cost_dollars": float(timeseries["total_cost"].sum()),
        "gross_alpha_per_unit_turnover": 0.0 if cumulative_turnover == 0.0 else gross_return / cumulative_turnover,
        "break_even_cost_bps_per_traded_notional": (
            0.0 if executed_notional == 0.0 else gross_pnl / executed_notional * 10_000.0
        ),
        "average_gross_exposure": float(grouped.apply(lambda value: value.abs().sum()).mean()),
        "average_absolute_net_exposure": float(grouped.sum().abs().mean()),
        "maximum_position": float(weights["weight"].abs().max()),
        "reconciliation_max_abs_error": float(reconciliation["difference"].abs().max()),
        "executed_notional_dollars": executed_notional,
        "runtime_seconds": runtime_seconds,
    }


def _run_mode(
    *,
    mode: str,
    panel: pd.DataFrame,
    frozen: FrozenFactorSet,
    config: Any,
    output_dir: Path,
) -> dict[str, Any]:
    frozen.verify_unchanged()
    started = time.perf_counter()
    timeseries, _, artifacts = backtest_selected_factors(
        panel=panel,
        config=config,
        selected_factors=list(frozen.factors),
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=BASELINE_COMMISSION_BPS,
        slippage_bps=BASELINE_SLIPPAGE_BPS,
        output_dir=str(output_dir),
        normalize_factor_signals=True,
        execution_mode=mode,
    )
    runtime = time.perf_counter() - started
    frozen.verify_unchanged()
    weights = pd.read_csv(artifacts["weights_path"])
    trades = pd.read_csv(artifacts["trade_ledger_path"])
    reconciliation = pd.read_csv(artifacts["accounting_reconciliation_path"])
    return summarize_execution(
        name=mode,
        timeseries=timeseries,
        weights=weights,
        trades=trades,
        reconciliation=reconciliation,
        annualization=config.evaluation.annualization,
        runtime_seconds=runtime,
    )


def _write_report(
    output_dir: Path,
    gap: dict[str, Any],
    execution: pd.DataFrame,
    decomposition: dict[str, Any],
    ic_decay: pd.DataFrame,
) -> None:
    def pct(value: Any) -> str:
        return f"{float(value):.2%}"

    lines = [
        "# Phase 3B Continuous-Crypto Execution Validation",
        "",
        "## Frozen scope",
        "",
        "The experiment uses the unchanged Phase-3A factors and directions, equal factor weighting, "
        "cross-sectional rank normalization, 0.70 gross leverage, 0.08 single-name limit, market neutrality, "
        "and baseline 8 bps commission plus 8 bps slippage per executed notional. Data ends on 2025-05-31; "
        "the final holdout was not materialized. No parameter or horizon search was performed.",
        "",
        "## Bar-boundary diagnosis",
        "",
        f"Across {gap['observations']:,} consecutive symbol-days, the close-to-next-open absolute gap was "
        f"{pct(gap['absolute_median'])} at the median, {pct(gap['absolute_p95'])} at P95, and "
        f"{pct(gap['absolute_p99'])} at P99. The detailed diagnosis was recorded before implementation in "
        "`bar_boundary_diagnosis.md`.",
        "",
        "## Execution comparison",
        "",
        "| Convention | Gross return | Net return | Mean / ann. executed turnover | Cost drag | Gross alpha / turnover | Break-even bps | Avg gross | Avg abs(net) | Max position | Reconciliation error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in execution.iterrows():
        lines.append(
            f"| {row.execution_mode} | {pct(row.gross_return)} | {pct(row.net_return)} | "
            f"{row.mean_realized_turnover:.4f} / {row.annualized_turnover:.2f} | "
            f"{pct(row.transaction_cost_drag)} | {row.gross_alpha_per_unit_turnover:.6f} | "
            f"{row.break_even_cost_bps_per_traded_notional:.3f} | {row.average_gross_exposure:.3f} | "
            f"{row.average_absolute_net_exposure:.2e} | {row.maximum_position:.3f} | "
            f"{row.reconciliation_max_abs_error:.2e} |"
        )
    lines += [
        "",
        "Target-weight turnover is distinct from executed turnover. Across the same rebalance dates, "
        f"cumulative target turnover was {decomposition['cumulative_target_turnover']:.4f}, flat-session "
        f"executed turnover was {decomposition['flat_cumulative_executed_turnover']:.4f}, and continuous "
        f"delta turnover was {decomposition['continuous_cumulative_executed_turnover']:.4f}. The exact "
        f"turnover avoided by removing forced flattening was {decomposition['avoided_forced_flattening_turnover']:.4f} "
        f"({decomposition['avoided_fraction_of_flat_turnover']:.2%} of flat-session turnover).",
        "",
        "The flat mode remains available for compatibility. The continuous mode marks existing holdings from "
        "the prior close to the next open, trades only the target-minus-current holding delta, then marks the "
        "remaining position to close without a forced exit.",
        "",
        "## Frozen-factor Rank-IC horizon diagnostics",
        "",
        "Returns are measured from open t+1 to close t+h. Directions remain frozen; horizons were declared "
        "in advance and are descriptive only.",
        "",
        "| Expression | 1d | 3d | 5d | 10d |",
        "|---|---:|---:|---:|---:|",
    ]
    pivot = ic_decay.pivot(index="expression", columns="horizon_days", values="mean_rank_ic")
    for expression, row in pivot.iterrows():
        lines.append(
            f"| `{expression}` | {row[1]:.4f} | {row[3]:.4f} | {row[5]:.4f} | {row[10]:.4f} |"
        )
    flat = execution.set_index("execution_mode").loc["flat_session"]
    continuous = execution.set_index("execution_mode").loc["continuous_crypto"]
    lines += [
        "",
        "## Product and cost audit",
        "",
        "The price data and symbol set are Binance spot-style OHLCV, while the accounting permits negative "
        "weights. The present result must therefore be classified as a **spot-price proxy for a spot-margin "
        "long/short portfolio**, not a fully specified tradable product. Spot borrow availability, borrow rates, "
        "margin requirements and forced liquidation are not modeled. If implemented with perpetual futures, "
        "funding, mark/index basis, contract specifications, liquidation and venue-specific fees would also be "
        "required. Shorting is not assumed costless for production interpretation; these omissions make reported "
        "net returns optimistic relative to a complete implementation.",
        "",
        "## Conclusion",
        "",
        f"Continuous rebalancing changed gross return from {pct(flat.gross_return)} to {pct(continuous.gross_return)} "
        f"and net return from {pct(flat.net_return)} to {pct(continuous.net_return)}, while reducing cumulative "
        f"executed turnover by {decomposition['avoided_fraction_of_flat_turnover']:.2%}. The failure is classified "
        "as **(3) a combination of both, dominated by artificial execution turnover**. Removing forced flattening "
        "preserved gross behavior and restored a small positive modeled net return, establishing (1) as the main "
        "cause. However, the remaining margin above the modeled 16 bps cost is narrow, and unmodeled borrow/funding/"
        "basis can consume it, so (2) cannot be ruled out for a real tradable product. The final classification is "
        "based on these untuned results and fixed horizon diagnostics; no Phase 3A decision is reopened.",
    ]
    (output_dir / "phase3b_continuous_crypto_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for protected in ("execution_comparison.csv", "experiment_manifest.json"):
        if (output_dir / protected).exists():
            raise FileExistsError(f"Refusing to overwrite completed experiment artifact: {output_dir / protected}")
    started = time.perf_counter()
    frozen = load_frozen_factor_set(args.registry_dir)
    source = load_source_config(args.source_config, args.registry_dir)
    baseline = build_phase3b_baseline_config(source)
    raw = load_permitted_raw_panel(args.panel, DEVELOPMENT_CUTOFF)
    raw = raw.loc[raw["symbol"].isin(source.universe_symbols)].copy()
    pit = prepare_point_in_time_crypto_panel(
        raw,
        asset_master_path=args.asset_master,
        minimum_historical_bars=20,
        liquidity_threshold=0.0,
        minimum_data_completeness=1.0,
    )
    featured = build_crypto_panel_features(pit.panel).reset_index(drop=True)
    panel = featured.loc[
        (featured["date"] >= EXPERIMENT_START) & (featured["date"] <= DEVELOPMENT_CUTOFF)
    ].copy().reset_index(drop=True)
    if panel.empty or panel["date"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("Final holdout entered the continuous-crypto experiment.")

    gap_panel = raw.loc[(raw["date"] >= EXPERIMENT_START) & (raw["date"] <= DEVELOPMENT_CUTOFF)].copy()
    gap = close_to_next_open_diagnostics(gap_panel)
    (output_dir / "gap_diagnostics.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")
    ic_decay = compute_frozen_rank_ic_decay(panel, frozen)
    ic_decay.to_csv(output_dir / "rank_ic_decay.csv", index=False)

    rows = [
        _run_mode(
            mode=mode,
            panel=panel,
            frozen=frozen,
            config=baseline,
            output_dir=output_dir / mode,
        )
        for mode in ("flat_session", "continuous_crypto")
    ]
    execution = pd.DataFrame(rows)
    execution.to_csv(output_dir / "execution_comparison.csv", index=False)
    by_mode = execution.set_index("execution_mode")
    flat_turnover = float(by_mode.at["flat_session", "cumulative_realized_turnover"])
    continuous_turnover = float(by_mode.at["continuous_crypto", "cumulative_realized_turnover"])
    avoided = flat_turnover - continuous_turnover
    decomposition = {
        "cumulative_target_turnover": float(by_mode.at["flat_session", "cumulative_target_weight_turnover"]),
        "flat_cumulative_executed_turnover": flat_turnover,
        "continuous_cumulative_executed_turnover": continuous_turnover,
        "avoided_forced_flattening_turnover": avoided,
        "avoided_fraction_of_flat_turnover": 0.0 if flat_turnover == 0.0 else avoided / flat_turnover,
    }
    (output_dir / "turnover_decomposition.json").write_text(
        json.dumps(decomposition, indent=2), encoding="utf-8"
    )
    frozen.verify_unchanged()
    audit = {
        "phase3a_frozen": True,
        "registry_sha256": frozen.registry_sha256,
        "final_holdout_materialized": False,
        "no_parameter_search": True,
        "execution_modes": ["flat_session", "continuous_crypto"],
        "continuous_costs_on_delta_only": True,
        "max_reconciliation_error": float(execution["reconciliation_max_abs_error"].max()),
        "all_reconciliation_checks_pass": bool((execution["reconciliation_max_abs_error"] <= 1e-8).all()),
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "experiment_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {
        "purpose": "untuned flat-session versus continuous-crypto execution comparison",
        "experiment_start": EXPERIMENT_START.strftime("%Y-%m-%d"),
        "development_cutoff": DEVELOPMENT_CUTOFF.strftime("%Y-%m-%d"),
        "final_holdout_start": FINAL_HOLDOUT_START.strftime("%Y-%m-%d"),
        "final_holdout_materialized": False,
        "frozen_registry_path": str(frozen.registry_path),
        "frozen_registry_sha256": frozen.registry_sha256,
        "frozen_expressions": list(frozen.expressions),
        "frozen_directions": list(frozen.directions),
        "portfolio_config": asdict(baseline.portfolio),
        "commission_bps_per_executed_notional": BASELINE_COMMISSION_BPS,
        "slippage_bps_per_executed_notional": BASELINE_SLIPPAGE_BPS,
        "predeclared_horizons_days": list(HORIZONS),
        "execution_modes": ["flat_session", "continuous_crypto"],
        "instrument_classification": "spot_price_proxy_for_spot_margin_long_short",
        "input_panel_rows": int(len(panel)),
        "input_panel_dates": int(panel["date"].nunique()),
        "input_panel_hash": hashlib.sha256(
            pd.util.hash_pandas_object(
                panel[["date", "symbol", "open", "high", "low", "close", "volume"]], index=False
            ).values.tobytes()
        ).hexdigest(),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    if not audit["all_reconciliation_checks_pass"]:
        raise RuntimeError(f"Continuous-crypto reconciliation failed: {audit}")
    _write_report(output_dir, gap, execution, decomposition, ic_decay)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="crypto_data/binance_crypto30_daily/panel.csv")
    parser.add_argument("--asset-master", default="crypto_data/asset_master.csv")
    parser.add_argument("--source-config", default="reports/phase3a_closure_research_20260906/workflow_config.json")
    parser.add_argument("--registry-dir", default="reports/phase3a_closure_research_20260906/alpha_mining_registry")
    parser.add_argument("--output-dir", default="reports/phase3b_continuous_crypto_20260907")
    return parser.parse_args()


if __name__ == "__main__":
    print(run_experiment(parse_args()))
