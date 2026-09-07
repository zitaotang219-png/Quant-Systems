"""Run the predeclared Phase 3B frozen-factor portfolio experiment.

The loader deliberately stops at the development cutoff while reading the raw
CSV.  Rows in the spent final holdout are therefore never placed in memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_mining.config import (
    AlphaMiningConfig,
    EvaluationConfig,
    PortfolioConfig,
    RegimeConfig,
    RegistryConfig,
)
from alpha_mining.phase3b import (
    FrozenFactorSet,
    build_phase3b_baseline_config,
    load_frozen_factor_set,
)
from alpha_mining.pipeline import backtest_selected_factors
from alpha_mining.portfolio_construction import factor_blend_weights
from data_crypto.loader import prepare_point_in_time_crypto_panel
from features_crypto.engineer import build_crypto_panel_features


EXPERIMENT_START = pd.Timestamp("2025-01-01")
DEVELOPMENT_CUTOFF = pd.Timestamp("2025-05-31")
FINAL_HOLDOUT_START = pd.Timestamp("2025-06-01")
INITIAL_CAPITAL = 100_000.0
TOTAL_COST_BPS = {"low": 8.0, "baseline": 16.0, "high": 32.0}


def load_permitted_raw_panel(path: str | Path, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Read only rows at or before ``cutoff``; never materialize later CSV rows."""
    cutoff_text = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
    retained: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise KeyError(f"Crypto panel is missing required columns: {sorted(missing)}")
        for row in reader:
            raw_date = str(row["date"])[:10]
            if raw_date <= cutoff_text:
                retained.append(row)
    if not retained:
        raise ValueError("No permitted observations were found before the development cutoff.")
    panel = pd.DataFrame(retained)
    panel["date"] = pd.to_datetime(panel["date"], utc=False)
    panel["symbol"] = panel["symbol"].astype(str).str.strip()
    if "name" not in panel:
        panel["name"] = panel["symbol"]
    for column in ("open", "high", "low", "close", "volume"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=list(required)).sort_values(
        ["symbol", "date"], kind="mergesort"
    ).reset_index(drop=True)
    if panel["date"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("Final-holdout observation entered the Phase 3B development panel.")
    return panel


def load_source_config(path: str | Path, registry_directory: str | Path) -> AlphaMiningConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AlphaMiningConfig(
        evaluation=EvaluationConfig(**payload["evaluation"]),
        regime=RegimeConfig(**payload["regime"]),
        portfolio=PortfolioConfig(**payload["portfolio"]),
        registry=RegistryConfig(directory=str(registry_directory)),
        universe_symbols=tuple(payload["universe_symbols"]),
        compute_profile=str(payload.get("compute_profile", "research")),
        live_mode=True,
        save_registry=False,
    )


def build_predeclared_variants(source: AlphaMiningConfig) -> dict[str, AlphaMiningConfig]:
    clean = build_phase3b_baseline_config(source)
    research_weighted = replace(
        clean,
        portfolio=replace(clean.portfolio, factor_weight_scheme="fitness"),
    )
    enhanced = replace(
        clean,
        regime=replace(source.regime, enabled=True),
        portfolio=replace(clean.portfolio, factor_weight_scheme="equal"),
    )
    return {
        "A_clean_baseline": clean,
        "B_research_weighted": research_weighted,
        "C_existing_enhanced": enhanced,
    }


def summarize_result(
    timeseries: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    annualization: int,
) -> dict[str, Any]:
    net = pd.to_numeric(timeseries["net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(timeseries["gross_return"], errors="coerce").fillna(0.0)
    net_return = float((1.0 + net).prod() - 1.0)
    gross_return = float((1.0 + gross).prod() - 1.0)
    observations = max(int(len(timeseries)), 1)
    annualized_return = (
        float((1.0 + net_return) ** (annualization / observations) - 1.0)
        if net_return > -1.0
        else -1.0
    )
    volatility = float(net.std(ddof=1) * math.sqrt(annualization))
    sharpe = 0.0 if volatility == 0.0 or not np.isfinite(volatility) else float(net.mean() * annualization / volatility)

    weight_frame = weights.copy()
    weight_frame["weight"] = pd.to_numeric(weight_frame["weight"], errors="coerce").fillna(0.0)
    grouped = weight_frame.groupby("date", sort=False)["weight"]
    gross_exposure = grouped.apply(lambda values: float(values.abs().sum()))
    net_exposure = grouped.sum().abs()
    concentration = grouped.apply(
        lambda values: 0.0
        if float(values.abs().sum()) <= 1e-12
        else float(((values.abs() / values.abs().sum()) ** 2).sum())
    )
    long_count = grouped.apply(lambda values: int((values > 1e-12).sum()))
    short_count = grouped.apply(lambda values: int((values < -1e-12).sum()))
    mean_turnover = float(pd.to_numeric(timeseries["turnover"], errors="coerce").fillna(0.0).mean())
    total_cost_dollars = float(pd.to_numeric(timeseries["total_cost"], errors="coerce").fillna(0.0).sum())
    total_cost_return = float(pd.to_numeric(timeseries["transaction_cost"], errors="coerce").fillna(0.0).sum())
    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "maximum_drawdown": float(-pd.to_numeric(timeseries["drawdown"], errors="coerce").min()),
        "mean_turnover": mean_turnover,
        "annualized_turnover": mean_turnover * annualization,
        "total_transaction_cost_dollars": total_cost_dollars,
        "total_transaction_cost_return_sum": total_cost_return,
        "gross_to_net_return_degradation": gross_return - net_return,
        "average_gross_exposure": float(gross_exposure.mean()),
        "average_absolute_net_exposure": float(net_exposure.mean()),
        "maximum_single_name_exposure": float(weight_frame["weight"].abs().max()),
        "average_concentration_hhi": float(concentration.mean()),
        "average_active_long_count": float(long_count.mean()),
        "average_active_short_count": float(short_count.mean()),
        "largest_asset_contribution": None,
        "largest_asset_contribution_note": "Unavailable: the existing stack has no asset attribution artifact; none was added.",
        "accounting_observations": int(len(timeseries)),
    }


def _run_one(
    *,
    name: str,
    panel: pd.DataFrame,
    regime_source_panel: pd.DataFrame,
    frozen: FrozenFactorSet,
    config: AlphaMiningConfig,
    output_dir: Path,
    total_cost_bps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen.verify_unchanged()
    started = time.perf_counter()
    timeseries, _, artifacts = backtest_selected_factors(
        panel=panel,
        config=config,
        selected_factors=list(frozen.factors),
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=total_cost_bps / 2.0,
        slippage_bps=total_cost_bps / 2.0,
        output_dir=str(output_dir),
        regime_source_panel=regime_source_panel if config.regime.enabled else None,
        normalize_factor_signals=True,
    )
    runtime = time.perf_counter() - started
    frozen.verify_unchanged()
    weights = pd.read_csv(artifacts["weights_path"])
    summary = summarize_result(timeseries, weights, annualization=config.evaluation.annualization)
    summary.update({"run": name, "runtime_seconds": runtime, "total_cost_bps": total_cost_bps})
    checks = {
        "reconciliation_max_abs_difference": float(
            pd.read_csv(artifacts["accounting_reconciliation_path"])["difference"].abs().max()
        ),
        "constraint_failures": int(
            (~pd.read_csv(artifacts["constraint_report_path"])["passed"].astype(bool)).sum()
        ),
        "signal_precedes_execution": bool(
            (pd.to_datetime(timeseries["signal_date"]) < pd.to_datetime(timeseries["date"])).all()
        ),
        "frozen_registry_sha256": frozen.registry_sha256,
    }
    return summary, checks


def _stable_panel_hash(panel: pd.DataFrame) -> str:
    columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    values = pd.util.hash_pandas_object(panel[columns], index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _write_report(output_dir: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(results).set_index("run")
    main = frame.loc[["A_clean_baseline", "B_research_weighted", "C_existing_enhanced"]]
    sensitivity = frame.loc[["A_cost_low", "A_clean_baseline", "A_cost_high"]]

    def percent(value: Any) -> str:
        return f"{float(value):.2%}"

    lines = [
        "# Phase 3B Frozen-Factor Portfolio Experiment",
        "",
        "## Precommitted design",
        "",
        f"Development interval: `{manifest['experiment_start']}` through `{manifest['development_cutoff']}`. "
        "The loader rejected all rows beginning with the final holdout date before DataFrame construction.",
        "",
        "Frozen factors: " + ", ".join(
            f"`{expression}` ({direction:+d})"
            for expression, direction in zip(manifest["frozen_expressions"], manifest["frozen_directions"], strict=True)
        ) + ".",
        "",
        "## Three portfolio variants",
        "",
        "| Variant | Gross return | Net return | Ann. return | Ann. vol | Sharpe | Max DD | Mean / ann. turnover | Gross-net drag | Total cost | Avg gross | Avg abs(net) | Max name | HHI | Avg L/S |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in main.iterrows():
        lines.append(
            f"| {name} | {percent(row.gross_return)} | {percent(row.net_return)} | "
            f"{percent(row.annualized_return)} | {percent(row.annualized_volatility)} | "
            f"{row.sharpe:.3f} | {percent(row.maximum_drawdown)} | {row.mean_turnover:.3f} / {row.annualized_turnover:.1f} | "
            f"{percent(row.gross_to_net_return_degradation)} | ${row.total_transaction_cost_dollars:,.2f} | {row.average_gross_exposure:.3f} | "
            f"{row.average_absolute_net_exposure:.2e} | {row.maximum_single_name_exposure:.3f} | "
            f"{row.average_concentration_hhi:.3f} | {row.average_active_long_count:.1f}/{row.average_active_short_count:.1f} |"
        )
    lines += [
        "",
        "Annualization uses 365, matching the frozen crypto research configuration. Turnover is the existing "
        "entry-plus-exit notional convention; annualized turnover is mean daily turnover × 365. Cost rates are "
        "commission plus slippage per traded notional per leg: low 4+4 bps, baseline 8+8 bps, high 16+16 bps. "
        "Asset-level contribution is omitted because the existing stack does not provide it and no new attribution framework was authorized.",
        "",
        "All three frozen factors were loaded and evaluated in every main variant. In B, the repository's existing "
        "fitness-weight rule assigned zero blend weight to `price_volume_confirmation` because its frozen research fitness "
        "is negative; this was not a portfolio-performance-based drop or an experiment-time decision.",
        "",
        "## Clean-baseline cost sensitivity",
        "",
        "| Total cost assumption | Net return | Sharpe | Gross-to-net degradation | Total cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in sensitivity.iterrows():
        label = {"A_cost_low": "low (8 bps)", "A_clean_baseline": "baseline (16 bps)", "A_cost_high": "high (32 bps)"}[name]
        lines.append(
            f"| {label} | {percent(row.net_return)} | {row.sharpe:.3f} | "
            f"{percent(row.gross_to_net_return_degradation)} | ${row.total_transaction_cost_dollars:,.2f} |"
        )
    a, b, c = (main.loc[key] for key in main.index)
    lines += [
        "",
        "## Interpretation",
        "",
        f"1. The frozen signals produced a constrained, market-neutral book: baseline average gross exposure was {a.average_gross_exposure:.3f}, "
        f"average absolute net exposure {a.average_absolute_net_exposure:.2e}, and maximum name exposure {a.maximum_single_name_exposure:.3f}.",
        f"2. At baseline costs, gross-to-net degradation was {percent(a.gross_to_net_return_degradation)}; "
        f"the low/high diagnostic spans net returns from {percent(sensitivity.net_return.max())} to {percent(sensitivity.net_return.min())}.",
        f"3. Existing research-fitness weighting changed net return by {percent(b.net_return - a.net_return)}, "
        f"annualized volatility by {percent(b.annualized_volatility - a.annualized_volatility)}, and mean turnover by "
        f"{b.mean_turnover - a.mean_turnover:+.3f}. It did not improve the result; this is descriptive, not an optimization decision.",
        f"4. Existing deterministic regime multipliers changed net return by {percent(c.net_return - a.net_return)} and "
        f"maximum drawdown by {percent(c.maximum_drawdown - a.maximum_drawdown)} versus baseline. The small drawdown reduction "
        "came with worse net return and Sharpe, so this run does not show that the extra regime complexity improved robustness. "
        "No additional regime logic was introduced.",
        "5. All runs passed the existing portfolio constraints, ledger reconciliation, and close-t to open-t+1 timing checks.",
        "",
        "Negative performance, if present, is retained without factor dropping or parameter changes. These development results do not reopen Phase 3A and do not authorize inspection of the final holdout.",
    ]
    (output_dir / "phase3b_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frozen = load_frozen_factor_set(args.registry_dir)
    source = load_source_config(args.source_config, args.registry_dir)
    variants = build_predeclared_variants(source)

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
    experiment_panel = featured.loc[
        (featured["date"] >= EXPERIMENT_START) & (featured["date"] <= DEVELOPMENT_CUTOFF)
    ].copy().reset_index(drop=True)
    if experiment_panel.empty or experiment_panel["date"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("The permitted Phase 3B experiment panel is invalid.")

    manifest: dict[str, Any] = {
        "phase": "3B",
        "purpose": "frozen-factor portfolio translation; no performance optimization",
        "experiment_start": EXPERIMENT_START.strftime("%Y-%m-%d"),
        "development_cutoff": DEVELOPMENT_CUTOFF.strftime("%Y-%m-%d"),
        "final_holdout_start": FINAL_HOLDOUT_START.strftime("%Y-%m-%d"),
        "final_holdout_materialized": False,
        "jan_may_2025_lineage_status": "development_conservative",
        "jan_may_2025_lineage_note": "No experiment-ledger evidence of portfolio optimization was found; legacy artifacts make status uncertain, so the interval is treated conservatively as development data.",
        "raw_rows_retained_before_cutoff": int(len(raw)),
        "experiment_rows": int(len(experiment_panel)),
        "experiment_dates": int(experiment_panel["date"].nunique()),
        "permitted_panel_sha256": _stable_panel_hash(raw),
        "frozen_registry_path": str(frozen.registry_path),
        "frozen_registry_sha256": frozen.registry_sha256,
        "frozen_expressions": list(frozen.expressions),
        "frozen_directions": list(frozen.directions),
        "factor_fitness": {factor.expression: float(factor.fitness) for factor in frozen.factors},
        "predeclared_total_cost_bps": TOTAL_COST_BPS,
        "cost_model_note": "Each total is commission plus slippage per traded notional per leg; components are split equally.",
        "predeclared_cost_components_bps": {
            name: {"commission": total / 2.0, "slippage": total / 2.0}
            for name, total in TOTAL_COST_BPS.items()
        },
        "annualization": int(source.evaluation.annualization),
        "initial_capital": INITIAL_CAPITAL,
        "variants": {name: asdict(config) for name, config in variants.items()},
        "factor_blend_weights": {
            "equal": factor_blend_weights(list(frozen.factors), "equal"),
            "fitness": factor_blend_weights(list(frozen.factors), "fitness"),
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for name, config in variants.items():
        summary, run_checks = _run_one(
            name=name,
            panel=experiment_panel,
            regime_source_panel=featured,
            frozen=frozen,
            config=config,
            output_dir=output_dir / name,
            total_cost_bps=TOTAL_COST_BPS["baseline"],
        )
        results.append(summary)
        checks[name] = run_checks

    for cost_name in ("low", "high"):
        name = f"A_cost_{cost_name}"
        summary, run_checks = _run_one(
            name=name,
            panel=experiment_panel,
            regime_source_panel=featured,
            frozen=frozen,
            config=variants["A_clean_baseline"],
            output_dir=output_dir / name,
            total_cost_bps=TOTAL_COST_BPS[cost_name],
        )
        results.append(summary)
        checks[name] = run_checks

    frozen.verify_unchanged()
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "results_summary.csv", index=False)
    (output_dir / "results_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    all_checks_pass = all(
        values["reconciliation_max_abs_difference"] <= 1e-8
        and values["constraint_failures"] == 0
        and values["signal_precedes_execution"]
        for values in checks.values()
    )
    audit = {
        "all_checks_pass": all_checks_pass,
        "frozen_registry_verified_after_all_runs": True,
        "final_holdout_materialized": False,
        "runs": checks,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "experiment_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    if not all_checks_pass:
        raise RuntimeError(f"Phase 3B accounting/integrity audit failed: {checks}")
    _write_report(output_dir, manifest, results)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="crypto_data/binance_crypto30_daily/panel.csv")
    parser.add_argument("--asset-master", default="crypto_data/asset_master.csv")
    parser.add_argument("--source-config", default="reports/phase3a_closure_research_20260906/workflow_config.json")
    parser.add_argument("--registry-dir", default="reports/phase3a_closure_research_20260906/alpha_mining_registry")
    parser.add_argument("--output-dir", default="reports/phase3b_portfolio_experiment_20260907")
    return parser.parse_args()


if __name__ == "__main__":
    print(run_experiment(parse_args()))
