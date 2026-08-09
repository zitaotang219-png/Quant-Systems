"""Replay the Phase 1 factors over the Phase 2 point-in-time universe."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from alpha_mining.config import AlphaMiningConfig, EvaluationConfig, FitnessConfig, GPConfig, PortfolioConfig, RegistryConfig, RegimeConfig
from alpha_mining.pipeline import backtest_selected_factors
from alpha_mining.registry import FactorRegistry
from alpha_mining.run_crypto_workflow import FINAL_BACKTEST_END, FINAL_BACKTEST_START, slice_panel_by_date
from data_crypto.loader import load_crypto_panel_csv, prepare_point_in_time_crypto_panel
from features_crypto.engineer import build_crypto_panel_features
from utils.random_state import set_global_seed


def run() -> None:
    baseline = Path("reports/phase1_full_verify")
    output = Path("reports/phase2_comparison")
    replay = output / "point_in_time_fixed_factor_replay"
    replay.mkdir(parents=True, exist_ok=True)
    payload = json.loads((baseline / "workflow_config.json").read_text(encoding="utf-8"))
    config = _config_from_payload(payload)
    set_global_seed(config.gp.seed)
    selected = FactorRegistry(baseline / "alpha_mining_registry").load(config)
    if not selected:
        raise ValueError("Phase 1 baseline selected factors are unavailable.")

    raw = load_crypto_panel_csv("crypto_data/binance_crypto30_daily/panel.csv")
    prepared = prepare_point_in_time_crypto_panel(
        raw,
        asset_master_path="crypto_data/asset_master.csv",
        minimum_historical_bars=20,
        liquidity_threshold=0.0,
        minimum_data_completeness=1.0,
    )
    panel = build_crypto_panel_features(prepared.panel).reset_index(drop=True)
    final_panel = slice_panel_by_date(panel, start=FINAL_BACKTEST_START, end=FINAL_BACKTEST_END)
    regime_panel = panel.loc[pd.to_datetime(panel["date"]) <= pd.Timestamp(FINAL_BACKTEST_END)].copy()
    result, metrics, _ = backtest_selected_factors(
        panel=final_panel,
        config=config,
        selected_factors=selected,
        initial_capital=100000.0,
        output_dir=str(replay / "backtest"),
        regime_source_panel=regime_panel,
    )
    (replay / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([factor.summary_row() for factor in selected]).to_csv(replay / "selected_factors_summary.csv", index=False)
    prepared.universe.to_csv(replay / "point_in_time_universe.csv", index=False)
    _write_comparisons(baseline, replay, output, metrics)


def _config_from_payload(payload: dict) -> AlphaMiningConfig:
    evaluation = dict(payload["evaluation"])
    # Phase 1 artifacts retain this legacy evaluator-only field.
    evaluation.pop("future_return_horizon", None)
    return AlphaMiningConfig(
        gp=GPConfig(**payload["gp"]), evaluation=EvaluationConfig(**evaluation), fitness=FitnessConfig(**payload["fitness"]),
        regime=RegimeConfig(**payload["regime"]), portfolio=PortfolioConfig(**payload["portfolio"]), registry=RegistryConfig(**payload["registry"]),
        **{key: payload[key] for key in ("live_mode", "fast_filter_keep", "deep_eval_keep", "walk_forward_enabled", "walk_forward_train_fraction", "walk_forward_validation_fraction", "walk_forward_backtest_fraction", "walk_forward_min_folds", "deduplicate_expressions", "save_registry", "universe_symbols")},
    )


def _write_comparisons(baseline: Path, replay: Path, output: Path, metrics: dict) -> None:
    fixed_panel = pd.read_csv(baseline / "panel.csv", parse_dates=["date"])
    universe = pd.read_csv(replay / "point_in_time_universe.csv", parse_dates=["date"])
    fixed = fixed_panel.groupby("date")["symbol"].agg(lambda x: set(x)).rename("fixed")
    pit = universe.loc[universe["eligible"]].groupby("date")["symbol"].agg(lambda x: set(x)).rename("pit")
    rows=[]
    for date in fixed.index.union(pit.index):
        left=fixed.get(date,set()); right=pit.get(date,set())
        rows.append({"date":date,"fixed_constituent_count":len(left),"point_in_time_constituent_count":len(right),"additions":"|".join(sorted(right-left)),"removals":"|".join(sorted(left-right))})
    pd.DataFrame(rows).to_csv(output / "universe_comparison.csv",index=False)
    factors=pd.read_csv(baseline / "selected_factors_summary.csv")["expression"].astype(str)
    pd.DataFrame({"selected_factor":factors,"in_baseline":True,"in_point_in_time":True,"overlap_ratio":1.0}).to_csv(output / "factor_overlap.csv",index=False)
    base_w=pd.read_csv(baseline / "backtest/cross_sectional_weights.csv",parse_dates=["date"]); pit_w=pd.read_csv(replay / "backtest/cross_sectional_weights.csv",parse_dates=["date"])
    pr=[]
    for date in sorted(set(base_w.date)&set(pit_w.date)):
        a=base_w.loc[(base_w.date==date)&base_w.weight.ne(0),"symbol"]; b=pit_w.loc[(pit_w.date==date)&pit_w.weight.ne(0),"symbol"]; sa,setb=set(a),set(b); union=sa|setb
        pr.append({"date":date,"holdings_overlap":len(sa&setb)/len(union) if union else 1.0,"baseline_gross_exposure":base_w.loc[base_w.date==date,"weight"].abs().sum(),"point_in_time_gross_exposure":pit_w.loc[pit_w.date==date,"weight"].abs().sum()})
    portfolio=pd.DataFrame(pr); base_metrics=json.loads((baseline/"metrics.json").read_text()); portfolio["baseline_turnover"]=base_metrics["turnover"]; portfolio["point_in_time_turnover"]=metrics["turnover"]; portfolio.to_csv(output/"portfolio_comparison.csv",index=False)
    pd.DataFrame([{ "baseline_return":base_metrics["total_return"],"point_in_time_return":metrics["total_return"],"baseline_sharpe":base_metrics["sharpe"],"point_in_time_sharpe":metrics["sharpe"],"baseline_drawdown":base_metrics["max_drawdown"],"point_in_time_drawdown":metrics["max_drawdown"],"baseline_cost":base_metrics["total_cost"],"point_in_time_cost":metrics["total_cost"]}]).to_csv(output/"performance_comparison.csv",index=False)


if __name__ == "__main__":
    run()
