from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_mining import (
    AlphaMiningConfig,
    EvaluationConfig,
    FitnessConfig,
    FactorRegistry,
    GPConfig,
    PortfolioConfig,
    RegistryConfig,
    RegimeConfig,
    SelectedFactor,
    backtest_selected_factors,
    build_candidate_factor_pool,
    rescore_candidate_pool,
    select_factors_from_pool,
)
from alpha_mining.pipeline import _build_pool_evaluator, _single_split_map
from alpha_mining.regime import build_regime_frame, filter_regime_frame_by_dates, summarize_regime_frame
from data_crypto.loader import (
    build_crypto_panel_from_directory,
    load_crypto_benchmark_csv,
    load_crypto_panel_csv,
    load_crypto_universe_csv,
    prepare_point_in_time_crypto_panel,
)
from data_crypto.data_quality import write_data_quality_report
from data_crypto.market_cap import build_lagged_market_cap_weights
from features_crypto.engineer import build_crypto_panel_features, engineered_crypto_feature_columns
from utils.experiment_manifest import ExperimentManifest, create_experiment_directory
from utils.io import ensure_dir, write_json
from utils.random_state import set_global_seed
from universes.point_in_time import build_universe_report


DEFAULT_OUTPUT_DIR = Path("reports") / "crypto_alpha_workflow"
DEFAULT_INITIAL_CAPITAL = 100000.0
ROLLING_POOL_PURGE_BARS = 1
ROLLING_POOL_EMBARGO_BARS = 3
ROLLING_POOL_LIMIT = 96
ROLLING_WINDOW_POOL_LIMIT = 48
ROLLING_WINDOW_DEEP_KEEP = 64
ROLLING_WINDOW_FAST_KEEP = 96

ROLLING_WINDOW_SPECS = [
    {
        "name": "window_1",
        "train_start": "2024-01-01",
        "train_end": "2024-06-30",
        "validation_start": "2024-07-01",
        "validation_end": "2024-08-31",
        "fitness_profile": "defensive",
    },
    {
        "name": "window_2",
        "train_start": "2024-03-01",
        "train_end": "2024-08-31",
        "validation_start": "2024-09-01",
        "validation_end": "2024-10-31",
        "fitness_profile": "aggressive",
    },
    {
        "name": "window_3",
        "train_start": "2024-05-01",
        "train_end": "2024-10-31",
        "validation_start": "2024-11-01",
        "validation_end": "2024-12-31",
        "fitness_profile": "balanced",
    },
]
FINAL_BACKTEST_START = "2025-06-01"
FINAL_BACKTEST_END = "2026-01-31"

DEFAULT_CRYPTO30_UNIVERSE = [
    {"symbol": "BTCUSDT", "name": "Bitcoin", "coingecko_id": "bitcoin"},
    {"symbol": "ETHUSDT", "name": "Ethereum", "coingecko_id": "ethereum"},
    {"symbol": "SOLUSDT", "name": "Solana", "coingecko_id": "solana"},
    {"symbol": "XRPUSDT", "name": "XRP", "coingecko_id": "ripple"},
    {"symbol": "BNBUSDT", "name": "BNB", "coingecko_id": "binancecoin"},
    {"symbol": "ADAUSDT", "name": "Cardano", "coingecko_id": "cardano"},
    {"symbol": "DOGEUSDT", "name": "Dogecoin", "coingecko_id": "dogecoin"},
    {"symbol": "TRXUSDT", "name": "TRON", "coingecko_id": "tron"},
    {"symbol": "AVAXUSDT", "name": "Avalanche", "coingecko_id": "avalanche-2"},
    {"symbol": "LINKUSDT", "name": "Chainlink", "coingecko_id": "chainlink"},
    {"symbol": "TONUSDT", "name": "Toncoin", "coingecko_id": "the-open-network"},
    {"symbol": "DOTUSDT", "name": "Polkadot", "coingecko_id": "polkadot"},
    {"symbol": "MATICUSDT", "name": "Polygon", "coingecko_id": "matic-network"},
    {"symbol": "LTCUSDT", "name": "Litecoin", "coingecko_id": "litecoin"},
    {"symbol": "BCHUSDT", "name": "Bitcoin Cash", "coingecko_id": "bitcoin-cash"},
    {"symbol": "UNIUSDT", "name": "Uniswap", "coingecko_id": "uniswap"},
    {"symbol": "APTUSDT", "name": "Aptos", "coingecko_id": "aptos"},
    {"symbol": "NEARUSDT", "name": "NEAR", "coingecko_id": "near"},
    {"symbol": "ATOMUSDT", "name": "Cosmos", "coingecko_id": "cosmos"},
    {"symbol": "FILUSDT", "name": "Filecoin", "coingecko_id": "filecoin"},
    {"symbol": "ETCUSDT", "name": "Ethereum Classic", "coingecko_id": "ethereum-classic"},
    {"symbol": "OPUSDT", "name": "Optimism", "coingecko_id": "optimism"},
    {"symbol": "ARBUSDT", "name": "Arbitrum", "coingecko_id": "arbitrum"},
    {"symbol": "INJUSDT", "name": "Injective", "coingecko_id": "injective-protocol"},
    {"symbol": "SUIUSDT", "name": "Sui", "coingecko_id": "sui"},
    {"symbol": "SEIUSDT", "name": "Sei", "coingecko_id": "sei-network"},
    {"symbol": "RENDERUSDT", "name": "Render", "coingecko_id": "render-token"},
    {"symbol": "ICPUSDT", "name": "Internet Computer", "coingecko_id": "internet-computer"},
    {"symbol": "XLMUSDT", "name": "Stellar", "coingecko_id": "stellar"},
    {"symbol": "HBARUSDT", "name": "Hedera", "coingecko_id": "hedera-hashgraph"},
]

CRYPTO_UNIVERSE_PRESETS = {
    "crypto30": DEFAULT_CRYPTO30_UNIVERSE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the crypto alpha mining workflow from local CSV data.")
    parser.add_argument("--panel-csv", default=None, help="Combined crypto panel CSV with date,symbol,open,high,low,close,volume.")
    parser.add_argument("--panel-dir", default=None, help="Directory with one CSV per crypto symbol.")
    parser.add_argument("--universe-csv", default=None, help="Optional crypto universe CSV with columns symbol,name.")
    parser.add_argument("--universe-preset", default="crypto30", choices=sorted(CRYPTO_UNIVERSE_PRESETS.keys()), help="Named crypto universe preset.")
    parser.add_argument("--asset-master-csv", default="crypto_data/asset_master.csv", help="Asset lifecycle CSV used for point-in-time eligibility.")
    parser.add_argument("--minimum-historical-bars", type=int, default=20, help="Bars required before an asset becomes eligible.")
    parser.add_argument("--liquidity-threshold", type=float, default=0.0, help="Minimum median trailing dollar volume for eligibility.")
    parser.add_argument("--minimum-data-completeness", type=float, default=1.0, help="Required completeness ratio across trailing history.")
    parser.add_argument("--btc-benchmark-csv", default=None, help="Optional BTC benchmark CSV. Defaults to panel-derived BTC benchmark when available.")
    parser.add_argument("--market-cap-benchmark-csv", default=None, help="Optional market-cap-weighted benchmark CSV.")
    parser.add_argument("--benchmark-name", default="BTC", help="Readable BTC benchmark name.")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="Initial capital for backtest.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write workflow outputs.")
    parser.add_argument("--fitness-config", default=None, help="Optional JSON file with FitnessConfig overrides.")
    parser.add_argument("--quick", action="store_true", help="Use lighter GP/pool settings for fast smoke runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible GP candidate generation.")
    return parser.parse_args()


def print_terminal_progress(step: int, total_steps: int, title: str, detail: str | None = None) -> None:
    safe_total = max(int(total_steps), 1)
    safe_step = min(max(int(step), 0), safe_total)
    pct = int(round((safe_step / safe_total) * 100))
    bar_width = 28
    filled = int(round((safe_step / safe_total) * bar_width))
    bar = ("#" * filled) + ("-" * (bar_width - filled))
    print(f"[{bar}] {pct:>3}% | {title}")
    if detail:
        print(f"           {detail}")


def print_info(message: str) -> None:
    print(f"[info] {message}")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")
    args = parse_args()
    set_global_seed(args.seed)
    output_dir = create_experiment_directory(args.output_dir)
    ensure_dir(output_dir / "factors")
    ensure_dir(output_dir / "backtest")
    logs_dir = ensure_dir(output_dir / "logs")
    manifest = ExperimentManifest(
        output_dir,
        random_seed=args.seed,
        configuration_path=args.fitness_config,
        input_paths=workflow_input_paths(args),
    )
    manifest.save()
    write_text(logs_dir / "workflow.log", "status=running\n")
    total_steps = 7
    print_terminal_progress(1, total_steps, "Loading Local Data", f"Output dir: {output_dir.resolve()}")

    universe = load_crypto_universe_from_args(args)
    panel = load_or_build_crypto_panel(args)
    btc_benchmark_df = load_or_build_btc_benchmark(args, panel)
    equal_weight_benchmark_df = build_equal_weight_benchmark(panel, benchmark_name="crypto30_equal_weight")
    market_cap_benchmark_df = load_or_build_market_cap_benchmark(args, panel, universe)
    manifest.set_dataset_metadata(build_dataset_metadata(panel))
    manifest.save()
    print_info(
        "Loaded panel with "
        f"{int(panel['symbol'].astype(str).nunique())} symbols and "
        f"{len(pd.to_datetime(panel['date'], utc=False).dropna().unique())} trading dates"
    )
    panel.to_csv(output_dir / "panel.csv", index=False)
    point_in_time_universe = pd.DataFrame(panel.attrs.get("point_in_time_universe_records", []))
    asset_master = pd.DataFrame(panel.attrs.get("asset_master_records", []))
    data_quality_report = panel.attrs.get("data_quality_report", {})
    liquidity_sensitivity = pd.DataFrame(panel.attrs.get("liquidity_sensitivity_records", []))
    if not point_in_time_universe.empty:
        point_in_time_universe.to_csv(output_dir / "point_in_time_universe.csv", index=False)
        write_text(output_dir / "universe_report.md", build_universe_report(point_in_time_universe, asset_master, panel))
    write_data_quality_report(data_quality_report, output_dir / "data_quality_report.json")
    liquidity_sensitivity.to_csv(output_dir / "liquidity_sensitivity.csv", index=False)
    btc_benchmark_df.to_csv(output_dir / "btc_benchmark.csv", index=False)
    equal_weight_benchmark_df.to_csv(output_dir / "equal_weight_benchmark.csv", index=False)
    market_cap_benchmark_df.to_csv(output_dir / "market_cap_benchmark.csv", index=False)
    pd.DataFrame(universe).to_csv(output_dir / "universe.csv", index=False)

    fitness_override = load_fitness_override(args.fitness_config)
    config = build_crypto_workflow_config(
        output_dir=output_dir,
        symbols=sorted(panel["symbol"].astype(str).unique()),
        panel=panel,
        fitness_override=fitness_override,
        quick=args.quick,
        seed=args.seed,
    )
    config = config.__class__(
        gp=config.gp,
        evaluation=config.evaluation,
        fitness=config.fitness,
        regime=config.regime,
        portfolio=config.portfolio,
        registry=config.registry,
        live_mode=config.live_mode,
        fast_filter_keep=config.fast_filter_keep,
        deep_eval_keep=config.deep_eval_keep,
        walk_forward_enabled=False,
        walk_forward_train_fraction=config.walk_forward_train_fraction,
        walk_forward_validation_fraction=config.walk_forward_validation_fraction,
        walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
        walk_forward_min_folds=config.walk_forward_min_folds,
        deduplicate_expressions=config.deduplicate_expressions,
        save_registry=True,
        universe_symbols=config.universe_symbols,
    )
    config_payload = to_jsonable(asdict(config))
    write_json(output_dir / "workflow_config.json", config_payload)
    manifest.set_config(portable_config_snapshot(config_payload))
    manifest.save()
    print_terminal_progress(2, total_steps, "Preparing Splits And Config")

    research_panel = slice_panel_by_date(
        panel,
        start=ROLLING_WINDOW_SPECS[0]["train_start"],
        end=ROLLING_WINDOW_SPECS[-1]["validation_end"],
    )
    final_backtest_panel = slice_panel_by_date(panel, start=FINAL_BACKTEST_START, end=FINAL_BACKTEST_END)
    if research_panel.empty or final_backtest_panel.empty:
        raise ValueError("Resolved rolling workflow windows produced an empty research or final backtest panel.")
    research_panel.to_csv(output_dir / "research_panel.csv", index=False)
    final_backtest_panel.to_csv(output_dir / "backtest_panel.csv", index=False)
    train_summary_panel = pd.concat(
        [slice_panel_by_date(research_panel, start=spec["train_start"], end=spec["train_end"]) for spec in ROLLING_WINDOW_SPECS],
        ignore_index=True,
    ).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)
    validation_summary_panel = pd.concat(
        [slice_panel_by_date(research_panel, start=spec["validation_start"], end=spec["validation_end"]) for spec in ROLLING_WINDOW_SPECS],
        ignore_index=True,
    ).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)

    full_regime_frame = build_regime_frame(panel, config.regime)
    validation_regime_frame = filter_regime_frame_by_dates(
        full_regime_frame,
        pd.concat(
            [slice_panel_by_date(research_panel, start=spec["validation_start"], end=spec["validation_end"])["date"] for spec in ROLLING_WINDOW_SPECS],
            ignore_index=True,
        ),
    )
    backtest_regime_frame = filter_regime_frame_by_dates(full_regime_frame, final_backtest_panel["date"])
    regime_dir = ensure_dir(output_dir / "regime")
    full_regime_frame.to_csv(regime_dir / "full_regime_history.csv", index=False)
    validation_regime_frame.to_csv(regime_dir / "validation_regime_history.csv", index=False)
    backtest_regime_frame.to_csv(regime_dir / "backtest_regime_history.csv", index=False)
    regime_summary = {
        "full": summarize_regime_frame(full_regime_frame),
        "validation": summarize_regime_frame(validation_regime_frame),
        "backtest": summarize_regime_frame(backtest_regime_frame),
    }
    write_json(regime_dir / "regime_summary.json", to_jsonable(regime_summary))
    print_terminal_progress(3, total_steps, "Building Regime Snapshots")

    rolling_workflow = run_rolling_pool_workflow(
        panel=research_panel,
        config=config,
        output_dir=output_dir,
        purge_bars=ROLLING_POOL_PURGE_BARS,
        embargo_bars=ROLLING_POOL_EMBARGO_BARS,
        progress_callback=print_info,
    )
    print_terminal_progress(
        4,
        total_steps,
        "Rolling Pool Complete",
        f"Pool size {rolling_workflow['rolling_summary']['final_pool_size']}, "
        f"selected {rolling_workflow['rolling_summary']['final_selected_factor_count']}",
    )
    selected = refine_selected_factors_before_backtest(
        selected=rolling_workflow["selected_factors"],
        validation_panel=validation_summary_panel,
        config=config,
        initial_capital=args.initial_capital,
        regime_source_panel=research_panel,
        min_count=config.portfolio.min_selected_factor_count,
        output_dir=output_dir,
        progress_callback=print_info,
    )
    print_terminal_progress(5, total_steps, "Validation Refinement Complete", f"Retained {len(selected)} factors")
    FactorRegistry(config.registry_dir()).save(selected, config, research_panel)
    write_selected_factors(output_dir, selected)
    backtest_regime_source_panel = pd.concat(
        [research_panel, final_backtest_panel],
        ignore_index=True,
    ).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)

    print_terminal_progress(6, total_steps, "Running Final Backtest", f"Window {FINAL_BACKTEST_START} -> {FINAL_BACKTEST_END}")
    result, metrics, _ = backtest_selected_factors(
        panel=final_backtest_panel,
        config=config,
        selected_factors=selected,
        initial_capital=args.initial_capital,
        output_dir=str(output_dir / "backtest"),
        regime_source_panel=backtest_regime_source_panel,
    )
    backtest_btc = summarize_market_baseline(btc_benchmark_df, result["date"], args.initial_capital)
    backtest_equal_weight = summarize_market_baseline(equal_weight_benchmark_df, result["date"], args.initial_capital)
    backtest_market_cap = summarize_market_baseline(market_cap_benchmark_df, result["date"], args.initial_capital)
    write_json(output_dir / "backtest" / "backtest_metrics.json", to_jsonable(metrics))
    write_json(output_dir / "metrics.json", to_jsonable(metrics))
    write_json(output_dir / "backtest" / "backtest_vs_btc.json", to_jsonable(backtest_btc))
    write_json(output_dir / "backtest" / "backtest_vs_equal_weight.json", to_jsonable(backtest_equal_weight))
    write_json(output_dir / "backtest" / "backtest_vs_market_cap.json", to_jsonable(backtest_market_cap))
    write_text(
        output_dir / "backtest" / "backtest_report.md",
        build_backtest_report(
            initial_capital=args.initial_capital,
            result=result,
            metrics=metrics,
            backtest_btc=backtest_btc,
            backtest_equal_weight=backtest_equal_weight,
            backtest_market_cap=backtest_market_cap,
            walk_forward_summary=rolling_workflow["rolling_summary"],
            selected_factor_expressions=[factor.expression for factor in selected],
            split_summary=workflow_split_summary(
                research_panel=research_panel,
                train_panel=train_summary_panel,
                validation_panel=validation_summary_panel,
                backtest_panel=final_backtest_panel,
            ),
        ),
    )

    backtest_report_path = output_dir / "backtest" / "backtest_report.md"
    print_terminal_progress(
        7,
        total_steps,
        "Final Backtest Complete",
        f"Return {metrics.get('total_return', 0.0):.2%}, Sharpe {metrics.get('sharpe', 0.0):.3f}",
    )

    split_summary = workflow_split_summary(
        research_panel=research_panel,
        train_panel=train_summary_panel,
        validation_panel=validation_summary_panel,
        backtest_panel=final_backtest_panel,
    )
    workflow_summary = {
        "market": "crypto",
        "split": split_summary,
        "research_backtest_target": "rolling_pool_then_final_backtest",
        "selected_factor_count": len(selected),
        "selected_factor_expressions": [factor.expression for factor in selected],
        "rolling_window_summary": rolling_workflow["rolling_summary"],
        "regime_summary": regime_summary,
        "backtest_metrics": metrics,
        "backtest_vs_btc": backtest_btc,
        "backtest_vs_equal_weight": backtest_equal_weight,
        "backtest_vs_market_cap": backtest_market_cap,
    }
    write_json(output_dir / "workflow_summary.json", to_jsonable(workflow_summary))
    workflow_report_path = output_dir / "workflow_report.md"
    write_text(
        workflow_report_path,
        build_workflow_report_index(
            workflow_summary=workflow_summary,
            backtest_report_path=backtest_report_path,
            paper_report_path=None,
        ),
    )

    generated_factor_count = sum(
        int(window.get("new_candidate_count", 0))
        for window in rolling_workflow["rolling_summary"].get("windows", [])
    )
    manifest.set_research_results(
        generated_factor_count=generated_factor_count,
        selected_factors=[factor.expression for factor in selected],
    )
    manifest.save(status="completed")
    write_text(logs_dir / "workflow.log", "status=completed\n")

    print("Crypto rolling-pool workflow complete.")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Selected factors: {len(selected)}")
    print(f"Workflow report: {workflow_report_path.resolve()}")
    print(f"Backtest report: {backtest_report_path.resolve()}")
    print(f"Backtest metrics: {output_dir / 'backtest' / 'backtest_metrics.json'}")


def load_or_build_crypto_panel(args: argparse.Namespace) -> pd.DataFrame:
    if args.panel_csv:
        panel = load_crypto_panel_csv(args.panel_csv)
    elif args.panel_dir:
        panel = build_crypto_panel_from_directory(args.panel_dir, universe_csv=args.universe_csv)
    else:
        raise ValueError("Provide either --panel-csv or --panel-dir for the crypto workflow.")
    universe = {record["symbol"] for record in load_crypto_universe_from_args(args)}
    panel = panel.loc[panel["symbol"].astype(str).isin(universe)].copy()
    if panel.empty:
        raise ValueError("No rows remain after applying the crypto universe filter.")
    point_in_time = prepare_point_in_time_crypto_panel(
        panel,
        asset_master_path=getattr(args, "asset_master_csv", "crypto_data/asset_master.csv"),
        minimum_historical_bars=max(int(getattr(args, "minimum_historical_bars", 20)), 1),
        liquidity_threshold=max(float(getattr(args, "liquidity_threshold", 0.0)), 0.0),
        minimum_data_completeness=min(max(float(getattr(args, "minimum_data_completeness", 1.0)), 0.0), 1.0),
    )
    featured = build_crypto_panel_features(point_in_time.panel).reset_index(drop=True)
    # attrs must remain equality-safe because pandas compares them during concat.
    featured.attrs["point_in_time_universe_records"] = point_in_time.universe.to_dict(orient="records")
    featured.attrs["asset_master_records"] = point_in_time.asset_master.to_dict(orient="records")
    featured.attrs["data_quality_report"] = point_in_time.data_quality.report
    featured.attrs["liquidity_sensitivity_records"] = point_in_time.liquidity_sensitivity.to_dict(orient="records")
    return featured


def workflow_input_paths(args: argparse.Namespace) -> list[str]:
    return [
        str(path)
        for path in (
            args.panel_csv,
            args.panel_dir,
            args.universe_csv,
            args.asset_master_csv,
            args.btc_benchmark_csv,
            args.market_cap_benchmark_csv,
            args.fitness_config,
        )
        if path
    ]


def build_dataset_metadata(panel: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(panel["date"], utc=False)
    return {
        "row_count": int(len(panel)),
        "column_count": int(len(panel.columns)),
        "columns": [str(column) for column in panel.columns],
        "symbol_count": int(panel["symbol"].astype(str).nunique()),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
    }


def portable_config_snapshot(config_payload: dict[str, Any]) -> dict[str, Any]:
    """Remove run-directory identity while preserving every research parameter."""

    snapshot = json.loads(json.dumps(config_payload))
    registry = snapshot.get("registry")
    if isinstance(registry, dict) and "directory" in registry:
        registry["directory"] = "alpha_mining_registry"
    return snapshot


def load_or_build_btc_benchmark(args: argparse.Namespace, panel: pd.DataFrame) -> pd.DataFrame:
    if args.btc_benchmark_csv:
        benchmark = load_crypto_benchmark_csv(args.btc_benchmark_csv, benchmark_name=args.benchmark_name)
        if "return" in benchmark.columns or "open" in benchmark.columns:
            return benchmark
        # A close-only series is not comparable with this workflow's next-open to close return.
        return build_symbol_benchmark(panel, symbol="BTCUSDT", benchmark_name=args.benchmark_name)
    return build_symbol_benchmark(panel, symbol="BTCUSDT", benchmark_name=args.benchmark_name)


def load_or_build_market_cap_benchmark(
    args: argparse.Namespace,
    panel: pd.DataFrame,
    universe: list[dict[str, str]],
) -> pd.DataFrame:
    if args.market_cap_benchmark_csv:
        benchmark = load_crypto_benchmark_csv(args.market_cap_benchmark_csv, benchmark_name="crypto30_market_cap_weighted")
        if "return" in benchmark.columns or "open" in benchmark.columns:
            return benchmark
        return build_market_cap_benchmark(panel, universe, benchmark_name="crypto30_market_cap_weighted")
    return build_market_cap_benchmark(panel, universe, benchmark_name="crypto30_market_cap_weighted")


def load_crypto_universe_from_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.universe_csv:
        return load_crypto_universe_csv(args.universe_csv)
    return [dict(record) for record in CRYPTO_UNIVERSE_PRESETS[str(args.universe_preset)]]


def build_equal_weight_benchmark(panel: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    ordered["intraday_return"] = (ordered["close"] / ordered["open"]) - 1.0
    daily_return = ordered.groupby("date", sort=False)["intraday_return"].mean().fillna(0.0)
    close = 100.0 * (1.0 + daily_return).cumprod()
    return pd.DataFrame({"date": daily_return.index, "return": daily_return.to_numpy(dtype=float), "close": close.to_numpy(dtype=float), "name": benchmark_name})


def build_symbol_benchmark(panel: pd.DataFrame, symbol: str, benchmark_name: str) -> pd.DataFrame:
    frame = panel.loc[panel["symbol"].astype(str).str.upper() == str(symbol).upper(), ["date", "open", "close"]].copy()
    if frame.empty:
        raise ValueError(f"Could not derive benchmark for symbol {symbol} from panel.")
    frame["name"] = benchmark_name
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


def build_market_cap_benchmark(
    panel: pd.DataFrame,
    universe: list[dict[str, str]],
    benchmark_name: str,
) -> pd.DataFrame:
    lagged_weights = build_lagged_market_cap_weights(panel)
    if not lagged_weights.empty and float(lagged_weights["weight"].sum()) > 0.0:
        merged = panel[["date", "symbol", "open", "close"]].merge(lagged_weights, on=["date", "symbol"], how="left")
        merged["intraday_return"] = (merged["close"] / merged["open"]) - 1.0
        weighted = merged.groupby("date", sort=False).apply(
            lambda day: float((day["intraday_return"].fillna(0.0) * day["weight"].fillna(0.0)).sum())
        )
        close = 100.0 * (1.0 + weighted).cumprod()
        return pd.DataFrame({"date": weighted.index, "return": weighted.to_numpy(dtype=float), "close": close.to_numpy(dtype=float), "name": benchmark_name})

    static_name = "static_reference_weight_benchmark"
    weight_map: dict[str, float] = {}
    for record in universe:
        symbol = str(record["symbol"]).upper()
        weight = record.get("market_cap_weight")
        if weight is None:
            continue
        weight_map[symbol] = float(weight)
    if not weight_map:
        return build_equal_weight_benchmark(panel, benchmark_name=static_name)

    weight_total = sum(max(weight, 0.0) for weight in weight_map.values())
    if weight_total <= 0.0:
        return build_equal_weight_benchmark(panel, benchmark_name=static_name)
    normalized = {symbol: max(weight, 0.0) / weight_total for symbol, weight in weight_map.items()}

    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    ordered["future_weight"] = ordered["symbol"].astype(str).str.upper().map(normalized).fillna(0.0)
    available_weight = ordered.groupby("date", sort=False)["future_weight"].transform("sum")
    ordered["future_weight"] = (ordered["future_weight"] / available_weight).fillna(0.0)
    ordered["intraday_return"] = (ordered["close"] / ordered["open"]) - 1.0
    weighted = ordered.groupby("date", sort=False).apply(
        lambda day: float((day["intraday_return"].fillna(0.0) * day["future_weight"]).sum())
    )
    close = 100.0 * (1.0 + weighted).cumprod()
    return pd.DataFrame({"date": weighted.index, "return": weighted.to_numpy(dtype=float), "close": close.to_numpy(dtype=float), "name": static_name})


def build_crypto_workflow_config(
    output_dir: Path,
    symbols: list[str],
    panel: pd.DataFrame,
    fitness_override: FitnessConfig | None = None,
    quick: bool = False,
    seed: int = 42,
) -> AlphaMiningConfig:
    feature_fields = tuple(engineered_crypto_feature_columns(panel))
    gp = GPConfig(
        population_size=40 if quick else 60,
        generations=3 if quick else 4,
        tournament_size=5,
        elitism=8,
        max_depth=5,
        init_max_depth=4,
        crossover_rate=0.45,
        subtree_mutation_rate=0.20,
        point_mutation_rate=0.15,
        reproduction_rate=0.20,
        seed=int(seed),
        field_names=("open", "high", "low", "close", "volume", *feature_fields),
        constant_range=(-2.0, 2.0),
        periods_choices=(1, 2, 3, 5, 10, 20),
        window_choices=(3, 5, 10, 20, 30),
        wrap_final_with_rank_or_zscore=True,
    )
    evaluation = EvaluationConfig(
        transaction_cost_bps=8.0,
        slippage_bps=8.0,
        long_quantile=0.3,
        short_quantile=0.3,
        min_finite_ratio=0.6,
        min_std=1e-6,
        min_abs_rank_ic=0.0025,
        max_turnover=2.0,
        max_allowed_drawdown=0.22,
        ic_sign_tolerance=0.002,
        annualization=365,
    )
    fitness = merge_fitness_config(
        FitnessConfig(
            fast_rank_ic_weight=5.0,
            validation_ic_weight=10.0,
            sharpe_weight=2.0,
            cumulative_return_weight=3.0,
            excess_return_weight=6.0,
            stability_weight=8.0,
            bear_return_weight=10.0,
            bear_sharpe_weight=2.5,
            turnover_penalty=6.0,
            drawdown_penalty=24.0,
            complexity_penalty=0.10,
        ),
        fitness_override,
    )
    regime = RegimeConfig(
        enabled=True,
        alpha=0.1,
        threshold=0.2,
        trend_window=20,
        volatility_window=20,
        zscore_window=252,
        return_column="daily_return",
        use_style_multipliers=True,
        factor_regime_weight_overrides={},
    )
    portfolio = PortfolioConfig(
        selected_factor_count=8 if quick else 10,
        min_selected_factor_count=6 if quick else 8,
        max_pairwise_correlation=0.45,
        factor_weight_scheme="equal",
        weighting_scheme="continuous",
        position_limit=0.08,
        turnover_limit=1.2,
        gross_leverage=0.70,
        signal_vol_window=20,
        signal_clip=2.5,
        smoothing=0.75,
        market_neutral=True,
        benchmark_follow_enabled=True,
        benchmark_follow_btc_symbol="BTCUSDT",
        benchmark_follow_btc_weight=0.35,
        regime_benchmark_blend={
            "bull_low_vol": 0.15,
            "bull_high_vol": 0.05,
            "bear_low_vol": 0.35,
            "bear_high_vol": 0.60,
        },
        regime_benchmark_direction={
            "bull_low_vol": 1.0,
            "bull_high_vol": 0.5,
            "bear_low_vol": -1.0,
            "bear_high_vol": -1.0,
        },
    )
    registry = RegistryConfig(directory=str(output_dir / "alpha_mining_registry"))
    return AlphaMiningConfig(
        gp=gp,
        evaluation=evaluation,
        fitness=fitness,
        regime=regime,
        portfolio=portfolio,
        registry=registry,
        live_mode=False,
        fast_filter_keep=20 if quick else 36,
        deep_eval_keep=10 if quick else 18,
        walk_forward_enabled=True,
        save_registry=True,
        universe_symbols=tuple(symbols),
    )


def slice_panel_by_date(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    return frame.loc[mask].copy().reset_index(drop=True)


def build_window_fitness_profiles(base: FitnessConfig) -> dict[str, FitnessConfig]:
    return {
        "defensive": FitnessConfig(
            fast_rank_ic_weight=base.fast_rank_ic_weight,
            validation_ic_weight=10.0,
            sharpe_weight=2.2,
            cumulative_return_weight=2.5,
            excess_return_weight=5.5,
            stability_weight=9.0,
            bear_return_weight=12.0,
            bear_sharpe_weight=3.0,
            turnover_penalty=6.5,
            drawdown_penalty=26.0,
            complexity_penalty=0.11,
        ),
        "aggressive": FitnessConfig(
            fast_rank_ic_weight=max(base.fast_rank_ic_weight, 6.0),
            validation_ic_weight=10.0,
            sharpe_weight=1.4,
            cumulative_return_weight=4.0,
            excess_return_weight=7.0,
            stability_weight=6.0,
            bear_return_weight=6.0,
            bear_sharpe_weight=1.2,
            turnover_penalty=4.0,
            drawdown_penalty=18.0,
            complexity_penalty=0.08,
        ),
        "balanced": FitnessConfig(
            fast_rank_ic_weight=base.fast_rank_ic_weight,
            validation_ic_weight=10.0,
            sharpe_weight=1.8,
            cumulative_return_weight=3.5,
            excess_return_weight=6.0,
            stability_weight=8.0,
            bear_return_weight=9.0,
            bear_sharpe_weight=2.0,
            turnover_penalty=5.0,
            drawdown_penalty=22.0,
            complexity_penalty=0.09,
        ),
    }


def clone_config_with_fitness(config: AlphaMiningConfig, fitness: FitnessConfig, save_registry: bool = False) -> AlphaMiningConfig:
    return AlphaMiningConfig(
        gp=config.gp,
        evaluation=config.evaluation,
        fitness=fitness,
        regime=config.regime,
        portfolio=config.portfolio,
        registry=config.registry,
        live_mode=config.live_mode,
        fast_filter_keep=config.fast_filter_keep,
        deep_eval_keep=config.deep_eval_keep,
        walk_forward_enabled=False,
        walk_forward_train_fraction=config.walk_forward_train_fraction,
        walk_forward_validation_fraction=config.walk_forward_validation_fraction,
        walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
        walk_forward_min_folds=config.walk_forward_min_folds,
        deduplicate_expressions=config.deduplicate_expressions,
        save_registry=save_registry,
        universe_symbols=config.universe_symbols,
    )


def apply_purge_and_embargo(train_panel: pd.DataFrame, validation_panel: pd.DataFrame, purge_bars: int, embargo_bars: int) -> pd.DataFrame:
    if train_panel.empty or validation_panel.empty:
        return train_panel.copy()
    train_dates = sorted(pd.to_datetime(train_panel["date"], utc=False).dropna().unique())
    if not train_dates:
        return train_panel.copy()
    buffer = int(max(purge_bars, 0) + max(embargo_bars, 0))
    if buffer <= 0:
        return train_panel.copy()
    if len(train_dates) <= buffer:
        raise ValueError("Purging and embargo removed the entire training window.")
    kept_dates = train_dates[:-buffer]
    filtered = train_panel.loc[pd.to_datetime(train_panel["date"], utc=False).isin(kept_dates)].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Purging and embargo produced an empty training panel.")
    return filtered


def revalidate_factor_pool(
    candidate_pool: list[SelectedFactor],
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    fitness_config: FitnessConfig,
) -> list[SelectedFactor]:
    if not candidate_pool or panel.empty:
        return []
    evaluator = _build_pool_evaluator(config)
    split_map = _single_split_map(panel["date"], "validation")
    validated: list[SelectedFactor] = []
    for factor in candidate_pool:
        evaluation = evaluator.evaluate_with_splits(factor.node, panel, split_map)
        if evaluation.fitness <= -1_000_000_000.0:
            continue
        metrics = dict(evaluation.metrics)
        previous_passes = int(factor.metrics.get("window_pass_count", 0))
        if previous_passes > 0:
            metrics = merge_numeric_metrics(dict(factor.metrics), metrics, previous_passes)
        metrics["window_pass_count"] = previous_passes + 1
        validated.append(
            SelectedFactor(
                expression=factor.expression,
                node=factor.node,
                direction=evaluation.direction,
                fitness=float(evaluation.fitness),
                metrics=metrics,
                complexity=factor.complexity,
                finite_ratio=float(evaluation.finite_ratio),
                values=evaluation.values,
            )
        )
    rescored = rescore_candidate_pool(validated, fitness_config)
    rescored.sort(key=lambda item: item.fitness, reverse=True)
    return rescored


def merge_numeric_metrics(previous: dict[str, Any], current: dict[str, Any], previous_count: int) -> dict[str, Any]:
    merged: dict[str, Any] = dict(current)
    for key, previous_value in previous.items():
        current_value = current.get(key)
        if isinstance(previous_value, (int, float, np.floating)) and isinstance(current_value, (int, float, np.floating)):
            merged[key] = ((float(previous_value) * float(previous_count)) + float(current_value)) / float(previous_count + 1)
    return merged


def dedupe_factor_pool(candidate_pool: list[SelectedFactor]) -> list[SelectedFactor]:
    best_by_expression: dict[str, SelectedFactor] = {}
    for factor in candidate_pool:
        previous = best_by_expression.get(factor.expression)
        if previous is None or float(factor.fitness) > float(previous.fitness):
            best_by_expression[factor.expression] = factor
    ordered = sorted(best_by_expression.values(), key=lambda item: item.fitness, reverse=True)
    return ordered


def trim_candidate_pool(candidate_pool: list[SelectedFactor], pool_limit: int) -> list[SelectedFactor]:
    deduped = dedupe_factor_pool(candidate_pool)
    if len(deduped) <= pool_limit:
        return deduped
    trimmed: list[SelectedFactor] = []
    seen_series: list[pd.Series] = []
    for factor in deduped:
        if factor.values is None:
            continue
        if any(_series_corr_too_high(factor.values, existing) for existing in seen_series):
            continue
        trimmed.append(factor)
        seen_series.append(factor.values)
        if len(trimmed) >= pool_limit:
            break
    if len(trimmed) < min(pool_limit, len(deduped)):
        used = {factor.expression for factor in trimmed}
        for factor in deduped:
            if factor.expression in used:
                continue
            trimmed.append(factor)
            if len(trimmed) >= pool_limit:
                break
    return trimmed[:pool_limit]


def _series_corr_too_high(left: pd.Series, right: pd.Series, threshold: float = 0.90) -> bool:
    aligned = pd.concat(
        [
            left.replace([np.inf, -np.inf], np.nan),
            right.replace([np.inf, -np.inf], np.nan),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 10:
        return False
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return pd.notna(corr) and abs(float(corr)) > threshold


def summarize_pool_window(
    *,
    name: str,
    fitness_profile: str,
    train_panel: pd.DataFrame,
    effective_train_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    new_pool: list[SelectedFactor],
    pool_after_validation: list[SelectedFactor],
) -> dict[str, Any]:
    return {
        "name": name,
        "fitness_profile": fitness_profile,
        "train_start": str(pd.to_datetime(train_panel["date"], utc=False).min().date()),
        "train_end": str(pd.to_datetime(train_panel["date"], utc=False).max().date()),
        "effective_train_start": str(pd.to_datetime(effective_train_panel["date"], utc=False).min().date()),
        "effective_train_end": str(pd.to_datetime(effective_train_panel["date"], utc=False).max().date()),
        "validation_start": str(pd.to_datetime(validation_panel["date"], utc=False).min().date()),
        "validation_end": str(pd.to_datetime(validation_panel["date"], utc=False).max().date()),
        "new_candidate_count": int(len(new_pool)),
        "pool_size_after_validation": int(len(pool_after_validation)),
    }


def run_rolling_pool_workflow(
    *,
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    output_dir: Path,
    purge_bars: int,
    embargo_bars: int,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    fitness_profiles = build_window_fitness_profiles(config.fitness)
    rolling_dir = ensure_dir(output_dir / "rolling_pool")
    pool: list[SelectedFactor] = []
    window_rows: list[dict[str, Any]] = []

    total_windows = len(ROLLING_WINDOW_SPECS)
    for window_index, spec in enumerate(ROLLING_WINDOW_SPECS, start=1):
        if progress_callback is not None:
            progress_callback(
                f"rolling | window {window_index}/{total_windows} | "
                f"{spec['name']} | {spec['train_start']} -> {spec['validation_end']}"
            )
        fitness = fitness_profiles[spec["fitness_profile"]]
        window_config = clone_config_with_fitness(config, fitness, save_registry=False)
        train_panel = slice_panel_by_date(panel, spec["train_start"], spec["train_end"])
        validation_panel = slice_panel_by_date(panel, spec["validation_start"], spec["validation_end"])
        effective_train_panel = apply_purge_and_embargo(train_panel, validation_panel, purge_bars, embargo_bars)
        new_pool = build_candidate_factor_pool(
            panel=effective_train_panel,
            config=window_config,
            pool_limit=ROLLING_WINDOW_POOL_LIMIT,
            fast_keep=min(ROLLING_WINDOW_FAST_KEEP, max(window_config.fast_filter_keep, 72)),
            deep_keep=max(ROLLING_WINDOW_DEEP_KEEP, window_config.deep_eval_keep),
            scoring_panel=validation_panel,
            scoring_split="validation",
        )
        combined_pool = dedupe_factor_pool([*pool, *new_pool])
        validated_pool = revalidate_factor_pool(combined_pool, validation_panel, window_config, fitness)
        pool = trim_candidate_pool(validated_pool, ROLLING_POOL_LIMIT)

        window_dir = ensure_dir(rolling_dir / spec["name"])
        pd.DataFrame([factor.summary_row() for factor in new_pool]).to_csv(window_dir / "new_candidates.csv", index=False)
        pd.DataFrame([factor.summary_row() for factor in pool]).to_csv(window_dir / "validated_pool.csv", index=False)
        window_rows.append(
            summarize_pool_window(
                name=spec["name"],
                fitness_profile=spec["fitness_profile"],
                train_panel=train_panel,
                effective_train_panel=effective_train_panel,
                validation_panel=validation_panel,
                new_pool=new_pool,
                pool_after_validation=pool,
            )
        )
        if progress_callback is not None:
            progress_callback(
                f"rolling | completed {spec['name']} | "
                f"new {len(new_pool)} | pool after validation {len(pool)}"
            )

    final_fitness = fitness_profiles["balanced"]
    final_pool = rescore_candidate_pool(pool, final_fitness)
    final_pool = trim_candidate_pool(final_pool, ROLLING_POOL_LIMIT)
    selected = select_factors_from_pool(final_pool, clone_config_with_fitness(config, final_fitness), final_fitness)
    min_required = int(config.portfolio.min_selected_factor_count)
    if len(selected) < min_required:
        raise ValueError(
            f"Final rolling pool selected only {len(selected)} factors; "
            f"expected at least {min_required} from portfolio config."
        )

    pd.DataFrame(window_rows).to_csv(rolling_dir / "rolling_window_summary.csv", index=False)
    pd.DataFrame([factor.summary_row() for factor in final_pool]).to_csv(rolling_dir / "final_candidate_pool.csv", index=False)
    return {
        "candidate_pool": final_pool,
        "selected_factors": selected[: min(len(selected), 30)],
        "rolling_summary": {
            "window_count": int(len(window_rows)),
            "purge_bars": int(purge_bars),
            "embargo_bars": int(embargo_bars),
            "pool_limit": int(ROLLING_POOL_LIMIT),
            "final_pool_size": int(len(final_pool)),
            "final_selected_factor_count": int(len(selected[: min(len(selected), 30)])),
            "windows": window_rows,
        },
    }


def load_fitness_override(path: str | None) -> FitnessConfig | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_keys = set(asdict(FitnessConfig()).keys())
    filtered_payload = {key: value for key, value in payload.items() if key in allowed_keys}
    return FitnessConfig(**filtered_payload)


def merge_fitness_config(base: FitnessConfig, override: FitnessConfig | None) -> FitnessConfig:
    if override is None:
        return base
    merged = asdict(base)
    merged.update(asdict(override))
    return FitnessConfig(**merged)


def write_selected_factors(output_dir: Path, selected: list[Any]) -> None:
    summary = pd.DataFrame([factor.summary_row() for factor in selected])
    summary.to_csv(output_dir / "selected_factors_summary.csv", index=False)
    factors_dir = ensure_dir(output_dir / "factors")
    summary.to_csv(factors_dir / "selected_factors_summary.csv", index=False)


def refine_selected_factors_before_backtest(
    *,
    selected: list[SelectedFactor],
    validation_panel: pd.DataFrame,
    config: Any,
    initial_capital: float,
    regime_source_panel: pd.DataFrame | None,
    min_count: int,
    output_dir: Path,
    progress_callback: Any | None = None,
) -> list[SelectedFactor]:
    if not selected or validation_panel.empty:
        return selected

    decisions: list[dict[str, Any]] = []
    kept: list[SelectedFactor] = []
    dropped: list[tuple[SelectedFactor, dict[str, Any]]] = []

    total_factors = len(selected)
    for factor_index, factor in enumerate(selected, start=1):
        if progress_callback is not None:
            progress_callback(
                f"refine | factor {factor_index}/{total_factors} | checking {factor.expression}"
            )
        long_factor = _clone_selected_factor_with_direction(factor, 1)
        short_factor = _clone_selected_factor_with_direction(factor, -1)
        _, long_metrics, _ = backtest_selected_factors(
            panel=validation_panel,
            config=config,
            selected_factors=[long_factor],
            initial_capital=initial_capital,
            output_dir=None,
            regime_source_panel=regime_source_panel,
        )
        _, short_metrics, _ = backtest_selected_factors(
            panel=validation_panel,
            config=config,
            selected_factors=[short_factor],
            initial_capital=initial_capital,
            output_dir=None,
            regime_source_panel=regime_source_panel,
        )

        preferred_direction = _preferred_direction_from_metrics(long_metrics, short_metrics)
        long_sharpe = float(long_metrics.get("sharpe", 0.0))
        short_sharpe = float(short_metrics.get("sharpe", 0.0))
        long_return = float(long_metrics.get("total_return", 0.0))
        short_return = float(short_metrics.get("total_return", 0.0))
        best_sharpe = max(long_sharpe, short_sharpe)
        best_return = max(long_return, short_return)
        research_rank_ic = float(factor.metrics.get("validation_rank_ic_mean", 0.0))

        should_drop = (
            best_sharpe <= 0.0
            and best_return <= 0.0
            and research_rank_ic <= 0.0
        )
        refined_factor = _clone_selected_factor_with_direction(factor, preferred_direction)
        decision = {
            "expression": factor.expression,
            "original_direction": int(factor.direction),
            "preferred_direction": int(preferred_direction),
            "direction_flip_suggested": bool(int(preferred_direction) != int(factor.direction)),
            "research_fitness": float(factor.fitness),
            "research_validation_rank_ic_mean": research_rank_ic,
            "long_sharpe": long_sharpe,
            "long_total_return": long_return,
            "short_sharpe": short_sharpe,
            "short_total_return": short_return,
            "drop_candidate": bool(should_drop),
            "final_action": "drop" if should_drop else ("flip" if int(preferred_direction) != int(factor.direction) else "keep"),
        }
        decisions.append(decision)
        if should_drop:
            dropped.append((refined_factor, decision))
        else:
            kept.append(refined_factor)

    if len(kept) < min_count:
        dropped.sort(
            key=lambda item: (
                max(float(item[1].get("long_sharpe", 0.0)), float(item[1].get("short_sharpe", 0.0))),
                max(float(item[1].get("long_total_return", 0.0)), float(item[1].get("short_total_return", 0.0))),
                float(item[1].get("research_fitness", 0.0)),
            ),
            reverse=True,
        )
        for revived_factor, decision in dropped:
            if len(kept) >= min_count:
                break
            kept.append(revived_factor)
            decision["final_action"] = "keep_to_meet_min_count"

    decisions_df = pd.DataFrame(decisions)
    if not decisions_df.empty:
        decisions_df = decisions_df.sort_values(
            ["drop_candidate", "direction_flip_suggested", "research_fitness"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    decisions_df.to_csv(output_dir / "factor_refinement_summary.csv", index=False)
    if progress_callback is not None:
        flip_count = int(decisions_df["direction_flip_suggested"].sum()) if not decisions_df.empty else 0
        drop_count = int((decisions_df["final_action"] == "drop").sum()) if not decisions_df.empty else 0
        progress_callback(f"refine | complete | flips {flip_count} | dropped {drop_count} | kept {len(kept)}")
    return kept


def _clone_selected_factor_with_direction(factor: Any, direction: int) -> SelectedFactor:
    metrics = dict(getattr(factor, "metrics", {}))
    metrics["direction_check_direction"] = int(direction)
    return SelectedFactor(
        expression=factor.expression,
        node=factor.node,
        direction=int(direction),
        fitness=float(getattr(factor, "fitness", 0.0)),
        metrics=metrics,
        complexity=int(getattr(factor, "complexity", 0)),
        finite_ratio=float(getattr(factor, "finite_ratio", 1.0)),
        values=getattr(factor, "values", None),
    )


def _preferred_direction_from_metrics(long_metrics: dict[str, Any], short_metrics: dict[str, Any]) -> int:
    long_sharpe = float(long_metrics.get("sharpe", 0.0))
    short_sharpe = float(short_metrics.get("sharpe", 0.0))
    if short_sharpe > long_sharpe + 1e-12:
        return -1
    if long_sharpe > short_sharpe + 1e-12:
        return 1

    long_return = float(long_metrics.get("total_return", 0.0))
    short_return = float(short_metrics.get("total_return", 0.0))
    if short_return > long_return + 1e-12:
        return -1
    return 1


def summarize_market_baseline(
    benchmark_df: pd.DataFrame,
    date_values: pd.Series | list[Any],
    initial_capital: float,
) -> dict[str, float]:
    benchmark = benchmark_df.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=False)
    aligned_dates = pd.to_datetime(pd.Series(date_values), utc=False).dropna().unique()
    benchmark = benchmark.loc[benchmark["date"].isin(aligned_dates)].copy()
    benchmark = benchmark.sort_values("date", kind="mergesort").reset_index(drop=True)
    if benchmark.empty:
        return {"final_equity": float(initial_capital), "total_return": 0.0}
    if "return" in benchmark.columns:
        daily_return = pd.to_numeric(benchmark["return"], errors="coerce").fillna(0.0)
    elif {"open", "close"}.issubset(benchmark.columns):
        daily_return = (pd.to_numeric(benchmark["close"], errors="coerce") / pd.to_numeric(benchmark["open"], errors="coerce")) - 1.0
        daily_return = daily_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        daily_return = pd.to_numeric(benchmark["close"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
    equity = float(initial_capital) * (1.0 + daily_return).cumprod()
    final_equity = float(equity.iloc[-1])
    return {
        "final_equity": final_equity,
        "total_return": (final_equity / float(initial_capital)) - 1.0,
    }


def build_backtest_report(
    *,
    initial_capital: float,
    result: pd.DataFrame,
    metrics: dict[str, Any],
    backtest_btc: dict[str, float],
    backtest_equal_weight: dict[str, float],
    backtest_market_cap: dict[str, float],
    walk_forward_summary: dict[str, Any],
    selected_factor_expressions: list[str],
    split_summary: dict[str, Any],
    pbo_summary: dict[str, Any] | None = None,
) -> str:
    trajectory = summarize_backtest_trajectory(result)
    strategy_total_return = float(metrics.get("total_return", 0.0))
    backtest_window_summary = split_summary.get("backtest", {})
    if backtest_window_summary.get("date_min", "n/a") == "n/a":
        backtest_window_summary = result_window_summary(result)
    rolling_lines: list[str]
    window_count = int(walk_forward_summary.get("window_count", 0))
    if window_count > 0:
        rolling_lines = [
            "## Rolling Validation Summary",
            "",
            f"- Window count: {window_count}",
            f"- Purge bars: {int(walk_forward_summary.get('purge_bars', 0))}",
            f"- Embargo bars: {int(walk_forward_summary.get('embargo_bars', 0))}",
            f"- Final pool size before selection: {int(walk_forward_summary.get('final_pool_size', 0))}",
            f"- Final selected factor count: {int(walk_forward_summary.get('final_selected_factor_count', len(selected_factor_expressions)))}",
        ]
        for window in walk_forward_summary.get("windows", []):
            rolling_lines.append(
                f"- {window.get('name', 'window')}: "
                f"train {window.get('train_start', 'n/a')} to {window.get('train_end', 'n/a')}, "
                f"val {window.get('validation_start', 'n/a')} to {window.get('validation_end', 'n/a')}, "
                f"profile {window.get('fitness_profile', 'n/a')}, "
                f"new candidates {int(window.get('new_candidate_count', 0))}, "
                f"pool after validation {int(window.get('pool_size_after_validation', 0))}"
            )
        rolling_lines.append("")
    else:
        rolling_lines = [
            "## Walk-Forward Stability",
            "",
            f"- Fold count: {int(walk_forward_summary.get('fold_count', 0))}",
            f"- Mean fold total return: {format_pct(walk_forward_summary.get('mean_total_return', 0.0))}",
            f"- Median fold total return: {format_pct(walk_forward_summary.get('median_total_return', 0.0))}",
            f"- Mean fold Sharpe: {format_num(walk_forward_summary.get('mean_sharpe', 0.0))}",
            f"- Median fold Sharpe: {format_num(walk_forward_summary.get('median_sharpe', 0.0))}",
            f"- Mean fold max drawdown: {format_pct(walk_forward_summary.get('mean_max_drawdown', 0.0))}",
            f"- Mean fold turnover: {format_num(walk_forward_summary.get('mean_turnover', 0.0))}",
            "",
        ]
    walk_forward_windows = split_summary.get("walk_forward_windows", [])
    lines = [
        "# Crypto Backtest Report",
        "",
        "## Executive Summary",
        "",
        build_backtest_takeaway(metrics, backtest_btc, backtest_equal_weight, backtest_market_cap),
        "",
        "## Evaluation Windows",
        "",
        f"- Research window: {format_window(split_summary.get('research', {}))}",
        f"- Train window: {format_window(split_summary.get('train', {}))}",
        f"- Validation window: {format_window(split_summary.get('validation', {}))}",
        f"- Final backtest window: {format_window(backtest_window_summary)}",
        "",
        "## Walk-Forward Windows",
        "",
        "## Core Metrics",
        "",
        f"- Initial capital: {initial_capital:,.2f}",
        f"- Final equity: {float(metrics.get('pnl', 0.0)) + initial_capital:,.2f}",
        f"- Total return: {format_pct(strategy_total_return)}",
        f"- Annual return: {format_pct(metrics.get('annual_return', 0.0))}",
        f"- Sharpe: {format_num(metrics.get('sharpe', 0.0))}",
        f"- Max drawdown: {format_pct(metrics.get('max_drawdown', 0.0))}",
        f"- Mean turnover: {format_num(metrics.get('turnover', 0.0))}",
        "",
        "## Path Diagnostics",
        "",
        f"- Trading days in backtest: {trajectory['days']}",
        f"- Positive-return days: {trajectory['positive_days']} / {trajectory['days']}",
        f"- Best day: {trajectory['best_day_date']} ({format_pct(trajectory['best_day_return'])})",
        f"- Worst day: {trajectory['worst_day_date']} ({format_pct(trajectory['worst_day_return'])})",
        f"- Final observed drawdown: {format_pct(trajectory['final_drawdown'])}",
        "",
        "## Benchmark Comparison",
        "",
        f"- Strategy total return: {format_pct(strategy_total_return)}",
        f"- BTC total return: {format_pct(backtest_btc.get('total_return', 0.0))} | excess: {format_pct(strategy_total_return - float(backtest_btc.get('total_return', 0.0)))}",
        f"- Equal-weight basket total return: {format_pct(backtest_equal_weight.get('total_return', 0.0))} | excess: {format_pct(strategy_total_return - float(backtest_equal_weight.get('total_return', 0.0)))}",
        f"- Market-cap basket total return: {format_pct(backtest_market_cap.get('total_return', 0.0))} | excess: {format_pct(strategy_total_return - float(backtest_market_cap.get('total_return', 0.0)))}",
        "",
    ]
    insert_at = lines.index("## Core Metrics")
    walk_forward_lines = []
    if walk_forward_windows:
        for window in walk_forward_windows:
            walk_forward_lines.append(
                f"- {window.get('name', 'window')}: "
                f"train {format_window(window.get('train', {}))}; "
                f"validation {format_window(window.get('validation', {}))}"
            )
    else:
        walk_forward_lines.append("- n/a")
    lines[insert_at:insert_at] = walk_forward_lines + [""]
    lines.extend(rolling_lines)
    lines.extend(
        [
            "## Selected Factors",
            "",
            f"- Factor count: {len(selected_factor_expressions)}",
        ]
    )
    for expression in selected_factor_expressions:
        lines.append(f"- `{expression}`")
    if pbo_summary is not None:
        lines.extend(
            [
                "",
                "## PBO Snapshot",
                "",
                f"- Strategy count in PBO run: {int(pbo_summary.get('n_strategies', 0))}",
                f"- Group count: {int(pbo_summary.get('n_groups', 0))}",
                f"- Split count: {int(pbo_summary.get('n_splits', 0))}",
                f"- Estimated PBO: {format_pct(pbo_summary.get('pbo', 0.0))}",
                f"- Mean OOS percentile: {format_pct(pbo_summary.get('mean_oos_percentile', 0.0))}",
                "",
                "PBO here should be read as a basic overfitting warning signal, not yet a full purged-CPCV implementation.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_backtest_takeaway(
    metrics: dict[str, Any],
    backtest_btc: dict[str, float],
    backtest_equal_weight: dict[str, float],
    backtest_market_cap: dict[str, float],
) -> str:
    strategy_total_return = float(metrics.get("total_return", 0.0))
    comparisons = {
        "BTC": strategy_total_return - float(backtest_btc.get("total_return", 0.0)),
        "equal-weight basket": strategy_total_return - float(backtest_equal_weight.get("total_return", 0.0)),
        "market-cap basket": strategy_total_return - float(backtest_market_cap.get("total_return", 0.0)),
    }
    strongest_name = max(comparisons, key=comparisons.get)
    strongest_value = comparisons[strongest_name]
    if strategy_total_return > 0:
        direction = "made money in the backtest window"
    elif strategy_total_return < 0:
        direction = "lost money in the backtest window"
    else:
        direction = "finished roughly flat in the backtest window"
    return (
        f"This strategy {direction}, delivering {format_pct(strategy_total_return)} total return with "
        f"{format_num(metrics.get('sharpe', 0.0))} Sharpe and {format_pct(metrics.get('max_drawdown', 0.0))} max drawdown. "
        f"Its strongest relative comparison was versus {strongest_name}, where excess return was {format_pct(strongest_value)}."
    )


def build_workflow_report_index(
    *,
    workflow_summary: dict[str, Any],
    backtest_report_path: Path,
    paper_report_path: Path | None,
) -> str:
    backtest_metrics = workflow_summary.get("backtest_metrics", {})
    lines = [
        "# Workflow Report Index",
        "",
        "## Summary",
        "",
        f"- Selected factors: {int(workflow_summary.get('selected_factor_count', 0))}",
        f"- Backtest total return: {format_pct(backtest_metrics.get('total_return', 0.0))}",
        f"- Backtest Sharpe: {format_num(backtest_metrics.get('sharpe', 0.0))}",
        "",
        "## Report Files",
        "",
        f"- Backtest report: `{backtest_report_path}`",
    ]
    if paper_report_path is not None:
        lines.append(f"- Paper report: `{paper_report_path}`")
    lines.extend(
        [
            f"- Workflow summary JSON: `workflow_summary.json`",
            f"- Backtest metrics JSON: `backtest/backtest_metrics.json`",
        ]
    )
    if paper_report_path is not None:
        lines.append(f"- Paper summary JSON: `paper/paper_summary.json`")
    lines.append("")
    return "\n".join(lines)


def summarize_backtest_trajectory(result: pd.DataFrame) -> dict[str, Any]:
    if result.empty:
        return {
            "days": 0,
            "positive_days": 0,
            "best_day_date": "n/a",
            "best_day_return": 0.0,
            "worst_day_date": "n/a",
            "worst_day_return": 0.0,
            "final_drawdown": 0.0,
        }
    ordered = result.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values("date", kind="mergesort").reset_index(drop=True)
    net_return = pd.to_numeric(ordered.get("net_return"), errors="coerce").fillna(0.0)
    drawdown = pd.to_numeric(ordered.get("drawdown"), errors="coerce").fillna(0.0)
    best_idx = int(net_return.idxmax())
    worst_idx = int(net_return.idxmin())
    return {
        "days": int(len(ordered)),
        "positive_days": int((net_return > 0).sum()),
        "best_day_date": str(ordered.loc[best_idx, "date"].date()),
        "best_day_return": float(net_return.loc[best_idx]),
        "worst_day_date": str(ordered.loc[worst_idx, "date"].date()),
        "worst_day_return": float(net_return.loc[worst_idx]),
        "final_drawdown": float(drawdown.iloc[-1]) if not drawdown.empty else 0.0,
    }


def workflow_split_summary(
    *,
    research_panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    backtest_panel: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "research": panel_window_summary(research_panel),
        "train": panel_window_summary(train_panel),
        "validation": panel_window_summary(validation_panel),
        "backtest": panel_window_summary(backtest_panel),
        "walk_forward_windows": [
            {
                "name": str(spec["name"]),
                "train": {"date_min": str(spec["train_start"]), "date_max": str(spec["train_end"])},
                "validation": {"date_min": str(spec["validation_start"]), "date_max": str(spec["validation_end"])},
            }
            for spec in ROLLING_WINDOW_SPECS
        ],
    }


def panel_window_summary(panel: pd.DataFrame) -> dict[str, str]:
    if panel.empty:
        return {"date_min": "n/a", "date_max": "n/a"}
    dates = pd.to_datetime(panel["date"], utc=False)
    return {
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
    }


def result_window_summary(result: pd.DataFrame) -> dict[str, str]:
    if result.empty or "date" not in result.columns:
        return {"date_min": "n/a", "date_max": "n/a"}
    dates = pd.to_datetime(result["date"], utc=False)
    return {
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
    }


def format_window(summary: dict[str, str]) -> str:
    return f"{summary.get('date_min', 'n/a')} to {summary.get('date_max', 'n/a')}"


def format_pct(value: Any) -> str:
    return f"{float(value):.2%}"


def format_num(value: Any) -> str:
    return f"{float(value):.3f}"


def write_text(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
