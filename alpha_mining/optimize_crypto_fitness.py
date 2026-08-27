from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any
import pickle
import warnings

import numpy as np
import pandas as pd

from utils.random_state import set_global_seed
from utils.experiment_ledger import ExperimentLedger
from utils.fitness_trial_ledger import FitnessTrialLedger

from alpha_mining import (
    FitnessConfig,
    backtest_selected_factors,
    build_candidate_factor_pool,
    run_walk_forward_evaluation,
    select_factors_from_pool,
)
from alpha_mining.pbo import compute_pbo, save_pbo_outputs
from alpha_mining.regime import build_regime_frame, filter_regime_frame_by_dates, summarize_regime_frame
from alpha_mining.registry import FactorRegistry
from alpha_mining.run_crypto_workflow import (
    CRYPTO_UNIVERSE_PRESETS,
    FINAL_BACKTEST_END,
    FINAL_BACKTEST_START,
    ROLLING_WINDOW_SPECS,
    build_backtest_report,
    build_workflow_report_index,
    build_crypto_workflow_config,
    build_equal_weight_benchmark,
    build_market_cap_benchmark,
    load_fitness_override,
    load_or_build_btc_benchmark,
    load_or_build_market_cap_benchmark,
    load_or_build_crypto_panel,
    load_crypto_universe_from_args,
    merge_fitness_config,
    summarize_market_baseline,
    to_jsonable,
    workflow_split_summary,
    slice_panel_by_date,
    write_text,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("reports") / "crypto_fitness_optimization"
DEFAULT_FINAL_OUTPUT_DIR = Path("reports") / "crypto_alpha_workflow_optimized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize crypto FitnessConfig on validation performance, then run final workflow.")
    parser.add_argument("--panel-csv", default=None, help="Combined crypto panel CSV.")
    parser.add_argument("--panel-dir", default=None, help="Directory with one CSV per crypto symbol.")
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with symbol,name.")
    parser.add_argument("--universe-preset", default="crypto30", choices=sorted(CRYPTO_UNIVERSE_PRESETS.keys()), help="Named crypto universe preset.")
    parser.add_argument("--btc-benchmark-csv", default=None, help="Optional BTC benchmark CSV.")
    parser.add_argument("--market-cap-benchmark-csv", default=None, help="Optional market-cap benchmark CSV.")
    parser.add_argument("--benchmark-name", default="BTC", help="Readable BTC benchmark name.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write optimization artifacts.")
    parser.add_argument("--final-output-dir", default=str(DEFAULT_FINAL_OUTPUT_DIR), help="Directory to write final workflow outputs.")
    parser.add_argument("--trials", type=int, default=16, help="Number of random fitness configurations to try.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fitness search.")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital for research backtests.")
    parser.add_argument("--fitness-config", default=None, help="Optional JSON file with base FitnessConfig overrides.")
    parser.add_argument("--candidate-pool-pkl", default=None, help="Optional prebuilt candidate factor pool pickle.")
    parser.add_argument("--reuse-candidate-pool", action="store_true", help="Reuse candidate_factor_pool.pkl under output-dir when available.")
    parser.add_argument("--skip-final-run", action="store_true", help="Only search fitness configs and skip the final full workflow.")
    parser.add_argument("--quick", action="store_true", help="Use lighter GP/pool settings for fast smoke runs.")
    return parser.parse_args()


def _main(lifecycle: dict[str, Any]) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")
    args = parse_args()
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    final_output_dir = Path(args.final_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    panel = load_or_build_crypto_panel(args)
    universe = load_crypto_universe_from_args(args)
    btc_benchmark_df = load_or_build_btc_benchmark(args, panel)
    equal_weight_benchmark_df = build_equal_weight_benchmark(panel, benchmark_name="crypto30_equal_weight")
    market_cap_benchmark_df = load_or_build_market_cap_benchmark(args, panel, universe)
    research_panel = slice_panel_by_date(
        panel,
        start=ROLLING_WINDOW_SPECS[0]["train_start"],
        end=ROLLING_WINDOW_SPECS[-1]["validation_end"],
    )
    final_backtest_panel = slice_panel_by_date(panel, start=FINAL_BACKTEST_START, end=FINAL_BACKTEST_END)
    train_panel = pd.concat(
        [slice_panel_by_date(research_panel, start=spec["train_start"], end=spec["train_end"]) for spec in ROLLING_WINDOW_SPECS],
        ignore_index=True,
    ).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)
    validation_panel = pd.concat(
        [slice_panel_by_date(research_panel, start=spec["validation_start"], end=spec["validation_end"]) for spec in ROLLING_WINDOW_SPECS],
        ignore_index=True,
    ).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)
    if research_panel.empty or train_panel.empty or validation_panel.empty or final_backtest_panel.empty:
        raise ValueError("Resolved optimization windows produced an empty research, train, validation, or final backtest panel.")

    fitness_override = load_fitness_override(args.fitness_config)
    baseline_config = build_crypto_workflow_config(
        output_dir=output_dir / "baseline_registry",
        symbols=sorted(panel["symbol"].astype(str).unique()),
        panel=panel,
        fitness_override=fitness_override,
        quick=args.quick,
    )
    candidate_pool = load_or_build_candidate_pool(
        output_dir=output_dir,
        candidate_pool_pkl=args.candidate_pool_pkl,
        reuse_candidate_pool=args.reuse_candidate_pool,
        train_panel=train_panel,
        validation_panel=validation_panel,
        config=baseline_config,
    )
    save_candidate_pool_artifacts(output_dir, candidate_pool)
    candidate_pool_identity = build_candidate_pool_identity(output_dir, candidate_pool)
    experiment_ledger = ExperimentLedger(output_dir.parent)
    parent_started = experiment_ledger.start(
        configuration={
            "workflow": "crypto_fitness_optimization",
            "seed": args.seed,
            "trial_count": args.trials,
            "quick": args.quick,
            "base_fitness": asdict(baseline_config.fitness),
            "candidate_pool_identity": candidate_pool_identity,
            "research_windows": ROLLING_WINDOW_SPECS,
        },
        metadata={
            "entrypoint": "alpha_mining.optimize_crypto_fitness",
            "output_directory": str(output_dir.resolve()),
            "candidate_pool": str((output_dir / "candidate_factor_pool.pkl").resolve()),
        },
    )
    trial_ledger = FitnessTrialLedger(output_dir.parent)
    optimization_started = trial_ledger.start_run(
        parent_experiment_id=parent_started["experiment_id"],
        configuration={
            "seed": args.seed,
            "trial_count": args.trials,
            "base_fitness": asdict(baseline_config.fitness),
            "candidate_pool_identity": candidate_pool_identity,
        },
        artifact_locations={
            "output_directory": str(output_dir.resolve()),
            "trial_ledger": str(trial_ledger.trials_path.resolve()),
            "results": str((output_dir / "fitness_search_results.csv").resolve()),
        },
    )
    lifecycle.update({
        "experiment_ledger": experiment_ledger,
        "parent_started": parent_started,
        "trial_ledger": trial_ledger,
        "optimization_started": optimization_started,
    })
    write_json(output_dir / "optimization_run.json", to_jsonable(optimization_started))

    candidates = [baseline_config.fitness]
    for _ in range(max(args.trials - 1, 0)):
        candidates.append(sample_fitness_config(rng, baseline_config.fitness))

    results: list[dict[str, Any]] = []
    for trial_index, fitness_cfg in enumerate(candidates, start=1):
        print_progress("fitness-search", trial_index, len(candidates), f"trial {trial_index}")
        effective_fitness = merge_fitness_config(baseline_config.fitness, fitness_cfg)
        try:
            config = build_crypto_workflow_config(
                output_dir=output_dir / f"trial_{trial_index:03d}" / "registry",
                symbols=sorted(panel["symbol"].astype(str).unique()),
                panel=panel,
                fitness_override=effective_fitness,
                quick=args.quick,
            )
            selected = select_factors_from_pool(candidate_pool, config, config.fitness)
            result, metrics, _ = backtest_selected_factors(
                panel=validation_panel,
                config=config,
                selected_factors=selected,
                initial_capital=args.initial_capital,
                output_dir=None,
                regime_source_panel=research_panel,
            )
            eq = summarize_market_baseline(equal_weight_benchmark_df, result["date"], args.initial_capital)
            summary = score_trial(selected, metrics, eq, config.portfolio.min_selected_factor_count)
            row = {"trial": trial_index, "status": "ok", **summary, **asdict(config.fitness)}
            trial_record = trial_ledger.record_trial(
                optimization_run_id=optimization_started["optimization_run_id"],
                parent_experiment_id=parent_started["experiment_id"],
                trial_index=trial_index,
                fitness_configuration=asdict(effective_fitness),
                candidate_pool_identity=candidate_pool_identity,
                seed=args.seed,
                status="completed",
                objective_value=float(summary["objective"]),
                validation_metrics=to_jsonable(metrics),
                selected_factor_count=len(selected),
            )
        except Exception as exc:  # pragma: no cover
            row = {"trial": trial_index, "status": "error", "objective": -1_000_000_000.0, "error": str(exc), **asdict(effective_fitness)}
            trial_record = trial_ledger.record_trial(
                optimization_run_id=optimization_started["optimization_run_id"],
                parent_experiment_id=parent_started["experiment_id"],
                trial_index=trial_index,
                fitness_configuration=asdict(effective_fitness),
                candidate_pool_identity=candidate_pool_identity,
                seed=args.seed,
                status="failed",
                objective_value=-1_000_000_000.0,
                validation_metrics={},
                selected_factor_count=0,
                failure_reason=str(exc),
            )
        row["trial_id"] = trial_record["trial_id"]
        row["optimization_run_id"] = optimization_started["optimization_run_id"]
        row["parent_experiment_id"] = parent_started["experiment_id"]
        results.append(row)
        pd.DataFrame(results).to_csv(output_dir / "fitness_search_results_partial.csv", index=False)

    results_df = pd.DataFrame(results).sort_values(["objective", "trial"], ascending=[False, True])
    results_df.to_csv(output_dir / "fitness_search_results.csv", index=False)
    best = results_df.iloc[0].to_dict()
    best_fitness = FitnessConfig(**{field: float(best[field]) for field in asdict(baseline_config.fitness).keys()})
    write_json(output_dir / "best_fitness_config.json", to_jsonable(asdict(best_fitness)))
    write_json(output_dir / "best_trial_summary.json", to_jsonable(best))
    durable_trials = trial_ledger.trials_for_run(optimization_started["optimization_run_id"])
    optimization_summary = build_optimization_summary(
        optimization_started=optimization_started,
        trials=durable_trials,
        winning_trial_id=str(best["trial_id"]),
        winning_configuration=asdict(best_fitness),
        artifact_locations={
            "output_directory": str(output_dir.resolve()),
            "results": str((output_dir / "fitness_search_results.csv").resolve()),
            "trials": str(trial_ledger.trials_path.resolve()),
            "best_fitness": str((output_dir / "best_fitness_config.json").resolve()),
        },
    )
    write_json(output_dir / "optimization_summary.json", to_jsonable(optimization_summary))
    trial_ledger.complete_run(optimization_started, summary=optimization_summary)

    strategy_returns: dict[str, pd.DataFrame] = {}
    ok_trials = results_df.loc[results_df["status"] == "ok"].head(min(12, len(results_df))).copy()
    for idx, (_, row) in enumerate(ok_trials.iterrows(), start=1):
        print_progress("pbo-sample", idx, len(ok_trials), f"trial {int(row['trial'])}")
        trial = int(row["trial"])
        trial_fitness = FitnessConfig(**{field: float(row[field]) for field in asdict(baseline_config.fitness).keys()})
        config = build_crypto_workflow_config(
            output_dir=output_dir / f"trial_{trial:03d}" / "registry",
            symbols=sorted(panel["symbol"].astype(str).unique()),
            panel=panel,
            fitness_override=trial_fitness,
            quick=args.quick,
        )
        selected = select_factors_from_pool(candidate_pool, config, config.fitness)
        result, _, _ = backtest_selected_factors(
            panel=research_panel,
            config=config,
            selected_factors=selected,
            initial_capital=args.initial_capital,
            output_dir=None,
            regime_source_panel=research_panel,
        )
        if not result.empty:
            strategy_returns[f"trial_{trial:03d}"] = result[["date", "net_return"]].copy()
    pbo_summary, pbo_group_scores, pbo_split_details = compute_pbo(
        strategy_returns,
        n_groups=6,
        test_group_count=2,
        embargo_groups=1,
    )
    save_pbo_outputs(output_dir / "pbo", pbo_summary, pbo_group_scores, pbo_split_details)
    print("Crypto fitness search complete.")
    print(f"Best objective: {best['objective']:.6f}")

    if args.skip_final_run:
        experiment_ledger.complete(parent_started, results={"optimization_summary": str((output_dir / "optimization_summary.json").resolve()), "winning_trial_id": best["trial_id"]})
        return

    print_progress("final-run", 1, 3, "writing inputs")
    final_output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(final_output_dir / "panel.csv", index=False)
    btc_benchmark_df.to_csv(final_output_dir / "btc_benchmark.csv", index=False)
    equal_weight_benchmark_df.to_csv(final_output_dir / "equal_weight_benchmark.csv", index=False)
    market_cap_benchmark_df.to_csv(final_output_dir / "market_cap_benchmark.csv", index=False)
    research_panel.to_csv(final_output_dir / "research_panel.csv", index=False)
    train_panel.to_csv(final_output_dir / "train_panel.csv", index=False)
    validation_panel.to_csv(final_output_dir / "validation_panel.csv", index=False)
    final_backtest_panel.to_csv(final_output_dir / "backtest_panel.csv", index=False)

    final_config = build_crypto_workflow_config(
        output_dir=final_output_dir,
        symbols=sorted(panel["symbol"].astype(str).unique()),
        panel=panel,
        fitness_override=best_fitness,
        quick=args.quick,
    )
    print_progress("final-run", 2, 3, "building backtest artifacts")
    full_regime_frame = build_regime_frame(panel, final_config.regime)
    validation_regime_frame = filter_regime_frame_by_dates(full_regime_frame, validation_panel["date"])
    backtest_regime_frame = filter_regime_frame_by_dates(full_regime_frame, final_backtest_panel["date"])
    regime_dir = final_output_dir / "regime"
    regime_dir.mkdir(parents=True, exist_ok=True)
    full_regime_frame.to_csv(regime_dir / "full_regime_history.csv", index=False)
    validation_regime_frame.to_csv(regime_dir / "validation_regime_history.csv", index=False)
    backtest_regime_frame.to_csv(regime_dir / "backtest_regime_history.csv", index=False)
    regime_summary = {
        "full": summarize_regime_frame(full_regime_frame),
        "validation": summarize_regime_frame(validation_regime_frame),
        "backtest": summarize_regime_frame(backtest_regime_frame),
    }
    write_json(regime_dir / "regime_summary.json", to_jsonable(regime_summary))
    selected = select_factors_from_pool(candidate_pool, final_config, final_config.fitness)
    FactorRegistry(final_config.registry_dir()).save(selected, final_config, research_panel)
    backtest_regime_source_panel = pd.concat(
        [research_panel, final_backtest_panel],
        ignore_index=True,
    ).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)
    result, metrics, _ = backtest_selected_factors(
        panel=final_backtest_panel,
        config=final_config,
        selected_factors=selected,
        initial_capital=args.initial_capital,
        output_dir=str(final_output_dir / "backtest"),
        regime_source_panel=backtest_regime_source_panel,
    )
    walk_forward_folds, walk_forward_summary, _ = run_walk_forward_evaluation(
        panel=research_panel,
        config=final_config,
        initial_capital=args.initial_capital,
        output_dir=final_output_dir / "walk_forward",
    )
    write_json(final_output_dir / "backtest" / "backtest_metrics.json", to_jsonable(metrics))
    write_text(
        final_output_dir / "backtest" / "backtest_report.md",
        build_backtest_report(
            initial_capital=args.initial_capital,
            result=result,
            metrics=metrics,
            backtest_btc=summarize_market_baseline(btc_benchmark_df, result["date"], args.initial_capital),
            backtest_equal_weight=summarize_market_baseline(equal_weight_benchmark_df, result["date"], args.initial_capital),
            backtest_market_cap=summarize_market_baseline(market_cap_benchmark_df, result["date"], args.initial_capital),
            walk_forward_summary=walk_forward_summary,
            selected_factor_expressions=[factor.expression for factor in selected],
            split_summary=workflow_split_summary(
                research_panel=research_panel,
                train_panel=train_panel,
                validation_panel=validation_panel,
                backtest_panel=final_backtest_panel,
            ),
            pbo_summary=pbo_summary,
        ),
    )
    backtest_report_path = final_output_dir / "backtest" / "backtest_report.md"
    write_json(final_output_dir / "optimized_fitness_config.json", to_jsonable(asdict(best_fitness)))
    workflow_summary = {
        "market": "crypto",
        "split": {
            "research": {
                "date_min": str(pd.to_datetime(research_panel["date"]).min().date()),
                "date_max": str(pd.to_datetime(research_panel["date"]).max().date()),
            },
            "train": {
                "date_min": str(pd.to_datetime(train_panel["date"]).min().date()),
                "date_max": str(pd.to_datetime(train_panel["date"]).max().date()),
            },
            "validation": {
                "date_min": str(pd.to_datetime(validation_panel["date"]).min().date()),
                "date_max": str(pd.to_datetime(validation_panel["date"]).max().date()),
            },
            "backtest": {
                "date_min": str(pd.to_datetime(final_backtest_panel["date"]).min().date()),
                "date_max": str(pd.to_datetime(final_backtest_panel["date"]).max().date()),
            },
        },
        "research_backtest_target": "validation_only_then_final_backtest",
        "selected_factor_count": len(selected),
        "selected_factor_expressions": [factor.expression for factor in selected],
        "regime_summary": regime_summary,
        "walk_forward_summary": walk_forward_summary,
        "walk_forward_fold_count": int(len(walk_forward_folds)),
        "pbo_summary": pbo_summary,
        "backtest_metrics": metrics,
        "backtest_vs_btc": summarize_market_baseline(btc_benchmark_df, result["date"], args.initial_capital),
        "backtest_vs_equal_weight": summarize_market_baseline(equal_weight_benchmark_df, result["date"], args.initial_capital),
        "backtest_vs_market_cap": summarize_market_baseline(market_cap_benchmark_df, result["date"], args.initial_capital),
    }
    write_json(final_output_dir / "workflow_summary.json", to_jsonable(workflow_summary))
    workflow_report_path = final_output_dir / "workflow_report.md"
    write_text(
        workflow_report_path,
        build_workflow_report_index(
            workflow_summary=workflow_summary,
            backtest_report_path=backtest_report_path,
            paper_report_path=None,
        ),
    )
    print(f"Workflow report: {workflow_report_path.resolve()}")
    print(f"Backtest report: {backtest_report_path.resolve()}")
    print_progress("final-run", 3, 3, "completed")
    experiment_ledger.complete(parent_started, results={"optimization_summary": str((output_dir / "optimization_summary.json").resolve()), "winning_trial_id": best["trial_id"], "workflow_report": str(workflow_report_path.resolve())})


def main() -> None:
    """Wrap the optimizer so a normal exception path closes durable ledgers."""
    lifecycle: dict[str, Any] = {}
    try:
        _main(lifecycle)
    except BaseException as error:
        trial_ledger = lifecycle.get("trial_ledger")
        optimization_started = lifecycle.get("optimization_started")
        if trial_ledger is not None and optimization_started is not None:
            trial_ledger.fail_run(optimization_started, error)
        experiment_ledger = lifecycle.get("experiment_ledger")
        parent_started = lifecycle.get("parent_started")
        if experiment_ledger is not None and parent_started is not None:
            experiment_ledger.fail(parent_started, error)
        raise


def load_or_build_candidate_pool(
    *,
    output_dir: Path,
    candidate_pool_pkl: str | None,
    reuse_candidate_pool: bool,
    train_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    config: Any,
) -> list[Any]:
    explicit_path = Path(candidate_pool_pkl) if candidate_pool_pkl else None
    cached_path = output_dir / "candidate_factor_pool.pkl"
    load_path = explicit_path or (cached_path if reuse_candidate_pool and cached_path.exists() else None)
    if load_path is not None and load_path.exists():
        with load_path.open("rb") as handle:
            return pickle.load(handle)

    return build_candidate_factor_pool(
        panel=train_panel,
        config=config,
        scoring_panel=validation_panel,
        scoring_split="validation",
    )


def save_candidate_pool_artifacts(output_dir: Path, candidate_pool: list[Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate_factor_pool.pkl").open("wb") as handle:
        pickle.dump(candidate_pool, handle)
    pd.DataFrame([factor.summary_row() for factor in candidate_pool]).to_csv(
        output_dir / "candidate_factor_pool.csv",
        index=False,
    )


def build_candidate_pool_identity(output_dir: Path, candidate_pool: list[Any]) -> dict[str, Any]:
    path = output_dir / "candidate_factor_pool.pkl"
    expressions = [str(factor.expression) for factor in candidate_pool]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "factor_count": len(candidate_pool),
        "expression_sha256": hashlib.sha256("\n".join(expressions).encode("utf-8")).hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_optimization_summary(
    *,
    optimization_started: dict[str, Any],
    trials: list[dict[str, Any]],
    winning_trial_id: str,
    winning_configuration: dict[str, Any],
    artifact_locations: dict[str, Any],
) -> dict[str, Any]:
    trial_ids = {str(trial["trial_id"]) for trial in trials}
    if winning_trial_id not in trial_ids:
        raise ValueError("Winning trial is missing from the durable trial ledger.")
    return {
        "optimization_run_id": optimization_started["optimization_run_id"],
        "parent_experiment_id": optimization_started["parent_experiment_id"],
        "total_attempted_trials": len(trials),
        "completed_trials": sum(trial["status"] == "completed" for trial in trials),
        "failed_trials": sum(trial["status"] == "failed" for trial in trials),
        "winning_trial_id": winning_trial_id,
        "winning_configuration": winning_configuration,
        "artifact_locations": artifact_locations,
    }


def score_trial(
    selected: list[Any],
    metrics: dict[str, Any],
    equal_weight_baseline: dict[str, float],
    min_selected_factor_count: int,
) -> dict[str, float]:
    strategy_excess = float(metrics.get("total_return", 0.0)) - float(equal_weight_baseline.get("total_return", 0.0))
    objective = (
        (8.0 * strategy_excess)
        + (1.5 * float(metrics.get("sharpe", 0.0)))
        - (2.0 * float(metrics.get("turnover", 0.0)))
        - (8.0 * abs(float(metrics.get("max_drawdown", 0.0))))
        + (2.5 * float(metrics.get("cumulative_return", 0.0)))
    )
    if len(selected) < min_selected_factor_count:
        objective -= 5.0 * float(min_selected_factor_count - len(selected))
    return {
        "objective": float(objective),
        "validation_total_return": float(metrics.get("total_return", 0.0)),
        "validation_sharpe": float(metrics.get("sharpe", 0.0)),
        "validation_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "validation_turnover": float(metrics.get("turnover", 0.0)),
        "validation_excess_vs_equal_weight": float(strategy_excess),
        "selected_factor_count": float(len(selected)),
    }


def sample_fitness_config(rng: np.random.Generator, base: FitnessConfig) -> FitnessConfig:
    return FitnessConfig(
        fast_rank_ic_weight=float(rng.uniform(4.0, 8.0)),
        validation_ic_weight=float(rng.uniform(10.0, 16.0)),
        sharpe_weight=float(rng.uniform(1.0, 1.6)),
        cumulative_return_weight=float(rng.uniform(4.0, 7.0)),
        excess_return_weight=float(rng.uniform(6.0, 10.0)),
        stability_weight=float(rng.uniform(4.0, 7.0)),
        bear_return_weight=float(rng.uniform(6.0, 14.0)),
        bear_sharpe_weight=float(rng.uniform(1.0, 3.0)),
        turnover_penalty=float(rng.uniform(2.5, 4.0)),
        drawdown_penalty=float(rng.uniform(10.0, 14.0)),
        complexity_penalty=float(rng.uniform(0.04, 0.08)),
    )


def print_progress(label: str, current: int, total: int, detail: str | None = None) -> None:
    total = max(int(total), 1)
    current = max(int(current), 0)
    pct = int(round((current / total) * 100))
    message = f"[progress] {label}: {current}/{total} ({pct}%)"
    if detail:
        message = f"{message} | {detail}"
    print(message, flush=True)


if __name__ == "__main__":
    main()
