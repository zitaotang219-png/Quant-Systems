from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.event_driven import EventDrivenPortfolioBacktester
from backtest.ledger import PortfolioLedger
from backtest.portfolio_constraints import PortfolioConstraints
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION
from execution.cost_model import TransactionCostModel

from .config import AlphaMiningConfig, SelectedFactor
from .evaluator import (
    FactorEvaluator,
    EvaluationResult,
    VERY_BAD_FITNESS,
    _prepare_panel,
    compute_factor_fitness,
)
from .gp_generator import GPCandidate, GPGenerator
from .portfolio_construction import build_weight_frame, combine_factor_columns
from .regime import build_regime_frame
from .registry import FactorRegistry


@dataclass
class PortfolioBacktestResult:
    timeseries: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame | None = None
    reconciliation: pd.DataFrame | None = None
    turnover_report: pd.DataFrame | None = None
    constraint_report: pd.DataFrame | None = None
    events: list[dict[str, Any]] | None = None
    trading_convention: dict[str, Any] | None = None
    ledger: PortfolioLedger | None = None


@dataclass
class ResearchSplit:
    train_panel: pd.DataFrame
    validation_panel: pd.DataFrame
    backtest_panel: pd.DataFrame


_STYLE_BUCKET_ORDER = (
    "anti_beta",
    "volume",
    "defensive",
    "pro_cyclical",
    "neutral",
)


@dataclass
class AlphaMiningStrategy:
    selected_factors: list[SelectedFactor]
    evaluator: FactorEvaluator
    long_quantile: float
    short_quantile: float
    factor_weight_scheme: str = "equal"
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
    regime_benchmark_blend: dict[str, float] | None = None
    regime_benchmark_direction: dict[str, float] | None = None
    regime_config: Any = None
    universe_symbols: tuple[str, ...] = ()

    def backtest(
        self,
        panel: pd.DataFrame,
        initial_capital: float,
        transaction_cost_bps: float,
        slippage_bps: float = 0.0,
        regime_source_panel: pd.DataFrame | None = None,
    ) -> PortfolioBacktestResult:
        panel = _filter_universe(panel, self.universe_symbols)
        if not self.selected_factors:
            empty_timeseries = pd.DataFrame(
                columns=["date", "gross_return", "transaction_cost", "net_return", "equity_curve", "drawdown", "turnover", "pnl"]
            )
            empty_weights = pd.DataFrame(columns=["date", "symbol", "weight"])
            return PortfolioBacktestResult(timeseries=empty_timeseries, weights=empty_weights)

        prepared = _prepare_panel(
            panel,
            trading_convention=self.evaluator.trading_convention,
        )
        factor_columns = _evaluate_factor_columns(prepared, self.selected_factors)
        regime_by_date = _regime_series_for_panel(
            prepared if regime_source_panel is None else _filter_universe(regime_source_panel, self.universe_symbols),
            self.regime_config,
        )
        weights = _build_combined_weights(
            panel=prepared,
            factor_columns=factor_columns,
            selected_factors=self.selected_factors,
            weighting_scheme=self.weighting_scheme,
            long_quantile=self.long_quantile,
            short_quantile=self.short_quantile,
            weight_scheme=self.factor_weight_scheme,
            position_limit=self.position_limit,
            turnover_limit=self.turnover_limit,
            gross_leverage=self.gross_leverage,
            signal_vol_window=self.signal_vol_window,
            signal_clip=self.signal_clip,
            smoothing=self.smoothing,
            market_neutral=self.market_neutral,
            benchmark_follow_enabled=self.benchmark_follow_enabled,
            benchmark_follow_btc_symbol=self.benchmark_follow_btc_symbol,
            benchmark_follow_btc_weight=self.benchmark_follow_btc_weight,
            regime_benchmark_blend=self.regime_benchmark_blend,
            regime_benchmark_direction=self.regime_benchmark_direction,
            regime_by_date=regime_by_date,
            regime_config=self.regime_config,
        )
        backtester = EventDrivenPortfolioBacktester(
            initial_capital=initial_capital,
            cost_model=TransactionCostModel(
                commission_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            ),
            constraints=PortfolioConstraints(
                max_position_size=self.position_limit,
                max_leverage=self.gross_leverage,
                max_gross_exposure=self.gross_leverage,
                max_net_exposure=self.gross_leverage,
            ),
            convention=DEFAULT_TRADING_CONVENTION,
        )
        accounting = backtester.run(prepared, weights)
        return PortfolioBacktestResult(
            timeseries=accounting.timeseries,
            weights=weights,
            trades=accounting.trades,
            reconciliation=accounting.reconciliation,
            turnover_report=accounting.turnover,
            constraint_report=accounting.constraints,
            events=accounting.events,
            trading_convention=DEFAULT_TRADING_CONVENTION.to_dict(),
            ledger=accounting.ledger,
        )

    def target_weights(self, panel: pd.DataFrame) -> dict[str, float]:
        panel = _filter_universe(panel, self.universe_symbols)
        if not self.selected_factors or panel.empty:
            return {}
        prepared = _prepare_panel(
            panel,
            trading_convention=self.evaluator.trading_convention,
        )
        factor_columns = _evaluate_factor_columns(prepared, self.selected_factors)
        regime_by_date = _regime_series_for_panel(prepared, self.regime_config)
        weights = _build_combined_weights(
            panel=prepared,
            factor_columns=factor_columns,
            selected_factors=self.selected_factors,
            weighting_scheme=self.weighting_scheme,
            long_quantile=self.long_quantile,
            short_quantile=self.short_quantile,
            weight_scheme=self.factor_weight_scheme,
            position_limit=self.position_limit,
            turnover_limit=self.turnover_limit,
            gross_leverage=self.gross_leverage,
            signal_vol_window=self.signal_vol_window,
            signal_clip=self.signal_clip,
            smoothing=self.smoothing,
            market_neutral=self.market_neutral,
            benchmark_follow_enabled=self.benchmark_follow_enabled,
            benchmark_follow_btc_symbol=self.benchmark_follow_btc_symbol,
            benchmark_follow_btc_weight=self.benchmark_follow_btc_weight,
            regime_benchmark_blend=self.regime_benchmark_blend,
            regime_benchmark_direction=self.regime_benchmark_direction,
            regime_by_date=regime_by_date,
            regime_config=self.regime_config,
        )
        latest_date = pd.to_datetime(weights["date"], utc=False).max()
        latest = weights.loc[pd.to_datetime(weights["date"], utc=False) == latest_date, ["symbol", "weight"]]
        latest = latest.loc[latest["weight"] != 0.0]
        return {str(row["symbol"]): float(row["weight"]) for _, row in latest.iterrows()}


def run_alpha_mining(panel: pd.DataFrame, config: AlphaMiningConfig) -> list[SelectedFactor]:
    panel = _filter_universe(panel, config.universe_symbols)
    registry = FactorRegistry(config.registry_dir())
    if config.live_mode:
        return registry.load(config)

    evaluator = _build_evaluator(config)
    generator = GPGenerator(config.gp)

    if config.walk_forward_enabled:
        selected = _run_walk_forward_alpha_mining(panel, config, evaluator, generator)
    else:
        selected = _run_single_pass_alpha_mining(panel, config, evaluator, generator)

    if config.save_registry:
        registry.save(selected, config, panel)
    return selected


def build_candidate_factor_pool(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    pool_limit: int | None = None,
    fast_keep: int | None = None,
    deep_keep: int | None = None,
    scoring_panel: pd.DataFrame | None = None,
    scoring_split: str = "validation",
    telemetry_callback: Any | None = None,
) -> list[SelectedFactor]:
    filtered_panel = _filter_universe(panel, config.universe_symbols)
    evaluator = _build_evaluator(config)
    pool_evaluator = _build_pool_evaluator(config)
    generator = GPGenerator(_build_pool_gp_config(config))
    resolved_pool_limit = pool_limit or max(
        config.deep_eval_keep * 4,
        config.fast_filter_keep * 3,
        config.portfolio.selected_factor_count * 12,
    )
    resolved_fast_keep = fast_keep or max(config.fast_filter_keep, resolved_pool_limit)
    resolved_deep_keep = deep_keep or resolved_pool_limit

    scoring_target = _filter_universe(scoring_panel, config.universe_symbols) if scoring_panel is not None else None

    if config.walk_forward_enabled and scoring_target is None:
        return _build_walk_forward_candidate_pool(
            filtered_panel,
            config,
            pool_evaluator,
            generator,
            pool_limit=resolved_pool_limit,
            fast_keep=resolved_fast_keep,
            deep_keep=resolved_deep_keep,
        )

    population = generator.evolve(filtered_panel, pool_evaluator, deduplicate=config.deduplicate_expressions)
    deep_results = _deep_evaluate_population(
        candidates=population,
        panel=scoring_target if scoring_target is not None else filtered_panel,
        evaluator=pool_evaluator,
        keep=resolved_deep_keep,
        fast_keep=resolved_fast_keep,
        split_label=scoring_split if scoring_target is not None else None,
        telemetry_generator=generator,
    )
    minimum_pool_floor = max(config.portfolio.min_selected_factor_count * 3, config.portfolio.selected_factor_count)
    if len(deep_results) < minimum_pool_floor:
        deep_results.extend(
            _fallback_expand_candidate_pool(
                candidates=population,
                existing_results=deep_results,
                panel=scoring_target if scoring_target is not None else filtered_panel,
                config=config,
                target_size=max(resolved_pool_limit, minimum_pool_floor * 2),
                split_label=scoring_split if scoring_target is not None else None,
                telemetry_generator=generator,
            )
        )
    result = _results_to_factor_pool(deep_results, resolved_pool_limit)
    if telemetry_callback is not None:
        source_by_expression = {candidate.node.describe(): candidate for candidate in population}
        retained = {factor.expression for factor in result}
        for node, _ in deep_results:
            candidate = source_by_expression.get(node.describe())
            if candidate is not None:
                generator.record_stage_event(
                    candidate,
                    stage="new_candidate_pool",
                    status="passed" if node.describe() in retained else "rejected",
                    reject_reason="" if node.describe() in retained else "new_pool_limit_or_diversity",
                )
    if telemetry_callback is not None:
        telemetry_callback(generator)
    return result


def build_alpha_mining_strategy(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    selected_factors: list[SelectedFactor] | None = None,
) -> AlphaMiningStrategy:
    panel = _filter_universe(panel, config.universe_symbols)
    resolved_selected = selected_factors if selected_factors is not None else run_alpha_mining(panel, config)
    return AlphaMiningStrategy(
        selected_factors=resolved_selected,
        evaluator=_build_evaluator(config),
        long_quantile=config.evaluation.long_quantile,
        short_quantile=config.evaluation.short_quantile,
        factor_weight_scheme=config.portfolio.factor_weight_scheme,
        weighting_scheme=config.portfolio.weighting_scheme,
        position_limit=config.portfolio.position_limit,
        turnover_limit=config.portfolio.turnover_limit,
        gross_leverage=config.portfolio.gross_leverage,
        signal_vol_window=config.portfolio.signal_vol_window,
        signal_clip=config.portfolio.signal_clip,
        smoothing=config.portfolio.smoothing,
        market_neutral=config.portfolio.market_neutral,
        benchmark_follow_enabled=config.portfolio.benchmark_follow_enabled,
        benchmark_follow_btc_symbol=config.portfolio.benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=config.portfolio.benchmark_follow_btc_weight,
        regime_benchmark_blend=dict(config.portfolio.regime_benchmark_blend),
        regime_benchmark_direction=dict(config.portfolio.regime_benchmark_direction),
        regime_config=config.regime,
        universe_symbols=config.universe_symbols,
    )


def select_factors_from_pool(
    candidate_pool: list[SelectedFactor],
    config: AlphaMiningConfig,
    fitness_config: Any | None = None,
) -> list[SelectedFactor]:
    if not candidate_pool:
        return []
    resolved_fitness = fitness_config or config.fitness
    rescored_pool = rescore_candidate_pool(candidate_pool, resolved_fitness)
    return _select_diversified_factor_pool(
        rescored_pool,
        config.portfolio.selected_factor_count,
        config.portfolio.min_selected_factor_count,
        config.portfolio.max_pairwise_correlation,
    )


def rescore_candidate_pool(
    candidate_pool: list[SelectedFactor],
    fitness_config: Any,
) -> list[SelectedFactor]:
    rescored: list[SelectedFactor] = []
    for factor in candidate_pool:
        rescored.append(
            SelectedFactor(
                expression=factor.expression,
                node=factor.node,
                direction=factor.direction,
                fitness=compute_factor_fitness(factor.metrics, factor.complexity, fitness_config),
                metrics=dict(factor.metrics),
                complexity=factor.complexity,
                finite_ratio=factor.finite_ratio,
                values=factor.values,
            )
        )
    rescored.sort(key=lambda item: item.fitness, reverse=True)
    return rescored


def backtest_selected_factors(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    selected_factors: list[SelectedFactor],
    initial_capital: float = 100000.0,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
    risk_fraction: float = 0.2,
    output_dir: str | None = None,
    regime_source_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    filtered_panel = _filter_universe(panel, config.universe_symbols)
    strategy = build_alpha_mining_strategy(filtered_panel, config, selected_factors=selected_factors)
    engine = BacktestEngine(
        initial_capital=initial_capital,
        transaction_cost_bps=(
            config.evaluation.transaction_cost_bps
            if transaction_cost_bps is None
            else transaction_cost_bps
        ),
        slippage_bps=(
            config.evaluation.slippage_bps
            if slippage_bps is None
            else slippage_bps
        ),
        risk_fraction=risk_fraction,
    )
    return engine.run_cross_sectional(
        filtered_panel,
        strategy,
        output_dir=output_dir,
        strategy_backtest_kwargs={"regime_source_panel": regime_source_panel},
    )


def run_walk_forward_evaluation(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    initial_capital: float = 100000.0,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    filtered_panel = _filter_universe(panel, config.universe_symbols)
    folds = _build_walk_forward_folds(
        filtered_panel,
        train_fraction=config.walk_forward_train_fraction,
        validation_fraction=config.walk_forward_validation_fraction,
        backtest_fraction=config.walk_forward_backtest_fraction,
        min_folds=max(config.walk_forward_min_folds, 1),
    )
    if not folds:
        return pd.DataFrame(), {"fold_count": 0}, pd.DataFrame()

    fold_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    return_rows: list[pd.DataFrame] = []
    for fold_id, fold in enumerate(folds, start=1):
        candidate_pool = build_candidate_factor_pool(
            panel=fold["train_panel"],
            config=config,
            pool_limit=max(config.portfolio.selected_factor_count * 10, config.deep_eval_keep * 4),
            fast_keep=max(config.fast_filter_keep * 2, config.portfolio.selected_factor_count * 12),
            deep_keep=max(config.deep_eval_keep * 4, config.portfolio.selected_factor_count * 12),
            scoring_panel=fold["validation_panel"],
            scoring_split="validation",
        )
        selected = select_factors_from_pool(candidate_pool, config, config.fitness)
        result, metrics, _ = backtest_selected_factors(
            panel=fold["backtest_panel"],
            config=config,
            selected_factors=selected,
            initial_capital=initial_capital,
            output_dir=None,
            regime_source_panel=fold["full_panel"],
        )
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": str(pd.to_datetime(fold["train_panel"]["date"]).min().date()),
                "train_end": str(pd.to_datetime(fold["train_panel"]["date"]).max().date()),
                "validation_start": str(pd.to_datetime(fold["validation_panel"]["date"]).min().date()),
                "validation_end": str(pd.to_datetime(fold["validation_panel"]["date"]).max().date()),
                "backtest_start": str(pd.to_datetime(fold["backtest_panel"]["date"]).min().date()),
                "backtest_end": str(pd.to_datetime(fold["backtest_panel"]["date"]).max().date()),
                "selected_factor_count": int(len(selected)),
                **metrics,
            }
        )
        for factor in selected:
            factor_rows.append(
                {
                    "fold": fold_id,
                    "expression": factor.expression,
                    "direction": factor.direction,
                    "fitness": factor.fitness,
                }
            )
        if not result.empty:
            fold_returns = result[["date", "net_return", "equity_curve"]].copy()
            fold_returns["fold"] = fold_id
            return_rows.append(fold_returns)

    fold_df = pd.DataFrame(fold_rows)
    factor_df = pd.DataFrame(factor_rows)
    return_df = pd.concat(return_rows, ignore_index=True) if return_rows else pd.DataFrame()
    summary = {
        "fold_count": int(len(fold_df)),
        "mean_total_return": float(fold_df["total_return"].mean()) if not fold_df.empty else 0.0,
        "median_total_return": float(fold_df["total_return"].median()) if not fold_df.empty else 0.0,
        "mean_sharpe": float(fold_df["sharpe"].mean()) if not fold_df.empty else 0.0,
        "median_sharpe": float(fold_df["sharpe"].median()) if not fold_df.empty else 0.0,
        "mean_max_drawdown": float(fold_df["max_drawdown"].mean()) if not fold_df.empty else 0.0,
        "mean_turnover": float(fold_df["turnover"].mean()) if not fold_df.empty else 0.0,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(root / "walk_forward_fold_metrics.csv", index=False)
        factor_df.to_csv(root / "walk_forward_selected_factors.csv", index=False)
        if not return_df.empty:
            return_df.to_csv(root / "walk_forward_returns.csv", index=False)
        import json

        (root / "walk_forward_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return fold_df, summary, return_df


def run_alpha_mining_backtest(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    universe_symbols: list[str] | tuple[str, ...] | None = None,
    initial_capital: float = 100000.0,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
    risk_fraction: float = 0.2,
    output_dir: str | None = None,
) -> tuple[list[SelectedFactor], pd.DataFrame, dict[str, float], dict[str, Any]]:
    effective_config = config
    if universe_symbols is not None:
        effective_config = AlphaMiningConfig(
            gp=config.gp,
            evaluation=config.evaluation,
            portfolio=config.portfolio,
            fitness=config.fitness,
            regime=config.regime,
            registry=config.registry,
            live_mode=config.live_mode,
            fast_filter_keep=config.fast_filter_keep,
            deep_eval_keep=config.deep_eval_keep,
            walk_forward_enabled=config.walk_forward_enabled,
            walk_forward_train_fraction=config.walk_forward_train_fraction,
            walk_forward_validation_fraction=config.walk_forward_validation_fraction,
            walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
            walk_forward_min_folds=config.walk_forward_min_folds,
            deduplicate_expressions=config.deduplicate_expressions,
            save_registry=config.save_registry,
            universe_symbols=tuple(universe_symbols),
        )

    filtered_panel = _filter_universe(panel, effective_config.universe_symbols)
    selected = run_alpha_mining(filtered_panel, effective_config)
    result, metrics, artifacts = backtest_selected_factors(
        panel=filtered_panel,
        config=effective_config,
        selected_factors=selected,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_fraction=risk_fraction,
        output_dir=output_dir,
    )
    return selected, result, metrics, artifacts


def _run_single_pass_alpha_mining(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    evaluator: FactorEvaluator,
    generator: GPGenerator,
) -> list[SelectedFactor]:
    generator.evolve(panel, evaluator, deduplicate=config.deduplicate_expressions)
    population = generator.archive_candidates()
    deep_results = _deep_evaluate_population(
        candidates=population,
        panel=panel,
        evaluator=evaluator,
        keep=config.deep_eval_keep,
        fast_keep=config.fast_filter_keep,
    )
    return _select_diversified_factors(
        deep_results,
        config.portfolio.selected_factor_count,
        config.portfolio.min_selected_factor_count,
        config.portfolio.max_pairwise_correlation,
    )


def _run_walk_forward_alpha_mining(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    evaluator: FactorEvaluator,
    generator: GPGenerator,
) -> list[SelectedFactor]:
    folds = _build_walk_forward_folds(
        panel,
        train_fraction=config.walk_forward_train_fraction,
        validation_fraction=config.walk_forward_validation_fraction,
        backtest_fraction=config.walk_forward_backtest_fraction,
        min_folds=config.walk_forward_min_folds,
    )
    if not folds:
        return _run_single_pass_alpha_mining(panel, config, evaluator, generator)

    aggregated: dict[str, dict[str, Any]] = {}
    for fold in folds:
        train_panel = fold["train_panel"]
        full_fold_panel = fold["full_panel"]
        split_map = fold["split_map"]

        population = generator.evolve(train_panel, evaluator, deduplicate=config.deduplicate_expressions)
        candidates = [candidate for candidate in population if candidate.evaluation.fitness > VERY_BAD_FITNESS]
        candidates = candidates[: config.fast_filter_keep]

        fold_results: list[tuple[Any, EvaluationResult]] = []
        for candidate in candidates:
            evaluation = evaluator.evaluate_with_splits(candidate.node, full_fold_panel, split_map)
            if evaluation.fitness <= VERY_BAD_FITNESS:
                continue
            fold_results.append((candidate.node, evaluation))

        fold_results.sort(key=lambda item: item[1].fitness, reverse=True)
        for node, evaluation in fold_results[: config.deep_eval_keep]:
            expression = node.describe()
            bucket = aggregated.setdefault(
                expression,
                {
                    "node": node,
                    "evaluations": [],
                },
            )
            bucket["evaluations"].append(evaluation)

    if not aggregated:
        return _run_single_pass_alpha_mining(panel, config, evaluator, generator)

    combined_results: list[tuple[Any, EvaluationResult]] = []
    for payload in aggregated.values():
        evaluations = payload["evaluations"]
        if len(evaluations) < config.walk_forward_min_folds:
            continue
        combined_results.append((payload["node"], _aggregate_evaluations(payload["node"], evaluations)))

    combined_results.sort(key=lambda item: item[1].fitness, reverse=True)
    selected = _select_diversified_factors(
        combined_results,
        config.portfolio.selected_factor_count,
        config.portfolio.min_selected_factor_count,
        config.portfolio.max_pairwise_correlation,
    )
    if not selected:
        return _run_single_pass_alpha_mining(panel, config, evaluator, generator)
    return selected


def _build_walk_forward_candidate_pool(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    evaluator: FactorEvaluator,
    generator: GPGenerator,
    pool_limit: int,
    fast_keep: int,
    deep_keep: int,
) -> list[SelectedFactor]:
    folds = _build_walk_forward_folds(
        panel,
        train_fraction=config.walk_forward_train_fraction,
        validation_fraction=config.walk_forward_validation_fraction,
        backtest_fraction=config.walk_forward_backtest_fraction,
        min_folds=config.walk_forward_min_folds,
    )
    if not folds:
        return build_candidate_factor_pool(
            panel=panel,
            config=AlphaMiningConfig(
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
                save_registry=config.save_registry,
                universe_symbols=config.universe_symbols,
                compute_profile=config.compute_profile,
            ),
            pool_limit=pool_limit,
            fast_keep=fast_keep,
            deep_keep=deep_keep,
        )

    aggregated: dict[str, dict[str, Any]] = {}
    for fold in folds:
        train_panel = fold["train_panel"]
        full_fold_panel = fold["full_panel"]
        split_map = fold["split_map"]

        population = generator.evolve(train_panel, evaluator, deduplicate=config.deduplicate_expressions)
        candidates = [candidate for candidate in population if candidate.evaluation.fitness > VERY_BAD_FITNESS]
        candidates = candidates[:fast_keep]

        fold_results: list[tuple[Any, EvaluationResult]] = []
        for candidate in candidates:
            evaluation = evaluator.evaluate_with_splits(candidate.node, full_fold_panel, split_map)
            if evaluation.fitness <= VERY_BAD_FITNESS:
                continue
            fold_results.append((candidate.node, evaluation))

        fold_results.sort(key=lambda item: item[1].fitness, reverse=True)
        for node, evaluation in fold_results[:deep_keep]:
            expression = node.describe()
            bucket = aggregated.setdefault(expression, {"node": node, "evaluations": []})
            bucket["evaluations"].append(evaluation)

    combined_results: list[tuple[Any, EvaluationResult]] = []
    for payload in aggregated.values():
        combined_results.append((payload["node"], _aggregate_evaluations(payload["node"], payload["evaluations"])))
    combined_results.sort(key=lambda item: item[1].fitness, reverse=True)
    return _results_to_factor_pool(combined_results, pool_limit)


def _deep_evaluate_population(
    candidates: list[GPCandidate],
    panel: pd.DataFrame,
    evaluator: FactorEvaluator,
    keep: int,
    fast_keep: int,
    split_label: str | None = None,
    telemetry_generator: GPGenerator | None = None,
) -> list[tuple[Any, EvaluationResult]]:
    prepared = evaluator.prepare_panel(panel)
    viable_candidates = [candidate for candidate in candidates if candidate.evaluation.fitness > VERY_BAD_FITNESS]
    fast_candidates = viable_candidates[:fast_keep]
    if telemetry_generator is not None:
        for index, candidate in enumerate(viable_candidates):
            telemetry_generator.record_stage_event(
                candidate,
                stage="fast_keep",
                status="passed" if index < fast_keep else "rejected",
                reject_reason="" if index < fast_keep else "fast_keep_limit",
            )

    deep_results: list[tuple[GPCandidate, EvaluationResult]] = []
    split_map = _single_split_map(prepared["date"], split_label) if split_label else None
    for candidate in fast_candidates:
        evaluation = (
            evaluator.evaluate_with_splits(candidate.node, prepared, split_map)
            if split_map is not None
            else evaluator.evaluate(candidate.node, prepared)
        )
        if telemetry_generator is not None:
            telemetry_generator.record_stage_event(candidate, stage="deep_evaluation", status="passed")
        if evaluation.fitness <= VERY_BAD_FITNESS:
            if telemetry_generator is not None:
                telemetry_generator.record_stage_event(
                    candidate,
                    stage="deep_evaluation_pass",
                    status="rejected",
                    reject_reason=str(evaluation.metrics.get("reject_reason", "very_bad_fitness")),
                )
            continue
        if telemetry_generator is not None:
            telemetry_generator.record_stage_event(candidate, stage="deep_evaluation_pass", status="passed")
        deep_results.append((candidate, evaluation))

    deep_results.sort(key=lambda item: item[1].fitness, reverse=True)
    return [(candidate.node, evaluation) for candidate, evaluation in deep_results[:keep]]


def _aggregate_evaluations(node: Any, evaluations: list[EvaluationResult]) -> EvaluationResult:
    metrics_keys = {
        key
        for evaluation in evaluations
        for key, value in evaluation.metrics.items()
        if isinstance(value, (int, float, np.floating))
    }
    averaged_metrics = {
        key: float(np.mean([float(evaluation.metrics.get(key, 0.0)) for evaluation in evaluations]))
        for key in metrics_keys
    }
    averaged_metrics["walk_forward_folds"] = len(evaluations)
    averaged_metrics["walk_forward_only"] = 1.0

    fitness = float(np.mean([evaluation.fitness for evaluation in evaluations]))
    finite_ratio = float(np.mean([evaluation.finite_ratio for evaluation in evaluations]))
    direction = int(np.sign(np.mean([evaluation.direction for evaluation in evaluations])) or 1)
    values = pd.concat([evaluation.values for evaluation in evaluations], axis=1).mean(axis=1)

    return EvaluationResult(
        node=node,
        values=values.astype(float),
        finite_ratio=finite_ratio,
        fitness=fitness,
        direction=direction,
        metrics=averaged_metrics,
        daily_returns=pd.concat([evaluation.daily_returns for evaluation in evaluations], axis=0, ignore_index=True),
        cumulative_return=pd.concat([evaluation.cumulative_return for evaluation in evaluations], axis=0, ignore_index=True),
        split_metrics={},
    )


def _results_to_factor_pool(
    results: list[tuple[Any, EvaluationResult]],
    limit: int | None,
) -> list[SelectedFactor]:
    ranked_candidates: list[SelectedFactor] = []
    seen_expressions: set[str] = set()
    for node, evaluation in results:
        expression = node.describe()
        if expression in seen_expressions:
            continue
        ranked_candidates.append(
            SelectedFactor(
                expression=expression,
                node=node,
                direction=evaluation.direction,
                fitness=evaluation.fitness,
                metrics=dict(evaluation.metrics),
                complexity=node.complexity(),
                finite_ratio=evaluation.finite_ratio,
                values=evaluation.values,
            )
        )
        seen_expressions.add(expression)
        if limit is not None and len(ranked_candidates) >= max(limit * 2, limit):
            break
    return _build_diverse_candidate_pool(ranked_candidates, limit)


def _factor_evaluator_kwargs(config: AlphaMiningConfig) -> dict[str, Any]:
    return {
        "transaction_cost_bps": config.evaluation.transaction_cost_bps,
        "slippage_bps": config.evaluation.slippage_bps,
        "long_quantile": config.evaluation.long_quantile,
        "short_quantile": config.evaluation.short_quantile,
        "trading_convention": DEFAULT_TRADING_CONVENTION,
        "min_finite_ratio": config.evaluation.min_finite_ratio,
        "min_std": config.evaluation.min_std,
        "min_abs_rank_ic": config.evaluation.min_abs_rank_ic,
        "max_turnover": config.evaluation.max_turnover,
        "max_allowed_drawdown": config.evaluation.max_allowed_drawdown,
        "ic_sign_tolerance": config.evaluation.ic_sign_tolerance,
        "annualization": config.evaluation.annualization,
        "fitness_config": config.fitness,
        "weighting_scheme": config.portfolio.weighting_scheme,
        "position_limit": config.portfolio.position_limit,
        "turnover_limit": config.portfolio.turnover_limit,
        "gross_leverage": config.portfolio.gross_leverage,
        "signal_vol_window": config.portfolio.signal_vol_window,
        "signal_clip": config.portfolio.signal_clip,
        "smoothing": config.portfolio.smoothing,
        "market_neutral": config.portfolio.market_neutral,
        "benchmark_follow_enabled": config.portfolio.benchmark_follow_enabled,
        "benchmark_follow_btc_symbol": config.portfolio.benchmark_follow_btc_symbol,
        "benchmark_follow_btc_weight": config.portfolio.benchmark_follow_btc_weight,
        "regime_benchmark_blend": dict(config.portfolio.regime_benchmark_blend),
        "regime_benchmark_direction": dict(config.portfolio.regime_benchmark_direction),
        "regime_config": config.regime,
    }


def _build_evaluator(
    config: AlphaMiningConfig,
    reject_non_profitable: bool = True,
    reject_unstable_return_path: bool = True,
) -> FactorEvaluator:
    return FactorEvaluator(
        **_factor_evaluator_kwargs(config),
        reject_non_profitable=reject_non_profitable,
        reject_unstable_return_path=reject_unstable_return_path,
    )


def _build_pool_gp_config(config: AlphaMiningConfig):
    pool_size = max(config.gp.population_size, config.fast_filter_keep * 2, config.portfolio.selected_factor_count * 10)
    generations = max(config.gp.generations, 6)
    elitism = max(config.gp.elitism, max(6, pool_size // 10))
    return config.gp.__class__(
        population_size=pool_size,
        generations=generations,
        tournament_size=min(config.gp.tournament_size, 4),
        elitism=elitism,
        max_depth=config.gp.max_depth,
        init_max_depth=config.gp.init_max_depth,
        crossover_rate=config.gp.crossover_rate,
        subtree_mutation_rate=config.gp.subtree_mutation_rate,
        point_mutation_rate=config.gp.point_mutation_rate,
        reproduction_rate=config.gp.reproduction_rate,
        seed=config.gp.seed,
        field_names=config.gp.field_names,
        constant_range=config.gp.constant_range,
        periods_choices=config.gp.periods_choices,
        window_choices=config.gp.window_choices,
        wrap_final_with_rank_or_zscore=config.gp.wrap_final_with_rank_or_zscore,
    )


def _build_relaxed_pool_config(config: AlphaMiningConfig) -> AlphaMiningConfig:
    relaxed_evaluation = config.evaluation.__class__(
        transaction_cost_bps=config.evaluation.transaction_cost_bps,
        slippage_bps=config.evaluation.slippage_bps,
        long_quantile=config.evaluation.long_quantile,
        short_quantile=config.evaluation.short_quantile,
        min_finite_ratio=min(config.evaluation.min_finite_ratio, 0.45),
        min_std=config.evaluation.min_std,
        min_abs_rank_ic=min(config.evaluation.min_abs_rank_ic, 0.001),
        max_turnover=max(config.evaluation.max_turnover, 3.0),
        max_allowed_drawdown=max(config.evaluation.max_allowed_drawdown, 0.30),
        ic_sign_tolerance=config.evaluation.ic_sign_tolerance,
        annualization=config.evaluation.annualization,
    )
    relaxed_portfolio = config.portfolio.__class__(
        selected_factor_count=config.portfolio.selected_factor_count,
        min_selected_factor_count=config.portfolio.min_selected_factor_count,
        max_pairwise_correlation=config.portfolio.max_pairwise_correlation,
        factor_weight_scheme=config.portfolio.factor_weight_scheme,
        weighting_scheme=config.portfolio.weighting_scheme,
        position_limit=max(config.portfolio.position_limit, 0.08),
        turnover_limit=max(config.portfolio.turnover_limit, 1.2),
        gross_leverage=max(config.portfolio.gross_leverage, 0.8),
        signal_vol_window=config.portfolio.signal_vol_window,
        signal_clip=max(config.portfolio.signal_clip, 3.0),
        smoothing=max(config.portfolio.smoothing, 0.6),
        market_neutral=config.portfolio.market_neutral,
        benchmark_follow_enabled=config.portfolio.benchmark_follow_enabled,
        benchmark_follow_btc_symbol=config.portfolio.benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=config.portfolio.benchmark_follow_btc_weight,
        regime_benchmark_blend=dict(config.portfolio.regime_benchmark_blend),
        regime_benchmark_direction=dict(config.portfolio.regime_benchmark_direction),
    )
    relaxed_fitness = config.fitness.__class__(
        fast_rank_ic_weight=min(config.fitness.fast_rank_ic_weight, 5.0),
        validation_ic_weight=config.fitness.validation_ic_weight,
        sharpe_weight=config.fitness.sharpe_weight,
        cumulative_return_weight=config.fitness.cumulative_return_weight,
        excess_return_weight=config.fitness.excess_return_weight,
        stability_weight=config.fitness.stability_weight,
        bear_return_weight=config.fitness.bear_return_weight,
        bear_sharpe_weight=config.fitness.bear_sharpe_weight,
        turnover_penalty=max(config.fitness.turnover_penalty, 4.0),
        drawdown_penalty=max(config.fitness.drawdown_penalty, 16.0),
        complexity_penalty=max(config.fitness.complexity_penalty, 0.08),
    )
    return AlphaMiningConfig(
        gp=config.gp,
        evaluation=relaxed_evaluation,
        fitness=relaxed_fitness,
        regime=config.regime,
        portfolio=relaxed_portfolio,
        registry=config.registry,
        live_mode=config.live_mode,
        fast_filter_keep=max(config.fast_filter_keep, config.portfolio.selected_factor_count * 6),
        deep_eval_keep=max(config.deep_eval_keep, config.portfolio.selected_factor_count * 6),
        walk_forward_enabled=False,
        walk_forward_train_fraction=config.walk_forward_train_fraction,
        walk_forward_validation_fraction=config.walk_forward_validation_fraction,
        walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
        walk_forward_min_folds=config.walk_forward_min_folds,
        deduplicate_expressions=config.deduplicate_expressions,
        save_registry=False,
        universe_symbols=config.universe_symbols,
        compute_profile=config.compute_profile,
    )


def _build_pool_evaluator(config: AlphaMiningConfig) -> FactorEvaluator:
    relaxed_config = _build_relaxed_pool_config(config)
    return _build_evaluator(
        relaxed_config,
        reject_non_profitable=False,
        reject_unstable_return_path=False,
    )


def _fallback_expand_candidate_pool(
    candidates: list[GPCandidate],
    existing_results: list[tuple[Any, EvaluationResult]],
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    target_size: int,
    split_label: str | None,
    telemetry_generator: GPGenerator | None = None,
) -> list[tuple[Any, EvaluationResult]]:
    seen_expressions = {node.describe() for node, _ in existing_results}
    relaxed_evaluator = _build_pool_evaluator(config)
    expanded: list[tuple[Any, EvaluationResult]] = []
    split_map = _split_dates(panel["date"]) if split_label is None else {
        pd.Timestamp(date): split_label for date in pd.to_datetime(panel["date"], utc=False)
    }
    for candidate in candidates:
        expression = candidate.node.describe()
        if expression in seen_expressions:
            continue
        evaluation = relaxed_evaluator.evaluate_with_splits(candidate.node, prepared, split_map)
        if evaluation.fitness <= VERY_BAD_FITNESS:
            if telemetry_generator is not None:
                telemetry_generator.record_stage_event(
                    candidate,
                    stage="fallback_expansion",
                    status="rejected",
                    reject_reason=str(evaluation.metrics.get("reject_reason", "very_bad_fitness")),
                )
            continue
        if telemetry_generator is not None:
            telemetry_generator.record_stage_event(candidate, stage="fallback_expansion", status="passed")
        expanded.append((candidate.node, evaluation))
        seen_expressions.add(expression)
        if len(existing_results) + len(expanded) >= target_size:
            break
    return expanded


def _select_diversified_factors(
    deep_results: list[tuple[Any, EvaluationResult]],
    limit: int,
    min_count: int,
    max_pairwise_correlation: float,
) -> list[SelectedFactor]:
    selected: list[SelectedFactor] = []
    backup_candidates: list[SelectedFactor] = []
    selected_value_map: dict[str, pd.Series] = {}
    seen_expressions: set[str] = set()

    for node, evaluation in deep_results:
        expression = node.describe()
        if expression in seen_expressions:
            continue
        candidate = SelectedFactor(
            expression=expression,
            node=node,
            direction=evaluation.direction,
            fitness=evaluation.fitness,
            metrics=evaluation.metrics,
            complexity=node.complexity(),
            finite_ratio=evaluation.finite_ratio,
            values=evaluation.values,
        )
        backup_candidates.append(candidate)
        if _is_too_correlated(evaluation.values, selected_value_map.values(), max_pairwise_correlation):
            continue
        selected.append(candidate)
        selected_value_map[expression] = evaluation.values
        seen_expressions.add(expression)
        if len(selected) >= limit:
            break
    return _fill_factor_shortfall(selected, backup_candidates, min_count, limit, max_pairwise_correlation)


def _select_diversified_factor_pool(
    candidate_pool: list[SelectedFactor],
    limit: int,
    min_count: int,
    max_pairwise_correlation: float,
) -> list[SelectedFactor]:
    selected: list[SelectedFactor] = []
    backup_candidates: list[SelectedFactor] = []
    selected_value_map: dict[str, pd.Series] = {}
    seen_expressions: set[str] = set()

    for factor in candidate_pool:
        if factor.expression in seen_expressions:
            continue
        if factor.values is None:
            continue
        backup_candidates.append(factor)
        if _is_too_correlated(factor.values, selected_value_map.values(), max_pairwise_correlation):
            continue
        selected.append(factor)
        selected_value_map[factor.expression] = factor.values
        seen_expressions.add(factor.expression)
        if len(selected) >= limit:
            break
    return _fill_factor_shortfall(selected, backup_candidates, min_count, limit, max_pairwise_correlation)


def _fill_factor_shortfall(
    selected: list[SelectedFactor],
    ranked_candidates: list[SelectedFactor],
    min_count: int,
    limit: int,
    max_pairwise_correlation: float,
) -> list[SelectedFactor]:
    if len(selected) >= min(min_count, limit):
        return selected[:limit]

    selected_expressions = {factor.expression for factor in selected}
    selected_value_map = {
        factor.expression: factor.values
        for factor in selected
        if factor.values is not None
    }
    staged_thresholds = [
        max_pairwise_correlation,
        min(max_pairwise_correlation + 0.05, 0.75),
        min(max_pairwise_correlation + 0.10, 0.85),
        min(max_pairwise_correlation + 0.15, 0.92),
    ]

    for threshold in staged_thresholds:
        if len(selected) >= min(min_count, limit):
            break
        for factor in ranked_candidates:
            if factor.expression in selected_expressions:
                continue
            if factor.values is not None and _is_too_correlated(factor.values, selected_value_map.values(), threshold):
                continue
            selected.append(factor)
            selected_expressions.add(factor.expression)
            if factor.values is not None:
                selected_value_map[factor.expression] = factor.values
            if len(selected) >= limit:
                break

    if len(selected) < min(min_count, limit):
        for factor in ranked_candidates:
            if factor.expression in selected_expressions:
                continue
            selected.append(factor)
            selected_expressions.add(factor.expression)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _build_diverse_candidate_pool(
    ranked_candidates: list[SelectedFactor],
    limit: int | None,
) -> list[SelectedFactor]:
    if limit is None or len(ranked_candidates) <= limit:
        return ranked_candidates if limit is None else ranked_candidates[:limit]

    per_style_cap = max(3, limit // max(len(_STYLE_BUCKET_ORDER), 1))
    buckets: dict[str, list[SelectedFactor]] = {style: [] for style in _STYLE_BUCKET_ORDER}
    for factor in ranked_candidates:
        style = _infer_pool_style(factor.expression)
        buckets.setdefault(style, []).append(factor)

    selected: list[SelectedFactor] = []
    seen: set[str] = set()
    for style in _STYLE_BUCKET_ORDER:
        for factor in buckets.get(style, [])[:per_style_cap]:
            if factor.expression in seen:
                continue
            selected.append(factor)
            seen.add(factor.expression)
            if len(selected) >= limit:
                return selected[:limit]

    for factor in ranked_candidates:
        if factor.expression in seen:
            continue
        selected.append(factor)
        seen.add(factor.expression)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _infer_pool_style(expression: str) -> str:
    lowered = expression.lower()
    if any(token in lowered for token in ("rolling_beta", "beta_adjusted", "basket_return")):
        return "anti_beta"
    if any(token in lowered for token in ("volume", "turnover", "liquidity")):
        return "volume"
    if any(token in lowered for token in ("volatility", "rolling_std", "amplitude", "residual", "idiosyncratic", "shadow")):
        return "defensive"
    if any(token in lowered for token in ("gap", "momentum", "relative_strength", "trend", "breakout")):
        return "pro_cyclical"
    return "neutral"


def _is_too_correlated(values: pd.Series, selected_values: Any, threshold: float) -> bool:
    current = values.replace([np.inf, -np.inf], np.nan)
    for existing in selected_values:
        aligned = pd.concat([current, existing.replace([np.inf, -np.inf], np.nan)], axis=1).dropna()
        if len(aligned) < 5:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.notna(corr) and abs(float(corr)) > threshold:
            return True
    return False


def _evaluate_factor_columns(panel: pd.DataFrame, selected_factors: list[SelectedFactor]) -> dict[str, pd.Series]:
    factor_columns: dict[str, pd.Series] = {}
    for selected in selected_factors:
        factor_columns[selected.expression] = selected.node.evaluate(panel).astype(float) * float(selected.direction)
    return factor_columns


def _build_combined_weights(
    panel: pd.DataFrame,
    factor_columns: dict[str, pd.Series],
    selected_factors: list[SelectedFactor],
    weighting_scheme: str,
    long_quantile: float,
    short_quantile: float,
    weight_scheme: str,
    position_limit: float,
    turnover_limit: float,
    gross_leverage: float,
    signal_vol_window: int,
    signal_clip: float,
    smoothing: float,
    market_neutral: bool,
    benchmark_follow_enabled: bool,
    benchmark_follow_btc_symbol: str,
    benchmark_follow_btc_weight: float,
    regime_benchmark_blend: dict[str, float] | None,
    regime_benchmark_direction: dict[str, float] | None,
    regime_by_date: pd.Series | None,
    regime_config: Any,
) -> pd.DataFrame:
    if not factor_columns:
        frame = panel[["date", "symbol"]].copy()
        frame["weight"] = 0.0
        return frame
    combined = combine_factor_columns(
        factor_columns,
        selected_factors,
        weight_scheme,
        dates=panel["date"],
        regime_by_date=regime_by_date,
        regime_config=regime_config,
    )
    return build_weight_frame(
        panel=panel,
        score_series=combined,
        weighting_scheme=weighting_scheme,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        position_limit=position_limit,
        gross_leverage=gross_leverage,
        signal_vol_window=signal_vol_window,
        signal_clip=signal_clip,
        smoothing=smoothing,
        market_neutral=market_neutral,
        turnover_limit=turnover_limit,
        regime_by_date=regime_by_date,
        benchmark_follow_enabled=benchmark_follow_enabled,
        benchmark_follow_btc_symbol=benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=benchmark_follow_btc_weight,
        regime_benchmark_blend=regime_benchmark_blend,
        regime_benchmark_direction=regime_benchmark_direction,
    )


def _regime_series_for_panel(panel: pd.DataFrame, regime_config: Any) -> pd.Series | None:
    if regime_config is None or not getattr(regime_config, "enabled", False):
        return None
    regime_frame = build_regime_frame(panel, regime_config)
    if regime_frame.empty:
        return None
    return regime_frame.set_index("date")["regime"]


def _filter_universe(panel: pd.DataFrame, universe_symbols: tuple[str, ...]) -> pd.DataFrame:
    if not universe_symbols:
        return panel.copy()
    filtered = panel.loc[panel["symbol"].astype(str).isin(tuple(str(symbol) for symbol in universe_symbols))].copy()
    if filtered.empty:
        raise ValueError("No rows remain after applying universe_symbols filter.")
    return filtered.reset_index(drop=True)


def build_research_split(
    panel: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    backtest_fraction: float,
) -> ResearchSplit:
    dates = np.sort(pd.to_datetime(panel["date"], utc=False).dropna().unique())
    total = len(dates)
    if total < 10:
        raise ValueError("Not enough research dates to create research split.")

    fraction_sum = float(train_fraction + validation_fraction + backtest_fraction)
    if fraction_sum <= 0.0:
        raise ValueError("Research split fractions must sum to a positive value.")
    normalized_train = train_fraction / fraction_sum
    normalized_validation = validation_fraction / fraction_sum

    train_len = max(5, int(total * normalized_train))
    validation_len = max(3, int(total * normalized_validation))
    remaining = total - train_len - validation_len
    if remaining < 0:
        validation_len = max(3, total - train_len)
        remaining = total - train_len - validation_len
    wants_backtest = float(backtest_fraction) > 0.0
    if wants_backtest:
        backtest_len = max(3, remaining)
        if train_len + validation_len + backtest_len > total:
            backtest_len = total - train_len - validation_len
        if backtest_len < 3:
            raise ValueError("Research split leaves too few backtest dates.")
    else:
        backtest_len = 0

    train_dates = dates[:train_len]
    validation_dates = dates[train_len : train_len + validation_len]
    backtest_dates = dates[train_len + validation_len : train_len + validation_len + backtest_len]

    train_panel = panel.loc[pd.to_datetime(panel["date"], utc=False).isin(train_dates)].copy().reset_index(drop=True)
    validation_panel = panel.loc[pd.to_datetime(panel["date"], utc=False).isin(validation_dates)].copy().reset_index(drop=True)
    backtest_panel = panel.loc[pd.to_datetime(panel["date"], utc=False).isin(backtest_dates)].copy().reset_index(drop=True)
    if train_panel.empty or validation_panel.empty:
        raise ValueError("Research split produced an empty train/validation panel.")
    if wants_backtest and backtest_panel.empty:
        raise ValueError("Research split produced an empty backtest panel.")
    return ResearchSplit(
        train_panel=train_panel,
        validation_panel=validation_panel,
        backtest_panel=backtest_panel,
    )


def _build_walk_forward_folds(
    panel: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    backtest_fraction: float,
    min_folds: int,
) -> list[dict[str, Any]]:
    dates = np.sort(pd.to_datetime(panel["date"], utc=False).dropna().unique())
    total = len(dates)
    if total < 15:
        return []

    denominator = float(train_fraction + validation_fraction + (backtest_fraction * max(min_folds, 1)))
    if denominator <= 0.0:
        return []
    train_len = max(5, int(total * (train_fraction / denominator)))
    validation_len = max(3, int(total * (validation_fraction / denominator)))
    backtest_len = max(3, int(total * (backtest_fraction / denominator)))
    window_len = train_len + validation_len + backtest_len
    if window_len > total:
        return []

    step = backtest_len
    folds: list[dict[str, Any]] = []
    for start in range(0, total - window_len + 1, step):
        train_dates = dates[start : start + train_len]
        validation_dates = dates[start + train_len : start + train_len + validation_len]
        backtest_dates = dates[start + train_len + validation_len : start + window_len]
        full_dates = np.concatenate([train_dates, validation_dates, backtest_dates])
        full_panel = panel.loc[pd.to_datetime(panel["date"], utc=False).isin(full_dates)].copy().reset_index(drop=True)
        train_panel = panel.loc[pd.to_datetime(panel["date"], utc=False).isin(train_dates)].copy().reset_index(drop=True)
        if full_panel.empty or train_panel.empty:
            continue
        split_map: dict[pd.Timestamp, str] = {}
        for date in train_dates:
            split_map[pd.Timestamp(date)] = "train"
        for date in validation_dates:
            split_map[pd.Timestamp(date)] = "validation"
        for date in backtest_dates:
            split_map[pd.Timestamp(date)] = "backtest"
        folds.append(
            {
                "train_panel": train_panel,
                "validation_panel": panel.loc[pd.to_datetime(panel["date"], utc=False).isin(validation_dates)].copy().reset_index(drop=True),
                "backtest_panel": panel.loc[pd.to_datetime(panel["date"], utc=False).isin(backtest_dates)].copy().reset_index(drop=True),
                "full_panel": full_panel,
                "split_map": split_map,
            }
        )
    return folds if len(folds) >= min_folds else []


def _single_split_map(dates: pd.Series, split_label: str) -> dict[pd.Timestamp, str]:
    return {
        pd.Timestamp(date): str(split_label)
        for date in pd.to_datetime(pd.Series(dates).dropna().unique(), utc=False)
    }
