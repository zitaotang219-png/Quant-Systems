from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from alpha_mining.config import AlphaMiningConfig, FitnessConfig, SelectedFactor
from alpha_mining.dsl import field, rank
from alpha_mining.evaluator import FactorEvaluator
from alpha_mining.hypothesis import HypothesisCandidate
from alpha_mining.pipeline import build_candidate_factor_pool
from alpha_mining.research_evaluator import (
    FactorResearchEvaluator,
    _daily_cross_sectional_rank_ic,
    _daily_cross_sectional_rank_ic_context,
    _signal_turnover,
    _signal_turnover_context,
    _temporal_ic_stability,
)
from alpha_mining.run_crypto_workflow import (
    finalize_factors_for_run,
    validate_rolling_window_pool,
)


def _research_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=4)
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC"), start=1):
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": 100.0 + symbol_index + date_index,
                "high": 101.0 + symbol_index + date_index,
                "low": 99.0 + symbol_index + date_index,
                "close": 100.5 + symbol_index + date_index,
                "volume": 1_000.0 * symbol_index,
                "signal": float(symbol_index),
                "other_signal": float(4 - symbol_index + date_index),
                "future_return": 0.01 * symbol_index,
            })
    return pd.DataFrame(rows)


def _selected(expression: str, node, *, fitness: float = 1.0, passes: int = 1) -> SelectedFactor:
    return SelectedFactor(
        expression=expression,
        node=node,
        direction=1,
        fitness=fitness,
        metrics={
            "validation_rank_ic_mean": 0.1,
            "temporal_ic_stability": 1.0,
            "signal_turnover": 0.0,
            "window_pass_count": passes,
        },
        complexity=node.complexity(),
        finite_ratio=1.0,
        values=pd.Series([1.0, 2.0, 3.0]),
    )


def test_research_evaluation_is_context_keyed_and_contains_no_portfolio_metrics() -> None:
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0)
    panel = _research_panel()
    split_map = {pd.Timestamp(date): "validation" for date in panel["date"].unique()}
    context = evaluator.create_context(panel, name="window_1:validation", split_map=split_map)

    first = evaluator.evaluate_context(field("signal"), context, mode="research")
    second = evaluator.evaluate_context(field("signal"), context, mode="research")

    assert first is second
    assert sum(evaluator.evaluation_calls.values()) == 1
    assert "temporal_ic_stability" in first.metrics
    assert "signal_turnover" in first.metrics
    forbidden = {
        "sharpe",
        "cumulative_return",
        "excess_return_vs_equal_weight",
        "max_drawdown",
        "bear_cumulative_return",
        "bear_sharpe",
        "transaction_cost",
    }
    assert forbidden.isdisjoint(first.metrics)

    other_context = evaluator.create_context(panel, name="window_2:validation", split_map=split_map)
    evaluator.evaluate_context(field("signal"), other_context, mode="research")
    assert sum(evaluator.evaluation_calls.values()) == 2


def test_ic_stability_uses_the_ic_series_through_time() -> None:
    stable = _temporal_ic_stability(pd.Series([0.2, 0.1, 0.3, 0.05]))
    unstable = _temporal_ic_stability(pd.Series([0.2, -0.1, 0.3, -0.05]))
    assert stable == 1.0
    assert unstable == 0.0


def test_native_rolling_rank_preserves_average_percentile_semantics() -> None:
    from alpha_mining.dsl import _rolling_percentile_rank

    values = pd.Series([1.0, 2.0, 2.0, np.nan, 3.0, 1.0, 1.0, 4.0, 4.0])

    def legacy_rank_last(window: pd.Series) -> float:
        valid = window.dropna()
        return np.nan if valid.empty else float(valid.rank(pct=True).iloc[-1])

    expected = values.rolling(window=3, min_periods=3).apply(legacy_rank_last, raw=False)
    actual = _rolling_percentile_rank(values, window=3)
    pdt.assert_series_equal(actual, expected)


def test_native_rolling_correlation_preserves_grouped_semantics() -> None:
    from alpha_mining.dsl import _rolling_correlation

    panel = _research_panel().sample(frac=1.0, random_state=17)
    left = panel["signal"]
    right = panel["other_signal"]
    expected = pd.Series(np.nan, index=panel.index, dtype=float)
    ordered = panel.sort_values(["symbol", "date"], kind="mergesort")
    for _, group in ordered.groupby("symbol", sort=False):
        expected.loc[group.index] = group["signal"].rolling(3, min_periods=3).corr(group["other_signal"])

    actual = _rolling_correlation(panel, left, right, window=3)
    pdt.assert_series_equal(actual, expected)


def test_precomputed_rank_ic_and_turnover_match_reference_paths() -> None:
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0)
    panel = _research_panel()
    split_map = {pd.Timestamp(date): "validation" for date in panel["date"].unique()}
    context = evaluator.create_context(panel, name="equivalence:validation", split_map=split_map)
    values = panel["signal"].astype(float).copy()
    values.iloc[[1, 7]] = np.nan

    reference_ic = _daily_cross_sectional_rank_ic(panel["date"], values, panel["future_return"])
    optimized_ic = _daily_cross_sectional_rank_ic_context(context, values)
    reference_ic = reference_ic.sort_values("date").reset_index(drop=True)
    optimized_ic = optimized_ic.sort_values("date").reset_index(drop=True)
    pdt.assert_frame_equal(optimized_ic, reference_ic, check_exact=False, atol=1e-12, rtol=1e-12)

    reference_turnover = _signal_turnover(panel, values, split_map)
    optimized_turnover = _signal_turnover_context(context, values)
    assert optimized_turnover == reference_turnover


def test_research_metrics_match_reference_components() -> None:
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0, max_signal_turnover=99.0)
    panel = _research_panel()
    split_map = {pd.Timestamp(date): "validation" for date in panel["date"].unique()}
    context = evaluator.create_context(panel, name="metric_equivalence:validation", split_map=split_map)
    result = evaluator.evaluate_context(field("signal"), context, mode="research")

    reference_ic = _daily_cross_sectional_rank_ic(panel["date"], panel["signal"], panel["future_return"])
    raw_mean_ic = float(reference_ic["rank_ic"].mean())
    direction = -1 if raw_mean_ic < 0.0 else 1
    oriented_ic = reference_ic["rank_ic"] * direction
    reference_turnover = _signal_turnover(panel, panel["signal"] * direction, split_map)

    assert result.direction == direction
    assert result.metrics["validation_rank_ic_mean"] == float(oriented_ic.mean())
    assert result.metrics["temporal_ic_stability"] == _temporal_ic_stability(oriented_ic)
    assert result.metrics["signal_turnover"] == reference_turnover
    pdt.assert_series_equal(result.values, panel["signal"].astype(float) * direction)


def test_instrumentation_records_value_and_result_reuse() -> None:
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0, max_signal_turnover=99.0)
    panel = _research_panel()
    context = evaluator.create_context(panel, name="instrumentation:validation")
    node = field("signal")

    evaluator.evaluate_context(node, context, mode="fast")
    evaluator.evaluate_context(node, context, mode="research")
    evaluator.evaluate_context(node, context, mode="research")
    stats = evaluator.instrumentation_snapshot()

    assert stats["total_expression_evaluations"] == 3
    assert stats["unique_expression_evaluations"] == 1
    assert stats["fast_screen_evaluations"] == 1
    assert stats["factor_research_evaluations"] == 1
    assert stats["evaluation_cache_hits"] == 1
    assert stats["value_cache_hits"] >= 1
    assert stats["rank_ic_cache_hits"] == 1
    assert stats["rank_ic_evaluations"] == 1
    assert stats["avoided_duplicate_evaluations"] >= 2
    assert stats["portfolio_simulation_calls"] == 0


def test_candidate_search_does_not_invoke_portfolio_simulation(monkeypatch) -> None:
    def forbidden_portfolio_call(*args, **kwargs):
        raise AssertionError("candidate search invoked the portfolio simulator")

    monkeypatch.setattr(FactorEvaluator, "_simulate_long_short_portfolio", forbidden_portfolio_call)

    class SingleHypothesisGenerator:
        def generate(self, panel, evaluator, *, deduplicate=True):
            node = field("signal")
            return [HypothesisCandidate(node=node, evaluation=evaluator.fast_filter(node, panel))]

    factors = build_candidate_factor_pool(
        panel=_research_panel(),
        scoring_panel=_research_panel(),
        config=AlphaMiningConfig(),
        pool_limit=1,
        fast_keep=1,
        deep_keep=1,
        hypothesis_generator=SingleHypothesisGenerator(),
        research_context_name="window_1:validation",
    )
    assert [factor.expression for factor in factors] == ["signal"]


def test_current_window_candidates_are_reused_while_carried_factors_are_revalidated(monkeypatch) -> None:
    previous_same = _selected("signal", field("signal"), passes=2)
    previous_carried = _selected("rank(signal)", rank(field("signal")), passes=2)
    current_same = _selected("signal", field("signal"), passes=1)
    current_new = _selected("other_signal", field("other_signal"), passes=1)
    captured = []

    def record_revalidation(candidate_pool, *args, **kwargs):
        captured.extend(factor.expression for factor in candidate_pool)
        return candidate_pool

    monkeypatch.setattr("alpha_mining.run_crypto_workflow.revalidate_factor_pool", record_revalidation)
    combined = validate_rolling_window_pool(
        previous_pool=[previous_same, previous_carried],
        new_pool=[current_same, current_new],
        validation_panel=_research_panel(),
        config=AlphaMiningConfig(),
        fitness_config=FitnessConfig(),
        context_name="window_2:validation",
    )

    assert captured == ["rank(signal)"]
    by_expression = {factor.expression: factor for factor in combined}
    assert set(by_expression) == {"signal", "rank(signal)", "other_signal"}
    assert by_expression["signal"].metrics["window_pass_count"] == 3
    assert by_expression["other_signal"].metrics["window_pass_count"] == 1


def test_carried_factor_with_current_context_result_is_not_revalidated(monkeypatch) -> None:
    panel = _research_panel()
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0, max_signal_turnover=99.0)
    context_name = "window_2:validation"
    split_map = {pd.Timestamp(date): "validation" for date in panel["date"].unique()}
    context = evaluator.create_context(panel, name=context_name, split_map=split_map)
    node = rank(field("signal"))
    evaluator.evaluate_context(node, context, mode="research")

    def forbidden_revalidation(*args, **kwargs):
        raise AssertionError("cached current-window factor was revalidated")

    monkeypatch.setattr("alpha_mining.run_crypto_workflow.revalidate_factor_pool", forbidden_revalidation)
    combined = validate_rolling_window_pool(
        previous_pool=[_selected("rank(signal)", node, passes=2)],
        new_pool=[],
        validation_panel=panel,
        config=AlphaMiningConfig(),
        fitness_config=FitnessConfig(),
        context_name=context_name,
        research_evaluator=evaluator,
    )

    assert [factor.expression for factor in combined] == ["rank(signal)"]
    assert combined[0].metrics["window_pass_count"] == 3
    assert evaluator.instrumentation_snapshot()["factor_research_evaluations"] == 1


def test_research_only_finalization_skips_strategy_refinement(monkeypatch, tmp_path: Path) -> None:
    selected = [_selected("signal", field("signal"))]

    def forbidden_refinement(**kwargs):
        raise AssertionError("research-only invoked strategy-level refinement")

    monkeypatch.setattr(
        "alpha_mining.run_crypto_workflow.refine_selected_factors_before_backtest",
        forbidden_refinement,
    )
    result = finalize_factors_for_run(
        research_only=True,
        selected=selected,
        validation_panel=_research_panel(),
        config=AlphaMiningConfig(),
        initial_capital=1_000.0,
        regime_source_panel=None,
        min_count=1,
        output_dir=tmp_path,
    )
    assert result is selected
