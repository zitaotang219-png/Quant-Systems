from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_mining.config import AlphaMiningConfig, FitnessConfig, SelectedFactor
from alpha_mining.dsl import field, rank
from alpha_mining.evaluator import FactorEvaluator
from alpha_mining.hypothesis import HypothesisCandidate
from alpha_mining.pipeline import build_candidate_factor_pool
from alpha_mining.research_evaluator import (
    FactorResearchEvaluator,
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
