from pathlib import Path
from dataclasses import asdict

import pandas as pd
import pytest

from alpha_mining.config import (
    AlphaMiningConfig,
    COMPUTE_PROFILES,
    FitnessConfig,
    GPConfig,
    PortfolioConfig,
    SelectedFactor,
    get_compute_profile,
)
from alpha_mining.dsl import parse_expression
from alpha_mining.pipeline import _build_pool_gp_config, select_factors_from_pool
from alpha_mining.run_crypto_workflow import (
    build_crypto_workflow_config,
    build_workflow_config_payload,
    build_window_fitness_profiles,
    resolve_rolling_window_specs,
    resolved_compute_budget,
    summarize_observed_compute_budget,
    trim_candidate_pool,
)


def test_compute_profiles_are_internally_coherent() -> None:
    for profile in COMPUTE_PROFILES.values():
        raw_budget = profile.population_size * (profile.generations + 1)
        assert profile.deep_eval_keep <= profile.fast_filter_keep <= raw_budget
        assert profile.selected_factor_count <= profile.deep_eval_keep
        assert profile.min_selected_factor_count <= profile.selected_factor_count
        assert profile.validated_pool_limit >= profile.selected_factor_count


def test_profile_ladder_has_explicit_bounded_scope() -> None:
    smoke = COMPUTE_PROFILES["smoke"]
    research = COMPUTE_PROFILES["research"]
    certified = COMPUTE_PROFILES["certified"]
    assert (
        smoke.rolling_window_count,
        smoke.population_size,
        smoke.generations,
        smoke.fast_filter_keep,
        smoke.deep_eval_keep,
        smoke.selected_factor_count,
    ) == (1, 8, 1, 6, 3, 2)
    assert smoke.min_selected_factor_count == 0
    assert (
        research.rolling_window_count,
        research.population_size,
        research.generations,
        research.fast_filter_keep,
        research.deep_eval_keep,
        research.validated_pool_limit,
        research.selected_factor_count,
    ) == (3, 12, 1, 8, 4, 8, 4)
    assert (certified.rolling_window_count, certified.population_size, certified.generations) == (3, 24, 2)
    assert len(resolve_rolling_window_specs("smoke")) == 1
    assert len(resolve_rolling_window_specs("research")) == 3
    assert len(resolve_rolling_window_specs("certified")) == 3


def test_profile_lookup_is_case_insensitive_and_rejects_unknown() -> None:
    assert get_compute_profile("RESEARCH") == COMPUTE_PROFILES["research"]
    try:
        get_compute_profile("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown profiles must be rejected")


def test_pool_gp_config_preserves_each_compute_profile_budget() -> None:
    for profile in COMPUTE_PROFILES.values():
        gp = GPConfig(population_size=profile.population_size, generations=profile.generations)
        assert _build_pool_gp_config(AlphaMiningConfig(gp=gp)) == gp


def test_resolved_budget_matches_workflow_config_and_is_manifest_ready() -> None:
    panel = pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]
    })
    for name in COMPUTE_PROFILES:
        config = build_crypto_workflow_config(Path("run"), ["AAA"], panel, compute_profile=name)
        profile = COMPUTE_PROFILES[name]
        budget = resolved_compute_budget(config)
        assert budget["rolling_window_count"] == profile.rolling_window_count
        assert budget["population_size"] == profile.population_size
        assert budget["generations"] == profile.generations
        assert budget["fast_screen_keep_per_window"] == profile.fast_filter_keep
        assert budget["factor_research_evaluation_cap_per_new_window_search"] == profile.deep_eval_keep
        assert budget["automatic_budget_expansion"] is False
        assert config.gp.elitism == 2
        payload = build_workflow_config_payload(config, resolve_rolling_window_specs(name))
        assert payload["resolved_compute_budget"] == budget
        assert len(payload["resolved_research_windows"]) == profile.rolling_window_count


def test_profiles_change_scope_not_research_methodology() -> None:
    panel = pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]
    })
    configs = {
        name: build_crypto_workflow_config(Path(name), ["AAA"], panel, compute_profile=name)
        for name in COMPUTE_PROFILES
    }
    reference = configs["research"]
    for config in configs.values():
        assert config.evaluation == reference.evaluation
        assert config.fitness == reference.fitness
        assert config.regime == reference.regime
        gp_method = asdict(config.gp)
        reference_gp_method = asdict(reference.gp)
        for budget_field in ("population_size", "generations"):
            gp_method.pop(budget_field)
            reference_gp_method.pop(budget_field)
        assert gp_method == reference_gp_method
        portfolio_method = asdict(config.portfolio)
        reference_portfolio_method = asdict(reference.portfolio)
        for budget_field in ("selected_factor_count", "min_selected_factor_count"):
            portfolio_method.pop(budget_field)
            reference_portfolio_method.pop(budget_field)
        assert portfolio_method == reference_portfolio_method


def test_observed_budget_enforcement_rejects_hidden_deep_expansion() -> None:
    profile = COMPUTE_PROFILES["smoke"]
    config = AlphaMiningConfig(
        gp=GPConfig(population_size=profile.population_size, generations=profile.generations),
        fast_filter_keep=profile.fast_filter_keep,
        deep_eval_keep=profile.deep_eval_keep,
        compute_profile="smoke",
    )
    specs = resolve_rolling_window_specs("smoke")
    rows = [
        {
            "window": "window_1", "stage": "deep_evaluation", "status": "passed",
            "expression_hash": str(index), "individual_id": str(index), "reject_reason": "",
        }
        for index in range(profile.deep_eval_keep + 1)
    ]
    with pytest.raises(RuntimeError, match="research-evaluation budget"):
        summarize_observed_compute_budget(
            config=config,
            window_specs=specs,
            stage_events=pd.DataFrame(rows),
            final_pool_size=0,
            selected_factor_count=0,
        )


def test_selection_does_not_relax_redundancy_threshold_to_fill_minimum() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    factors = [
        SelectedFactor(
            expression=expression,
            node=parse_expression(expression),
            direction=1,
            fitness=1.0,
            metrics={"validation_rank_ic": 0.1},
            complexity=1,
            finite_ratio=1.0,
            values=values.copy(),
        )
        for expression in ("volume", "rank(volume)")
    ]
    config = AlphaMiningConfig(
        portfolio=PortfolioConfig(
            selected_factor_count=2,
            min_selected_factor_count=2,
            max_pairwise_correlation=0.45,
        )
    )
    selected = select_factors_from_pool(factors, config)
    assert len(selected) == 1


def test_pool_trim_does_not_refill_with_redundant_factors() -> None:
    values = pd.Series(range(12), dtype=float)
    factors = [
        SelectedFactor(
            expression=expression,
            node=parse_expression(expression),
            direction=1,
            fitness=float(3 - index),
            metrics={},
            complexity=1,
            finite_ratio=1.0,
            values=values.copy(),
        )
        for index, expression in enumerate(("volume", "rank(volume)", "zscore(volume)"))
    ]
    assert len(trim_candidate_pool(factors, pool_limit=2)) == 1


def test_rolling_windows_share_the_factor_research_fitness() -> None:
    base = FitnessConfig(
        fast_rank_ic_weight=7.0,
        validation_ic_weight=999.0,
        sharpe_weight=999.0,
        cumulative_return_weight=999.0,
        excess_return_weight=999.0,
        stability_weight=999.0,
        bear_return_weight=999.0,
        bear_sharpe_weight=999.0,
        turnover_penalty=999.0,
        drawdown_penalty=999.0,
        complexity_penalty=999.0,
    )
    profiles = build_window_fitness_profiles(base)
    assert len(set(profiles.values())) == 1
    fitness = profiles["balanced"]
    assert fitness.fast_rank_ic_weight == 7.0
    assert fitness.validation_ic_weight == 25.0
    assert fitness.stability_weight == 8.0
    assert fitness.turnover_penalty == 4.0
    assert fitness.complexity_penalty == 0.1
    assert fitness.sharpe_weight == 0.0
    assert fitness.cumulative_return_weight == 0.0
    assert fitness.excess_return_weight == 0.0
    assert fitness.drawdown_penalty == 0.0
    assert fitness.bear_return_weight == 0.0
    assert fitness.bear_sharpe_weight == 0.0
