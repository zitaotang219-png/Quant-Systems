from alpha_mining.config import AlphaMiningConfig, COMPUTE_PROFILES, FitnessConfig, GPConfig, get_compute_profile
from alpha_mining.pipeline import _build_pool_gp_config
from alpha_mining.run_crypto_workflow import build_window_fitness_profiles


def test_compute_profiles_are_internally_coherent() -> None:
    for profile in COMPUTE_PROFILES.values():
        raw_budget = profile.population_size * (profile.generations + 1)
        assert profile.deep_eval_keep <= profile.fast_filter_keep <= raw_budget
        assert profile.selected_factor_count <= profile.deep_eval_keep
        assert profile.min_selected_factor_count <= profile.selected_factor_count
        assert profile.validated_pool_limit >= profile.selected_factor_count


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
