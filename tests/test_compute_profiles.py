from alpha_mining.config import COMPUTE_PROFILES, get_compute_profile


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