"""Event-derived search accounting for the vanilla GP research workflow."""

from __future__ import annotations

from typing import Any

import pandas as pd


STAGE_ORDER = (
    "raw_gp_population",
    "exact_expression_dedup",
    "fast_filter",
    "fast_keep",
    "deep_evaluation",
    "deep_evaluation_pass",
    "new_candidate_pool",
    "current_window_validation_reuse",
    "rolling_revalidation",
    "rolling_pool_trim",
    "final_candidate_pool",
    "initial_factor_selection",
    "portfolio_refinement",
    "final_research_selection",
)


def build_search_funnel(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate persisted events only; no counter-derived funnel rows."""
    required = {"window", "stage", "status", "expression_hash", "individual_id", "reject_reason"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"Stage events missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for window in _ordered_windows(events):
        window_events = events.loc[events["window"] == window]
        for stage in STAGE_ORDER:
            subset = window_events.loc[window_events["stage"] == stage]
            if subset.empty:
                continue
            hashes = set(subset["expression_hash"].dropna().astype(str))
            seen_hashes.update(hashes)
            passed = subset.loc[subset["status"] == "passed"]
            rejected = subset.loc[subset["status"] == "rejected"]
            reasons = rejected["reject_reason"].dropna().astype(str)
            rows.append({
                "window": window,
                "stage": stage,
                "input_count": int(len(subset)),
                "pass_count": int(len(passed)),
                "reject_count": int(len(rejected)),
                "unique_expression_count": int(len(hashes)),
                "cumulative_unique_expression_count": int(len(seen_hashes)),
                "evaluation_count": int(len(subset)) if stage in {"fast_filter", "deep_evaluation", "rolling_revalidation"} else 0,
                "rejection_reasons": ";".join(sorted(reason for reason in reasons.unique() if reason)),
            })
    return pd.DataFrame(rows)


def build_hypothesis_statistics(events: pd.DataFrame, selected_factor_count: int) -> dict[str, Any]:
    """Keep occurrence, evaluation, and unique-hypothesis concepts separate."""
    raw = events.loc[events["stage"] == "raw_gp_population"].copy()
    fast = events.loc[events["stage"] == "fast_filter"].copy()
    deep = events.loc[events["stage"] == "deep_evaluation"].copy()
    if raw.empty:
        return {
            "raw_individual_occurrences": 0,
            "expression_evaluation_calls": 0,
            "fast_evaluation_calls": 0,
            "deep_evaluation_calls": 0,
            "unique_expressions_global": 0,
            "unique_archived_candidates": 0,
            "unique_window_expression_pairs": 0,
            "selected_factor_count": int(selected_factor_count),
            "per_window": {},
        }
    raw["expression_hash"] = raw["expression_hash"].astype(str)
    per_window = {}
    for window, subset in raw.groupby("window", sort=False):
        per_window[str(window)] = {
            "raw_individual_occurrences": int(len(subset)),
            "unique_expressions": int(subset["expression_hash"].nunique()),
            "expression_evaluation_calls": int((fast["window"] == window).sum()),
            "deep_evaluation_calls": int((deep["window"] == window).sum()),
        }
    return {
        "raw_individual_occurrences": int(len(raw)),
        "expression_evaluation_calls": int(len(fast)),
        "fast_evaluation_calls": int(len(fast)),
        "deep_evaluation_calls": int(len(deep)),
        "unique_expressions_global": int(raw["expression_hash"].nunique()),
        "unique_window_expression_pairs": int(raw[["window", "expression_hash"]].drop_duplicates().shape[0]),
        "unique_archived_candidates": int(raw[["window", "expression_hash"]].drop_duplicates().shape[0]),
        "selected_factor_count": int(selected_factor_count),
        "per_window": per_window,
    }


def _ordered_windows(events: pd.DataFrame) -> list[str]:
    windows = [str(value) for value in events["window"].dropna().unique()]
    return sorted(windows, key=lambda value: (value == "final", value))
