from __future__ import annotations

import json

import pandas as pd
import pytest

from alpha_mining.config import AlphaMiningConfig, GPConfig, PortfolioConfig, SelectedFactor
from alpha_mining.evaluator import EvaluationResult
from alpha_mining.gp_generator import GPGenerator
from alpha_mining.pipeline import select_factors_from_pool
from alpha_mining.run_crypto_workflow import main, namespace_gp_telemetry, prepare_workflow_panels
from alpha_mining.search_funnel import STAGE_ORDER, build_hypothesis_statistics, build_search_funnel
from alpha_mining.dsl import rank
from utils.experiment_ledger import ExperimentLedger


class _Evaluator:
    def fast_filter(self, node, panel):
        return EvaluationResult(
            node=node,
            values=pd.Series(dtype=float),
            finite_ratio=1.0,
            fitness=float(-node.complexity()),
            direction=1,
        )


class _SilentGenerator(GPGenerator):
    def _record_generation(self, *args, **kwargs) -> None:
        return None


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "symbol": ["AAA"],
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "feature": [1.0],
    })


@pytest.mark.parametrize("operator", ["crossover", "subtree_mutation", "point_mutation", "reproduction"])
def test_lineage_uses_real_parent_individual_ids(operator: str) -> None:
    rates = {"crossover_rate": 0.0, "subtree_mutation_rate": 0.0, "point_mutation_rate": 0.0, "reproduction_rate": 0.0}
    rates[f"{operator}_rate"] = 1.0
    config = GPConfig(
        population_size=8, generations=1, elitism=1, seed=11,
        field_names=("volume", "feature"), disallowed_raw_field_names=(), **rates,
    )
    generator = GPGenerator(config)
    generator.evolve(_panel(), _Evaluator(), deduplicate=False)
    _, audit = generator.telemetry_frames()
    ids = set(audit["individual_id"])
    assert audit.loc[audit["generation"] == 0, "parent_individual_ids"].eq("").all()
    children = audit.loc[(audit["generation"] == 1) & (audit["creation_operator"] == operator)]
    assert not children.empty
    for parents in children["parent_individual_ids"]:
        assert parents
        assert set(parents.split(" | ")).issubset(ids)
    elite = audit.loc[(audit["generation"] == 1) & (audit["creation_operator"] == "elite")]
    assert not elite.empty
    assert set(elite.iloc[0]["parent_individual_ids"].split(" | ")).issubset(ids)


def test_window_namespace_keeps_ids_and_parent_edges_unambiguous() -> None:
    generation = pd.DataFrame([{"generation": 0}])
    candidates = pd.DataFrame([{
        "individual_id": "generation-1-slot-2", "duplicate_of_id": "", "parent_individual_ids": "generation-0-slot-1",
    }])
    events = pd.DataFrame([{"individual_id": "generation-1-slot-2"}])
    _, left, left_events = namespace_gp_telemetry(generation, candidates, events, "window_1")
    _, right, right_events = namespace_gp_telemetry(generation, candidates, events, "window_2")
    assert set(left["individual_id"]).isdisjoint(set(right["individual_id"]))
    assert left.loc[0, "parent_individual_ids"] == "window_1-generation-0-slot-1"
    assert "parent_1_id" not in left or left.loc[0, "parent_1_id"] == ""
    assert left_events.loc[0, "individual_id"] != right_events.loc[0, "individual_id"]


def test_complete_funnel_reconciles_all_persisted_stages() -> None:
    rows = []
    stages = list(STAGE_ORDER)
    for index, stage in enumerate(stages):
        rows.append({"window": "window_1", "stage": stage, "status": "passed", "individual_id": "one", "expression_hash": "h1", "reject_reason": ""})
        if stage != "final_research_selection":
            rows.append({"window": "window_1", "stage": stage, "status": "rejected", "individual_id": "two", "expression_hash": "h2", "reject_reason": f"reject_{index}"})
    events = pd.DataFrame(rows)
    funnel = build_search_funnel(events).set_index("stage")
    for stage in stages:
        expected = events.loc[events["stage"] == stage]
        assert funnel.at[stage, "input_count"] == len(expected)
        assert funnel.at[stage, "pass_count"] == int((expected["status"] == "passed").sum())
        assert funnel.at[stage, "reject_count"] == int((expected["status"] == "rejected").sum())


def test_hypothesis_counting_keeps_global_and_window_concepts_separate() -> None:
    events = pd.DataFrame([
        {"window": "window_1", "stage": "raw_gp_population", "status": "passed", "individual_id": "a", "expression_hash": "same", "reject_reason": ""},
        {"window": "window_1", "stage": "raw_gp_population", "status": "passed", "individual_id": "b", "expression_hash": "same", "reject_reason": ""},
        {"window": "window_2", "stage": "raw_gp_population", "status": "passed", "individual_id": "c", "expression_hash": "same", "reject_reason": ""},
        {"window": "window_1", "stage": "fast_filter", "status": "passed", "individual_id": "a", "expression_hash": "same", "reject_reason": ""},
        {"window": "window_2", "stage": "fast_filter", "status": "passed", "individual_id": "c", "expression_hash": "same", "reject_reason": ""},
    ])
    stats = build_hypothesis_statistics(events, selected_factor_count=1)
    assert stats["raw_individual_occurrences"] == 3
    assert stats["unique_expressions_global"] == 1
    assert stats["unique_window_expression_pairs"] == 2
    assert stats["per_window"]["window_1"]["unique_expressions"] == 1


def test_research_selection_is_invariant_to_telemetry() -> None:
    gp = GPConfig(
        population_size=12, generations=2, elitism=2, seed=29,
        field_names=("volume", "feature"), disallowed_raw_field_names=(),
    )
    config = AlphaMiningConfig(
        gp=gp,
        portfolio=PortfolioConfig(selected_factor_count=3, min_selected_factor_count=2),
    )

    def selected_output(generator: GPGenerator) -> list[tuple[str, float]]:
        candidates = generator.evolve(_panel(), _Evaluator(), deduplicate=False)
        factors = [
            SelectedFactor(
                expression=candidate.node.describe(), node=candidate.node, direction=1,
                fitness=candidate.evaluation.fitness, metrics={}, complexity=candidate.node.complexity(),
                finite_ratio=1.0, values=pd.Series([float(index + offset) for offset in range(6)]),
            )
            for index, candidate in enumerate(candidates)
        ]
        selected = select_factors_from_pool(factors, config)
        return [(factor.expression, factor.fitness) for factor in selected]

    assert selected_output(_SilentGenerator(gp)) == selected_output(GPGenerator(gp))


def test_operator_effectiveness_observes_unchanged_and_changed_children(monkeypatch) -> None:
    config = GPConfig(
        population_size=6, generations=1, elitism=1, seed=41,
        field_names=("volume", "feature"), disallowed_raw_field_names=(),
        crossover_rate=1.0, subtree_mutation_rate=0.0, point_mutation_rate=0.0, reproduction_rate=0.0,
    )
    unchanged = GPGenerator(config)
    monkeypatch.setattr(unchanged, "subtree_crossover", lambda left, right: left)
    unchanged.evolve(_panel(), _Evaluator(), deduplicate=False)
    unchanged_stats, _ = unchanged.telemetry_frames()
    assert unchanged_stats.loc[1, "crossover_attempted"] > 0
    assert unchanged_stats.loc[1, "crossover_effective"] == 0

    changed = GPGenerator(config)
    monkeypatch.setattr(changed, "subtree_crossover", lambda left, right: rank(left))
    changed.evolve(_panel(), _Evaluator(), deduplicate=False)
    changed_stats, _ = changed.telemetry_frames()
    assert changed_stats.loc[1, "crossover_effective"] > 0
    assert "offspring_duplicate_in_generation_count" in changed_stats.columns


def test_research_only_panel_preparation_does_not_materialize_final_holdout(monkeypatch) -> None:
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-12-31", "2025-06-01"]),
        "symbol": ["AAA", "AAA", "AAA"], "close": [1.0, 1.0, 1.0],
    })
    calls = []

    def tracked_slice(frame, start, end):
        calls.append((start, end))
        return frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()

    monkeypatch.setattr("alpha_mining.run_crypto_workflow.slice_panel_by_date", tracked_slice)
    _, final_panel = prepare_workflow_panels(panel, research_only=True)
    assert final_panel is None
    assert ("2025-06-01", "2026-01-31") not in calls


def test_main_wrapper_persists_normal_exception_path(monkeypatch, tmp_path) -> None:
    ledger = ExperimentLedger(tmp_path)

    def failing_workflow(lifecycle):
        started = ledger.start(configuration={"seed": 5}, metadata={})
        lifecycle.update({"ledger": ledger, "started": started})
        raise RuntimeError("expected failure")

    monkeypatch.setattr("alpha_mining.run_crypto_workflow._main_workflow", failing_workflow)
    with pytest.raises(RuntimeError, match="expected failure"):
        main()
    records = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["started", "failed"]


def test_main_wrapper_preserves_completed_lifecycle(monkeypatch, tmp_path) -> None:
    ledger = ExperimentLedger(tmp_path)

    def completed_workflow(lifecycle):
        started = ledger.start(configuration={"seed": 5}, metadata={})
        lifecycle.update({"ledger": ledger, "started": started})
        ledger.complete(started, results={"selected_factors": ["rank(volume)"]})

    monkeypatch.setattr("alpha_mining.run_crypto_workflow._main_workflow", completed_workflow)
    main()
    records = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["started", "completed"]
