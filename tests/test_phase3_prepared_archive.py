from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alpha_mining.config import AlphaMiningConfig, GPConfig
from alpha_mining.evaluation_types import EvaluationResult
from alpha_mining.gp_generator import GPCandidate, GPGenerator
from alpha_mining.dsl import field, rank, zscore
from alpha_mining.pipeline import _deep_evaluate_population, build_candidate_factor_pool
from alpha_mining.research_evaluator import FactorResearchEvaluator


def _panel() -> pd.DataFrame:
    rows = []
    for symbol, base in (("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0)):
        for offset, date in enumerate(pd.date_range("2024-01-01", periods=8)):
            rows.append({"date": date, "symbol": symbol, "open": base + offset, "high": base + offset + 1, "low": base + offset - 1, "close": base + offset + 0.5, "volume": base * 100 + offset})
    return pd.DataFrame(rows)


def test_prepared_panel_preserves_fast_evaluation() -> None:
    evaluator = FactorResearchEvaluator(min_abs_rank_ic=0.0)
    raw = evaluator.fast_filter(field("volume"), _panel())
    prepared = evaluator.prepare_panel(_panel())
    reused = evaluator.fast_filter(field("volume"), prepared)
    assert raw.fitness == reused.fitness
    assert raw.direction == reused.direction
    assert raw.metrics == reused.metrics
    pdt.assert_series_equal(raw.values, reused.values)


def test_archive_is_unique_and_does_not_change_final_population() -> None:
    config = GPConfig(population_size=8, generations=2, elitism=2, seed=17, field_names=("volume",), disallowed_raw_field_names=())
    generator = GPGenerator(config)
    final_population = generator.evolve(_panel(), FactorResearchEvaluator(min_abs_rank_ic=0.0))
    archive = generator.archive_candidates()
    assert len({candidate.node.describe() for candidate in archive}) == len(archive)
    assert {candidate.node.describe() for candidate in final_population}.issubset({candidate.node.describe() for candidate in archive})


def _candidate(node, fitness: float = 1.0) -> GPCandidate:
    return GPCandidate(
        node=node,
        evaluation=EvaluationResult(
            node=node,
            values=pd.Series(dtype=float),
            finite_ratio=1.0,
            fitness=fitness,
            direction=1,
        ),
    )


def test_candidate_pool_screens_the_cross_generation_archive(monkeypatch) -> None:
    final_candidate = _candidate(field("close"))
    archived_candidate = _candidate(field("volume"))
    captured = {}

    class FakeGenerator:
        def __init__(self, config):
            pass

        def generate(self, panel, evaluator, deduplicate=True):
            return [archived_candidate]

    def capture_deep(**kwargs):
        captured["candidates"] = kwargs["candidates"]
        return []

    monkeypatch.setattr("alpha_mining.pipeline.GPGenerator", FakeGenerator)
    monkeypatch.setattr("alpha_mining.pipeline._build_research_evaluator", lambda config: object())
    monkeypatch.setattr("alpha_mining.pipeline._deep_evaluate_population", capture_deep)
    build_candidate_factor_pool(_panel(), AlphaMiningConfig(), pool_limit=1, fast_keep=1, deep_keep=1)
    assert captured["candidates"] == [archived_candidate]


def test_deep_evaluation_calls_are_strictly_capped() -> None:
    nodes = [
        field("volume"),
        rank(field("volume")),
        zscore(field("volume")),
        rank(zscore(field("volume"))),
    ]
    calls = []

    class RecordingEvaluator:
        def prepare_panel(self, panel):
            return panel

        def create_context(self, panel, **kwargs):
            return panel

        def evaluate_context(self, node, context, mode):
            calls.append(node.describe())
            return EvaluationResult(
                node=node,
                values=pd.Series(dtype=float),
                finite_ratio=1.0,
                fitness=1.0,
                direction=1,
            )

    results = _deep_evaluate_population(
        candidates=[_candidate(node) for node in nodes],
        panel=_panel(),
        evaluator=RecordingEvaluator(),
        keep=2,
        fast_keep=4,
    )
    assert len(calls) == 2
    assert len(results) == 2
