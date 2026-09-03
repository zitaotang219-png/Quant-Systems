from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alpha_mining.config import GPConfig
from alpha_mining.evaluator import FactorEvaluator
from alpha_mining.gp_generator import GPGenerator
from alpha_mining.dsl import field


def _panel() -> pd.DataFrame:
    rows = []
    for symbol, base in (("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0)):
        for offset, date in enumerate(pd.date_range("2024-01-01", periods=8)):
            rows.append({"date": date, "symbol": symbol, "open": base + offset, "high": base + offset + 1, "low": base + offset - 1, "close": base + offset + 0.5, "volume": base * 100 + offset})
    return pd.DataFrame(rows)


def test_prepared_panel_preserves_fast_evaluation() -> None:
    evaluator = FactorEvaluator(min_abs_rank_ic=0.0)
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
    final_population = generator.evolve(_panel(), FactorEvaluator(min_abs_rank_ic=0.0))
    archive = generator.archive_candidates()
    assert len({candidate.node.describe() for candidate in archive}) == len(archive)
    assert {candidate.node.describe() for candidate in final_population}.issubset({candidate.node.describe() for candidate in archive})