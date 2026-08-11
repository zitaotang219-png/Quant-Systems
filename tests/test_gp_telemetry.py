from __future__ import annotations

import pandas as pd

from alpha_mining.config import GPConfig
from alpha_mining.evaluator import EvaluationResult
from alpha_mining.gp_generator import GPGenerator


class _Evaluator:
    def fast_filter(self, node, panel):
        # Deterministic surrogate: telemetry must not affect generator control flow.
        return EvaluationResult(node=node, values=pd.Series(dtype=float), finite_ratio=1.0, fitness=float(-node.complexity()), direction=1)


class _SilentGenerator(GPGenerator):
    def _record_generation(self, *args, **kwargs) -> None:
        return None


def test_telemetry_does_not_change_seeded_gp_search() -> None:
    config = GPConfig(population_size=12, generations=2, elitism=3, seed=123, field_names=("volume", "feature"), disallowed_raw_field_names=())
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "symbol": ["AAA"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "feature": [1.0]})
    baseline = _SilentGenerator(config).evolve(panel, _Evaluator())
    instrumented_generator = GPGenerator(config)
    instrumented = instrumented_generator.evolve(panel, _Evaluator())
    assert [item.node.describe() for item in instrumented] == [item.node.describe() for item in baseline]
    assert [item.evaluation.fitness for item in instrumented] == [item.evaluation.fitness for item in baseline]
    assert [item.node.describe() for item in instrumented[:4]] == [item.node.describe() for item in baseline[:4]]
    generation, audit = instrumented_generator.telemetry_frames()
    assert len(generation) == config.generations + 1
    assert not audit.empty
