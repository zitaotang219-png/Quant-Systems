from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .dsl import FactorNode


VERY_BAD_FITNESS = -1_000_000_000.0


@dataclass
class EvaluationResult:
    node: FactorNode
    values: pd.Series
    finite_ratio: float
    fitness: float
    direction: int
    metrics: dict[str, Any] = field(default_factory=dict)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cumulative_return: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    split_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
