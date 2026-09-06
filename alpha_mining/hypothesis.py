"""Search-engine-neutral contracts for alpha hypothesis generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .dsl import FactorNode
from .evaluation_types import EvaluationResult


@dataclass
class HypothesisCandidate:
    node: FactorNode
    evaluation: EvaluationResult
    individual_id: str = ""


class HypothesisEvaluator(Protocol):
    def prepare_panel(self, panel: pd.DataFrame) -> pd.DataFrame: ...

    def fast_filter(self, node: FactorNode, panel: pd.DataFrame) -> EvaluationResult: ...


class HypothesisGenerator(Protocol):
    def generate(
        self,
        panel: pd.DataFrame,
        evaluator: HypothesisEvaluator,
        *,
        deduplicate: bool = True,
    ) -> list[HypothesisCandidate]: ...
