"""Frozen-factor Phase 3B portfolio boundary.

This module consumes a completed Phase 3A registry. It never generates,
evaluates, rescoring, or selects factors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AlphaMiningConfig, RegistryConfig, SelectedFactor
from .pipeline import backtest_selected_factors, build_alpha_mining_strategy
from .registry import FactorRegistry


@dataclass(frozen=True)
class FrozenFactorSet:
    """Immutable identity and provenance for executable Phase 3A factors."""

    factors: tuple[SelectedFactor, ...]
    expressions: tuple[str, ...]
    directions: tuple[int, ...]
    registry_path: Path
    registry_sha256: str

    def verify_unchanged(self) -> None:
        current_expressions = tuple(str(factor.expression) for factor in self.factors)
        current_directions = tuple(int(factor.direction) for factor in self.factors)
        if current_expressions != self.expressions or current_directions != self.directions:
            raise RuntimeError("Frozen Phase 3A factor identities were modified.")
        if hashlib.sha256(self.registry_path.read_bytes()).hexdigest() != self.registry_sha256:
            raise RuntimeError("Frozen Phase 3A registry changed after it was loaded.")


def load_frozen_factor_set(registry_directory: str | Path) -> FrozenFactorSet:
    """Load executable expressions/directions directly from a Phase 3A registry."""
    root = Path(registry_directory)
    config = AlphaMiningConfig(registry=RegistryConfig(directory=str(root)), live_mode=True)
    registry_path = root / config.registry.pkl_name
    if not registry_path.exists():
        raise FileNotFoundError(f"Frozen Phase 3A registry not found: {registry_path}")
    factors = tuple(FactorRegistry(root).load(config))
    if not factors:
        raise ValueError("Frozen Phase 3A registry contains no selected factors.")
    directions = tuple(int(factor.direction) for factor in factors)
    if any(direction not in {-1, 1} for direction in directions):
        raise ValueError("Frozen executable factor directions must be either -1 or +1.")
    return FrozenFactorSet(
        factors=factors,
        expressions=tuple(str(factor.expression) for factor in factors),
        directions=directions,
        registry_path=registry_path,
        registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    )


def build_phase3b_baseline_config(source: AlphaMiningConfig) -> AlphaMiningConfig:
    """Freeze one unoptimized baseline while preserving existing risk parameters."""
    return replace(
        source,
        regime=replace(source.regime, enabled=False),
        portfolio=replace(
            source.portfolio,
            factor_weight_scheme="equal",
            weighting_scheme="continuous",
            market_neutral=True,
            benchmark_follow_enabled=False,
        ),
        live_mode=True,
        walk_forward_enabled=False,
        save_registry=False,
    )


def build_phase3b_baseline_strategy(
    panel: pd.DataFrame,
    frozen_factors: FrozenFactorSet,
    source_config: AlphaMiningConfig,
):
    """Build the existing strategy with the Phase 3B normalization boundary enabled."""
    frozen_factors.verify_unchanged()
    baseline_config = build_phase3b_baseline_config(source_config)
    return build_alpha_mining_strategy(
        panel,
        baseline_config,
        selected_factors=list(frozen_factors.factors),
        normalize_factor_signals=True,
    )


def run_phase3b_baseline(
    *,
    panel: pd.DataFrame,
    frozen_factors: FrozenFactorSet,
    source_config: AlphaMiningConfig,
    initial_capital: float = 100_000.0,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    """Run the frozen baseline through the repository's existing accounting stack."""
    frozen_factors.verify_unchanged()
    baseline_config = build_phase3b_baseline_config(source_config)
    result = backtest_selected_factors(
        panel=panel,
        config=baseline_config,
        selected_factors=list(frozen_factors.factors),
        initial_capital=initial_capital,
        output_dir=output_dir,
        regime_source_panel=None,
        normalize_factor_signals=True,
    )
    frozen_factors.verify_unchanged()
    return result
