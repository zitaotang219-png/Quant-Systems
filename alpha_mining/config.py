from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 80
    generations: int = 8
    tournament_size: int = 5
    elitism: int = 8
    max_depth: int = 5
    init_max_depth: int = 4
    crossover_rate: float = 0.45
    subtree_mutation_rate: float = 0.2
    point_mutation_rate: float = 0.15
    reproduction_rate: float = 0.2
    seed: int = 42
    field_names: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    disallowed_raw_field_names: tuple[str, ...] = ("open", "high", "low", "close")
    constant_range: tuple[float, float] = (-2.0, 2.0)
    periods_choices: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    window_choices: tuple[int, ...] = (3, 5, 10, 20, 30)
    wrap_final_with_rank_or_zscore: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    min_finite_ratio: float = 0.6
    min_std: float = 1e-6
    min_abs_rank_ic: float = 0.005
    max_turnover: float = 2.5
    max_allowed_drawdown: float = 0.25
    ic_sign_tolerance: float = 0.002
    annualization: int = 252


@dataclass(frozen=True)
class FitnessConfig:
    fast_rank_ic_weight: float = 10.0
    validation_ic_weight: float = 35.0
    sharpe_weight: float = 0.75
    cumulative_return_weight: float = 6.0
    excess_return_weight: float = 10.0
    stability_weight: float = 4.0
    bear_return_weight: float = 0.0
    bear_sharpe_weight: float = 0.0
    turnover_penalty: float = 4.0
    drawdown_penalty: float = 18.0
    complexity_penalty: float = 0.08


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = False
    alpha: float = 0.1
    threshold: float = 0.2
    trend_window: int = 20
    volatility_window: int = 20
    zscore_window: int = 252
    return_column: str = "daily_return"
    use_style_multipliers: bool = True
    factor_regime_weight_overrides: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioConfig:
    selected_factor_count: int = 8
    min_selected_factor_count: int = 4
    max_pairwise_correlation: float = 0.8
    factor_weight_scheme: str = "equal"
    weighting_scheme: str = "continuous"
    position_limit: float = 0.08
    turnover_limit: float = 1.5
    gross_leverage: float = 0.8
    signal_vol_window: int = 20
    signal_clip: float = 3.0
    smoothing: float = 0.7
    market_neutral: bool = True
    benchmark_follow_enabled: bool = False
    benchmark_follow_btc_symbol: str = "BTCUSDT"
    benchmark_follow_btc_weight: float = 0.5
    regime_benchmark_blend: dict[str, float] = field(default_factory=dict)
    regime_benchmark_direction: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistryConfig:
    directory: str = "alpha_mining_registry"
    pkl_name: str = "selected_factors.pkl"
    csv_name: str = "selected_factors.csv"
    metadata_name: str = "metadata.json"


@dataclass(frozen=True)
class AlphaMiningConfig:
    gp: GPConfig = field(default_factory=GPConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    live_mode: bool = False
    fast_filter_keep: int = 24
    deep_eval_keep: int = 12
    walk_forward_enabled: bool = False
    walk_forward_train_fraction: float = 0.6
    walk_forward_validation_fraction: float = 0.2
    walk_forward_backtest_fraction: float = 0.2
    walk_forward_min_folds: int = 2
    deduplicate_expressions: bool = True
    save_registry: bool = True
    universe_symbols: tuple[str, ...] = ()

    def registry_dir(self) -> Path:
        return Path(self.registry.directory)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedFactor:
    expression: str
    node: Any
    direction: int
    fitness: float
    metrics: dict[str, Any]
    complexity: int
    finite_ratio: float
    values: pd.Series | None = None

    def summary_row(self) -> dict[str, Any]:
        payload = {
            "expression": self.expression,
            "direction": self.direction,
            "fitness": self.fitness,
            "complexity": self.complexity,
            "finite_ratio": self.finite_ratio,
        }
        payload.update(self.metrics)
        return payload
