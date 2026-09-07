from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from alpha_mining.config import (
    AlphaMiningConfig,
    EvaluationConfig,
    PortfolioConfig,
    RegimeConfig,
    RegistryConfig,
    SelectedFactor,
)
from alpha_mining.dsl import field
from alpha_mining.phase3b import (
    build_phase3b_baseline_config,
    build_phase3b_baseline_strategy,
    load_frozen_factor_set,
    run_phase3b_baseline,
)
from alpha_mining.run_phase3b_experiment import (
    build_predeclared_variants,
    load_permitted_raw_panel,
)
from alpha_mining.pipeline import _evaluate_factor_columns
from alpha_mining.portfolio_construction import (
    combine_factor_columns,
    cross_sectional_rank_normalize,
)
from alpha_mining.registry import FactorRegistry


def _factors() -> list[SelectedFactor]:
    return [
        SelectedFactor("signal_a", field("signal_a"), 1, 2.0, {}, 1, 1.0),
        SelectedFactor("signal_b", field("signal_b"), -1, 1.0, {}, 1, 1.0),
    ]


def _source_config(registry_directory: Path) -> AlphaMiningConfig:
    return AlphaMiningConfig(
        evaluation=EvaluationConfig(transaction_cost_bps=8.0, slippage_bps=8.0),
        regime=RegimeConfig(enabled=True),
        portfolio=PortfolioConfig(
            factor_weight_scheme="fitness",
            weighting_scheme="bucket",
            position_limit=0.15,
            turnover_limit=0.20,
            gross_leverage=0.60,
            signal_vol_window=5,
            signal_clip=3.0,
            smoothing=0.0,
            market_neutral=False,
            benchmark_follow_enabled=True,
        ),
        registry=RegistryConfig(directory=str(registry_directory)),
    )


def _panel() -> pd.DataFrame:
    rows = []
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for date_index, date in enumerate(pd.date_range("2024-01-01", periods=12)):
        for symbol_index, symbol in enumerate(symbols):
            open_price = 100.0 + symbol_index + date_index
            move = 0.002 * (((date_index + symbol_index) % 5) - 2)
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": open_price,
                "high": open_price * 1.01,
                "low": open_price * 0.99,
                "close": open_price * (1.0 + move),
                "volume": 1_000.0 + symbol_index,
                "signal_a": float(symbol_index + ((date_index % 3) * (5 - symbol_index))),
                "signal_b": float(100 * (5 - symbol_index) + date_index),
            })
    return pd.DataFrame(rows)


def _write_registry(root: Path, panel: pd.DataFrame) -> None:
    config = _source_config(root)
    FactorRegistry(root).save(_factors(), config, panel)


def test_frozen_registry_identity_and_baseline_configuration(tmp_path: Path) -> None:
    panel = _panel()
    _write_registry(tmp_path, panel)
    frozen = load_frozen_factor_set(tmp_path)
    baseline = build_phase3b_baseline_config(_source_config(tmp_path))

    assert frozen.expressions == ("signal_a", "signal_b")
    assert frozen.directions == (1, -1)
    assert baseline.portfolio.factor_weight_scheme == "equal"
    assert baseline.portfolio.weighting_scheme == "continuous"
    assert baseline.portfolio.market_neutral is True
    assert baseline.portfolio.benchmark_follow_enabled is False
    assert baseline.regime.enabled is False
    assert baseline.portfolio.position_limit == 0.15
    assert baseline.portfolio.gross_leverage == 0.60
    assert baseline.portfolio.turnover_limit == 0.20
    frozen.verify_unchanged()


def test_rank_normalization_is_deterministic_centered_and_point_in_time() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4))
    values = pd.Series([3.0, 1.0, np.nan, 2.0, 2.0, 8.0, 4.0, 6.0])
    first = cross_sectional_rank_normalize(dates, values)
    second = cross_sectional_rank_normalize(dates, values)
    changed_future = values.copy()
    changed_future.iloc[4:] = [1_000.0, -1_000.0, 500.0, -500.0]
    revised = cross_sectional_rank_normalize(dates, changed_future)

    pdt.assert_series_equal(first, second)
    pdt.assert_series_equal(first.iloc[:4], revised.iloc[:4])
    assert first.iloc[2] == 0.0
    assert first.groupby(dates).sum().abs().max() < 1e-12
    assert first.iloc[1] < first.iloc[3] < first.iloc[0]


def test_equal_combination_uses_oriented_normalized_factor_values() -> None:
    panel = _panel().iloc[:6].copy()
    factors = _factors()
    columns = _evaluate_factor_columns(panel, factors, normalize_cross_sectionally=True)
    combined = combine_factor_columns(columns, factors, "equal")
    expected = (columns["signal_a"] + columns["signal_b"]) / 2.0

    pdt.assert_series_equal(combined, expected, check_names=False)
    assert combined.abs().sum() > 0.0
    assert abs(float(combined.mean())) < 1e-12


def test_phase3b_baseline_reuses_constraints_turnover_costs_and_timing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    panel = _panel()
    registry = tmp_path / "registry"
    _write_registry(registry, panel)
    frozen = load_frozen_factor_set(registry)
    source = _source_config(registry)

    def forbidden_selection(*args, **kwargs):
        raise AssertionError("Phase 3B fed portfolio results back into Phase 3A selection")

    monkeypatch.setattr("alpha_mining.pipeline.run_alpha_mining", forbidden_selection)
    monkeypatch.setattr("alpha_mining.pipeline.select_factors_from_pool", forbidden_selection)
    result, _, artifacts = run_phase3b_baseline(
        panel=panel,
        frozen_factors=frozen,
        source_config=source,
        initial_capital=10_000.0,
    )

    weights = pd.DataFrame(artifacts["weights"])
    by_date = weights.groupby("date")["weight"]
    assert by_date.sum().abs().max() < 1e-10
    assert by_date.apply(lambda values: values.abs().sum()).max() <= 0.60 + 1e-10
    assert weights["weight"].abs().max() <= 0.15 + 1e-10
    wide = weights.pivot(index="date", columns="symbol", values="weight").fillna(0.0)
    assert wide.diff().abs().sum(axis=1).iloc[1:].max() <= 0.20 + 1e-10
    assert (result["transaction_cost"] >= 0.0).all()
    assert result["transaction_cost"].sum() > 0.0
    assert (pd.to_datetime(result["date"]) > pd.to_datetime(result["signal_date"])).all()
    assert result.iloc[0]["signal_date"] == pd.Timestamp("2024-01-01")
    assert result.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    frozen.verify_unchanged()


def test_phase3b_strategy_enables_normalization_without_mutating_factors(tmp_path: Path) -> None:
    panel = _panel()
    _write_registry(tmp_path, panel)
    frozen = load_frozen_factor_set(tmp_path)
    strategy = build_phase3b_baseline_strategy(panel, frozen, _source_config(tmp_path))

    assert strategy.normalize_factor_signals is True
    assert tuple(f.expression for f in strategy.selected_factors) == frozen.expressions
    assert tuple(f.direction for f in strategy.selected_factors) == frozen.directions
    frozen.verify_unchanged()


def test_phase3b_experiment_loader_never_materializes_final_holdout(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    path.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2025-05-31,AAA,1,2,0.5,1.5,10\n"
        "2025-06-01,AAA,1,200,0.1,150,999999\n",
        encoding="utf-8",
    )
    panel = load_permitted_raw_panel(path, "2025-05-31")

    assert panel["date"].max() == pd.Timestamp("2025-05-31")
    assert len(panel) == 1
    assert float(panel.iloc[0]["close"]) == 1.5


def test_phase3b_predeclared_variants_change_only_authorized_switches(tmp_path: Path) -> None:
    source = _source_config(tmp_path)
    variants = build_predeclared_variants(source)

    assert tuple(variants) == (
        "A_clean_baseline",
        "B_research_weighted",
        "C_existing_enhanced",
    )
    clean, weighted, enhanced = variants.values()
    assert clean.portfolio.factor_weight_scheme == "equal"
    assert weighted.portfolio.factor_weight_scheme == "fitness"
    assert enhanced.portfolio.factor_weight_scheme == "equal"
    assert clean.regime.enabled is False
    assert weighted.regime.enabled is False
    assert enhanced.regime.enabled is True
    for config in variants.values():
        assert config.portfolio.weighting_scheme == "continuous"
        assert config.portfolio.market_neutral is True
        assert config.portfolio.benchmark_follow_enabled is False
        assert config.portfolio.gross_leverage == source.portfolio.gross_leverage
        assert config.portfolio.position_limit == source.portfolio.position_limit
        assert config.portfolio.turnover_limit == source.portfolio.turnover_limit
