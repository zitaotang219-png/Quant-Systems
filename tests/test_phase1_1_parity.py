from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pandas.testing as pdt
import pytest

from alpha_mining.config import AlphaMiningConfig, EvaluationConfig, PortfolioConfig, SelectedFactor
from alpha_mining.dsl import parse_expression
from alpha_mining.evaluator import FactorEvaluator, _prepare_panel
from alpha_mining.pipeline import build_alpha_mining_strategy
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION
from execution_crypto.paper import run_crypto_paper_trading_with_factors


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in range(8):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for index, symbol in enumerate(("AAA", "BBB", "CCC"), start=1):
            open_price = 100.0 + (day * 2.0) + index
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": open_price * 1.02,
                    "low": open_price * 0.98,
                    "close": open_price * (1.0 + ((index - 2) * 0.01)),
                    "volume": float((day + 1) * index * 100),
                }
            )
    return pd.DataFrame(rows)


def _config() -> AlphaMiningConfig:
    return AlphaMiningConfig(
        evaluation=EvaluationConfig(
            transaction_cost_bps=5.0,
            slippage_bps=3.0,
            long_quantile=0.25,
            short_quantile=0.25,
        ),
        portfolio=PortfolioConfig(
            selected_factor_count=1,
            min_selected_factor_count=1,
            weighting_scheme="bucket",
            position_limit=0.5,
            gross_leverage=1.0,
            turnover_limit=5.0,
            smoothing=0.0,
            market_neutral=True,
        ),
        universe_symbols=("AAA", "BBB", "CCC"),
    )


def _factor() -> SelectedFactor:
    node = parse_expression("volume")
    return SelectedFactor(
        expression="volume",
        node=node,
        direction=1,
        fitness=1.0,
        metrics={},
        complexity=node.complexity(),
        finite_ratio=1.0,
    )


def _run_parity(tmp_path):
    panel = _panel()
    config = _config()
    factor = _factor()
    strategy = build_alpha_mining_strategy(panel, config, selected_factors=[factor])
    backtest = strategy.backtest(
        panel=panel,
        initial_capital=1_000.0,
        transaction_cost_bps=config.evaluation.transaction_cost_bps,
        slippage_bps=config.evaluation.slippage_bps,
    )
    paper = run_crypto_paper_trading_with_factors(
        panel=panel,
        config=config,
        selected_factors=[factor],
        initial_cash=1_000.0,
        output_dir=tmp_path,
        paper_start_date=panel["date"].min(),
        paper_end_date=panel["date"].max(),
    )
    return backtest, paper


def test_backtest_and_paper_same_weights(tmp_path) -> None:
    backtest, paper = _run_parity(tmp_path)
    pdt.assert_frame_equal(
        backtest.weights.sort_values(["date", "symbol"]).reset_index(drop=True),
        paper.target_weights.sort_values(["date", "symbol"]).reset_index(drop=True),
    )


def test_backtest_and_paper_same_costs(tmp_path) -> None:
    backtest, paper = _run_parity(tmp_path)
    assert backtest.trades is not None
    pdt.assert_frame_equal(backtest.trades.reset_index(drop=True), paper.trades.reset_index(drop=True))
    assert float(backtest.trades["total_cost"].sum()) == pytest.approx(float(paper.trades["total_cost"].sum()))


def test_backtest_and_paper_same_equity_path(tmp_path) -> None:
    backtest, paper = _run_parity(tmp_path)
    assert backtest.ledger is not None
    assert backtest.ledger.positions.snapshot() == paper.broker.positions.snapshot()
    pdt.assert_series_equal(
        backtest.timeseries["equity_curve"].reset_index(drop=True),
        paper.equity_curve["equity"].reset_index(drop=True),
        check_names=False,
    )


def test_factor_evaluation_uses_trading_convention() -> None:
    convention = Mock(wraps=DEFAULT_TRADING_CONVENTION)
    prepared = _prepare_panel(_panel(), trading_convention=convention)
    assert "future_return" in prepared
    convention.forward_return.assert_called_once()
    evaluator = FactorEvaluator(trading_convention=convention, min_abs_rank_ic=0.0)
    evaluator.fast_filter(parse_expression("volume"), _panel())
    assert convention.forward_return.call_count >= 2


def test_turnover_consistency_across_modules(tmp_path) -> None:
    backtest, paper = _run_parity(tmp_path)
    assert backtest.turnover_report is not None
    pdt.assert_frame_equal(
        backtest.turnover_report.reset_index(drop=True),
        paper.accounting.turnover.reset_index(drop=True),
    )
