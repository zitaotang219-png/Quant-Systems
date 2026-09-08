from __future__ import annotations

import pandas as pd

from backtest.event_driven import (
    ContinuousCryptoPortfolioBacktester,
    EventDrivenPortfolioBacktester,
)
from backtest.portfolio_constraints import PortfolioConstraints
from execution.cost_model import TransactionCostModel


def _panel() -> pd.DataFrame:
    rows = []
    prices = {
        "AAA": [(100.0, 100.0), (110.0, 110.0), (121.0, 121.0)],
        "BBB": [(100.0, 100.0), (90.0, 90.0), (81.0, 81.0)],
    }
    for symbol, bars in prices.items():
        for date, (open_price, close_price) in zip(pd.date_range("2025-01-01", periods=3), bars, strict=True):
            rows.append({"date": date, "symbol": symbol, "open": open_price, "close": close_price})
    return pd.DataFrame(rows)


def _weights() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2025-01-01", periods=3):
        rows.extend(
            [
                {"date": date, "symbol": "AAA", "weight": 0.5},
                {"date": date, "symbol": "BBB", "weight": -0.5},
            ]
        )
    return pd.DataFrame(rows)


def _constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        max_position_size=0.5,
        max_leverage=1.0,
        max_gross_exposure=1.0,
        max_net_exposure=1e-10,
    )


def test_continuous_crypto_holds_across_gap_and_trades_only_delta() -> None:
    continuous = ContinuousCryptoPortfolioBacktester(
        initial_capital=100_000.0,
        cost_model=TransactionCostModel(),
        constraints=_constraints(),
    ).run(_panel(), _weights())
    flat = EventDrivenPortfolioBacktester(
        initial_capital=100_000.0,
        cost_model=TransactionCostModel(),
        constraints=_constraints(),
    ).run(_panel(), _weights())

    assert continuous.timeseries.iloc[1]["gap_pnl"] > 0.0
    assert continuous.timeseries.iloc[-1]["equity_curve"] > flat.timeseries.iloc[-1]["equity_curve"]
    assert continuous.turnover["turnover"].sum() < flat.turnover["turnover"].sum()
    assert set(continuous.trades["reason"]) == {"rebalance_delta"}
    assert all(abs(position.units) > 0.0 for position in continuous.ledger.positions.positions.values())
    assert all(abs(position.units) < 1e-12 for position in flat.ledger.positions.positions.values())
    assert continuous.reconciliation["difference"].abs().max() < 1e-9
    assert (continuous.timeseries["signal_date"] < continuous.timeseries["date"]).all()


def test_continuous_crypto_costs_apply_exactly_to_executed_delta_notional() -> None:
    result = ContinuousCryptoPortfolioBacktester(
        initial_capital=100_000.0,
        cost_model=TransactionCostModel(commission_bps=10.0, slippage_bps=5.0),
        constraints=_constraints(),
    ).run(_panel(), _weights())

    expected = float(result.trades["notional"].sum() * 15.0 / 10_000.0)
    assert abs(float(result.trades["total_cost"].sum()) - expected) < 1e-10
    assert abs(float(result.timeseries["total_cost"].sum()) - expected) < 1e-10
    assert abs(result.ledger.cash.transaction_costs - expected) < 1e-10
    assert result.reconciliation["difference"].abs().max() < 1e-9
    assert not result.constraints.empty
    assert result.constraints["passed"].all()
