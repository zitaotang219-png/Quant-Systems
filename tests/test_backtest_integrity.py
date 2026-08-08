from __future__ import annotations

import pandas as pd
import pytest

from backtest.event_driven import EventDrivenPortfolioBacktester
from backtest.portfolio_constraints import PortfolioConstraints, validate_target_weights
from execution.cost_model import TransactionCostModel


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "open": [10.0, 10.0, 20.0],
            "close": [10.0, 11.0, 20.0],
        }
    )


def _weights(weight: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "weight": [weight, weight],
        }
    )


def _backtester(cost_model: TransactionCostModel | None = None) -> EventDrivenPortfolioBacktester:
    return EventDrivenPortfolioBacktester(
        initial_capital=100.0,
        cost_model=cost_model or TransactionCostModel(),
        constraints=PortfolioConstraints(
            max_position_size=1.0,
            max_leverage=1.0,
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
        ),
    )


def test_equity_reconciliation() -> None:
    result = _backtester().run(_panel(), _weights())
    assert result.reconciliation["difference"].abs().max() < 1e-9


def test_cash_position_consistency() -> None:
    result = _backtester().run(_panel(), _weights())
    assert (result.timeseries["equity_curve"] - result.timeseries["cash"] - result.timeseries["market_value"]).abs().max() < 1e-9
    assert (result.timeseries["market_value"].abs() < 1e-9).all()


def test_turnover_calculation() -> None:
    result = _backtester().run(_panel(), _weights(0.5))
    assert (result.turnover["entry_turnover"] == 0.5).all()
    assert (result.turnover["exit_turnover"] == 0.5).all()
    assert (result.turnover["turnover"] == 1.0).all()


def test_transaction_cost_application() -> None:
    result = _backtester(TransactionCostModel(commission_bps=100.0)).run(_panel().iloc[:2], _weights().iloc[:1])
    # Buy $50 at 10, sell $55 at 11, and apply 1% commission to both fills.
    assert result.timeseries.iloc[0]["equity_curve"] == pytest.approx(103.95)
    assert result.reconciliation.iloc[0]["fees"] == pytest.approx(1.05)


def test_no_same_bar_execution_bias() -> None:
    weights = _weights()
    weights.loc[weights["date"] == pd.Timestamp("2024-01-02"), "weight"] = -0.5
    result = _backtester().run(_panel(), weights)
    # The price spike on Jan 2 is traded using only the Jan 1 signal; Jan 2's signal executes Jan 3.
    assert result.timeseries.iloc[0]["signal_date"] == pd.Timestamp("2024-01-01")
    assert result.timeseries.iloc[0]["turnover"] == pytest.approx(1.0)
    assert result.timeseries.iloc[1]["signal_date"] == pd.Timestamp("2024-01-02")


def test_position_constraints() -> None:
    constraints = PortfolioConstraints(0.25, 0.5, 0.5, 0.25)
    result = validate_target_weights({"BTCUSDT": 0.30, "ETHUSDT": -0.30}, constraints)
    assert not result.passed
    assert "max_position_size" in result.violations
    assert "max_gross_exposure" in result.violations
    assert "max_net_exposure" not in result.violations
