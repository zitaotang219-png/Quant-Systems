from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pandas.testing as pdt
import pytest

from alpha_mining.config import EvaluationConfig
from alpha_mining.evaluator import _prepare_panel
from backtest.event_driven import EventDrivenPortfolioBacktester
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["AAA"] * 3 + ["BBB"] * 3,
            "open": [10.0, 11.0, 12.0, 20.0, 22.0, 24.0],
            "high": [11.0, 12.0, 13.0, 21.0, 23.0, 25.0],
            "low": [9.0, 10.0, 11.0, 19.0, 21.0, 23.0],
            "close": [10.5, 11.5, 12.5, 20.5, 22.5, 24.5],
            "volume": [100.0] * 6,
        }
    )


def test_evaluator_forward_return_equals_trading_convention() -> None:
    panel = _panel()
    prepared = _prepare_panel(panel, trading_convention=DEFAULT_TRADING_CONVENTION)
    pdt.assert_series_equal(
        prepared["future_return"].reset_index(drop=True),
        DEFAULT_TRADING_CONVENTION.forward_return(prepared).reset_index(drop=True),
        check_names=False,
    )


def test_evaluator_configuration_cannot_override_return_label_horizon() -> None:
    assert "future_return_horizon" not in {field.name for field in fields(EvaluationConfig)}
    with pytest.raises(TypeError):
        EvaluationConfig(future_return_horizon=2)  # type: ignore[call-arg]


def test_convention_matches_documented_backtest_holding_assumption() -> None:
    convention = DEFAULT_TRADING_CONVENTION
    assert EventDrivenPortfolioBacktester.__init__.__kwdefaults__ is not None
    assert EventDrivenPortfolioBacktester.__init__.__kwdefaults__["convention"] is DEFAULT_TRADING_CONVENTION
    assert convention.signal_timestamp == "close_t"
    assert convention.execution_timestamp == "open_t_plus_1"
    assert convention.exit_rule == "close_t_plus_1"
    assert convention.return_interval == "open_t_plus_1_to_close_t_plus_1"
