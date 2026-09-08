"""The explicit, shared timing contract used by portfolio backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TradingConvention:
    """Defines when information becomes tradable and how a session is measured."""

    signal_timestamp: str = "close_t"
    execution_timestamp: str = "open_t_plus_1"
    fill_price_rule: str = "next_bar_open"
    holding_period: str = "one_session"
    exit_rule: str = "close_t_plus_1"
    return_interval: str = "open_t_plus_1_to_close_t_plus_1"
    rebalance_frequency: str = "daily"
    cost_model: str = "commission_plus_spread_slippage_impact"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def forward_return(self, panel: pd.DataFrame) -> pd.Series:
        """Return implied by this convention: next open through the session close."""

        next_open = panel.groupby("symbol", sort=False)["open"].shift(-1)
        exit_close = panel.groupby("symbol", sort=False)["close"].shift(-1)
        return (exit_close / next_open) - 1.0


DEFAULT_TRADING_CONVENTION = TradingConvention()


CONTINUOUS_CRYPTO_TRADING_CONVENTION = TradingConvention(
    signal_timestamp="close_t",
    execution_timestamp="open_t_plus_1",
    fill_price_rule="next_bar_open_delta_rebalance",
    holding_period="continuous_across_daily_bar_boundaries",
    exit_rule="trade_only_target_delta_at_next_rebalance",
    return_interval="continuous_mark_to_market_including_close_to_next_open",
    rebalance_frequency="daily",
    cost_model="commission_plus_spread_slippage_impact_on_executed_delta_only",
)
