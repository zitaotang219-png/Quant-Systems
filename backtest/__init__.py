"""Backtesting package."""

from backtest.engine import BacktestEngine, run_backtest, run_cross_sectional_backtest
from backtest.event_driven import ContinuousCryptoPortfolioBacktester, EventDrivenPortfolioBacktester
from backtest.runner import run_backtest_pipeline, run_cross_sectional_backtest_pipeline
from backtest.strategy import BaseStrategy
from backtest.trading_convention import (
    CONTINUOUS_CRYPTO_TRADING_CONVENTION,
    DEFAULT_TRADING_CONVENTION,
    TradingConvention,
)
from backtest.turnover import (
    CONTINUOUS_CRYPTO_TURNOVER_CONVENTION,
    DEFAULT_TURNOVER_CONVENTION,
    ContinuousRebalanceTurnoverConvention,
    TurnoverConvention,
)

__all__ = [
    "BaseStrategy",
    "BacktestEngine",
    "EventDrivenPortfolioBacktester",
    "ContinuousCryptoPortfolioBacktester",
    "TradingConvention",
    "DEFAULT_TRADING_CONVENTION",
    "CONTINUOUS_CRYPTO_TRADING_CONVENTION",
    "TurnoverConvention",
    "DEFAULT_TURNOVER_CONVENTION",
    "ContinuousRebalanceTurnoverConvention",
    "CONTINUOUS_CRYPTO_TURNOVER_CONVENTION",
    "run_backtest",
    "run_cross_sectional_backtest",
    "run_backtest_pipeline",
    "run_cross_sectional_backtest_pipeline",
]
