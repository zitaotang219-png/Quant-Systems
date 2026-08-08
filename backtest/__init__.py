"""Backtesting package."""

from backtest.engine import BacktestEngine, run_backtest, run_cross_sectional_backtest
from backtest.event_driven import EventDrivenPortfolioBacktester
from backtest.runner import run_backtest_pipeline, run_cross_sectional_backtest_pipeline
from backtest.strategy import BaseStrategy
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION, TradingConvention

__all__ = [
    "BaseStrategy",
    "BacktestEngine",
    "EventDrivenPortfolioBacktester",
    "TradingConvention",
    "DEFAULT_TRADING_CONVENTION",
    "run_backtest",
    "run_cross_sectional_backtest",
    "run_backtest_pipeline",
    "run_cross_sectional_backtest_pipeline",
]
