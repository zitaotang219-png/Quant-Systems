"""Daily event-driven accounting for cross-sectional alpha portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.events import AccountingEvent, FillEvent, MarketEvent, OrderEvent, PositionEvent, SignalEvent, event_to_dict
from backtest.ledger import CashLedger, PortfolioLedger
from backtest.portfolio_constraints import PortfolioConstraints, validate_target_weights
from backtest.reconciliation import build_reconciliation_frame
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION, TradingConvention
from backtest.turnover import DEFAULT_TURNOVER_CONVENTION, TurnoverConvention
from execution.cost_model import TransactionCostModel


@dataclass
class EventDrivenBacktestResult:
    timeseries: pd.DataFrame
    trades: pd.DataFrame
    reconciliation: pd.DataFrame
    turnover: pd.DataFrame
    constraints: pd.DataFrame
    events: list[dict[str, Any]]
    ledger: PortfolioLedger


class EventDrivenPortfolioBacktester:
    """Executes close signals at the next open and closes the book that session."""

    def __init__(
        self,
        *,
        initial_capital: float,
        cost_model: TransactionCostModel,
        constraints: PortfolioConstraints,
        convention: TradingConvention = DEFAULT_TRADING_CONVENTION,
        turnover_convention: TurnoverConvention = DEFAULT_TURNOVER_CONVENTION,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cost_model = cost_model
        self.constraints = constraints
        self.convention = convention
        self.turnover_convention = turnover_convention

    def run(self, panel: pd.DataFrame, target_weights: pd.DataFrame) -> EventDrivenBacktestResult:
        required = {"date", "symbol", "open", "close"}
        missing = required.difference(panel.columns)
        if missing:
            raise KeyError(f"Panel is missing required execution columns: {sorted(missing)}")
        if target_weights.empty:
            return self._empty_result()

        prices = panel[["date", "symbol", "open", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"], utc=False)
        prices["symbol"] = prices["symbol"].astype(str)
        prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.dropna(subset=["open", "close"]).sort_values(["date", "symbol"], kind="mergesort")

        signals = target_weights[["date", "symbol", "weight"]].copy()
        signals["date"] = pd.to_datetime(signals["date"], utc=False)
        signals["symbol"] = signals["symbol"].astype(str)
        signals["weight"] = pd.to_numeric(signals["weight"], errors="coerce").fillna(0.0)

        dates = list(pd.Index(prices["date"].drop_duplicates()).sort_values())
        signal_by_date = {
            date: group.set_index("symbol")["weight"].to_dict()
            for date, group in signals.groupby("date", sort=False)
        }
        ledger = PortfolioLedger(cash=CashLedger(balance=self.initial_capital))
        cash = ledger.cash
        positions = ledger.positions
        trades = ledger.trades
        event_log: list[dict[str, Any]] = []
        time_rows: list[dict[str, Any]] = []
        reconciliation_rows: list[dict[str, object]] = []
        turnover_rows: list[dict[str, object]] = []
        constraint_rows: list[dict[str, object]] = []

        # Signal t is known only at its close and is first executable on market date t+1.
        for index in range(1, len(dates)):
            signal_date = dates[index - 1]
            execution_date = dates[index]
            raw_weights = signal_by_date.get(signal_date, {})
            day = prices.loc[prices["date"] == execution_date].set_index("symbol")
            weights = {symbol: float(weight) for symbol, weight in raw_weights.items() if symbol in day.index}
            result = validate_target_weights(weights, self.constraints)
            constraint_rows.append({"date": execution_date, "signal_date": signal_date, **result.to_dict()})
            if not result.passed:
                raise ValueError(f"Portfolio constraints violated on {execution_date.date()}: {result.violations}")

            market_prices = {
                symbol: {"open": float(row["open"]), "close": float(row["close"])}
                for symbol, row in day.iterrows()
            }
            event_log.append(event_to_dict(MarketEvent(timestamp=execution_date, prices=market_prices)))
            event_log.append(event_to_dict(SignalEvent(timestamp=signal_date, target_weights=weights)))
            starting_equity = cash.balance + positions.market_value
            entry_notional = 0.0
            exit_notional = 0.0
            entry_cost = 0.0
            exit_cost = 0.0
            commission_cost = 0.0
            execution_cost = 0.0
            gross_pnl = 0.0

            for symbol, weight in weights.items():
                if abs(weight) <= 1e-12:
                    continue
                open_price = float(day.at[symbol, "open"])
                close_price = float(day.at[symbol, "close"])
                signed_notional = starting_equity * weight
                quantity = signed_notional / open_price
                costs = self.cost_model.estimate(signed_notional)
                event_log.append(event_to_dict(OrderEvent(execution_date, symbol, quantity, open_price, "entry")))
                cash.apply_trade(signed_notional, costs.total_cost)
                position = positions.apply_fill(symbol, quantity, open_price)
                trades.record(
                    timestamp=execution_date, symbol=symbol, side="BUY" if quantity > 0 else "SELL_SHORT",
                    quantity=quantity, price=open_price, fee=costs.commission,
                    slippage=costs.slippage_cost, spread=costs.spread_cost,
                    market_impact=costs.impact_cost, reason="entry",
                )
                event_log.append(event_to_dict(FillEvent(execution_date, symbol, quantity, open_price, costs.commission, costs.slippage_cost, "entry")))
                event_log.append(event_to_dict(PositionEvent(execution_date, symbol, position.units, position.market_value)))
                entry_notional += abs(signed_notional)
                entry_cost += costs.total_cost
                commission_cost += costs.commission
                execution_cost += costs.spread_cost + costs.slippage_cost + costs.impact_cost
                gross_pnl += quantity * (close_price - open_price)

            positions.mark_to_market({symbol: float(row["close"]) for symbol, row in day.iterrows()})
            for symbol, position in list(positions.positions.items()):
                if abs(position.units) <= 1e-12:
                    continue
                close_price = float(day.at[symbol, "close"])
                quantity = -position.units
                signed_notional = quantity * close_price
                costs = self.cost_model.estimate(signed_notional)
                event_log.append(event_to_dict(OrderEvent(execution_date, symbol, quantity, close_price, "session_exit")))
                cash.apply_trade(signed_notional, costs.total_cost)
                position = positions.apply_fill(symbol, quantity, close_price)
                trades.record(
                    timestamp=execution_date, symbol=symbol, side="SELL" if quantity < 0 else "BUY_TO_COVER",
                    quantity=quantity, price=close_price, fee=costs.commission,
                    slippage=costs.slippage_cost, spread=costs.spread_cost,
                    market_impact=costs.impact_cost, reason="session_exit",
                )
                event_log.append(event_to_dict(FillEvent(execution_date, symbol, quantity, close_price, costs.commission, costs.slippage_cost, "session_exit")))
                event_log.append(event_to_dict(PositionEvent(execution_date, symbol, position.units, position.market_value)))
                exit_cost += costs.total_cost
                exit_notional += abs(signed_notional)
                commission_cost += costs.commission
                execution_cost += costs.spread_cost + costs.slippage_cost + costs.impact_cost

            ending_equity = cash.balance + positions.market_value
            total_cost = entry_cost + exit_cost
            difference = ending_equity - (starting_equity + gross_pnl - total_cost)
            event_log.append(event_to_dict(AccountingEvent(execution_date, starting_equity, gross_pnl, total_cost, ending_equity)))
            reconciliation_rows.append(
                {
                    "date": execution_date,
                    "starting_equity": starting_equity,
                    "gross_pnl": gross_pnl,
                    "fees": commission_cost,
                    "slippage": execution_cost,
                    "ending_equity": ending_equity,
                    "difference": difference,
                }
            )
            turnover_rows.append({
                "date": execution_date,
                "signal_date": signal_date,
                **self.turnover_convention.report(starting_equity, entry_notional, exit_notional),
            })
            time_rows.append(
                {
                    "date": execution_date,
                    "signal_date": signal_date,
                    "gross_pnl": gross_pnl,
                    "total_cost": total_cost,
                    "gross_return": 0.0 if starting_equity == 0.0 else gross_pnl / starting_equity,
                    "transaction_cost": 0.0 if starting_equity == 0.0 else total_cost / starting_equity,
                    "net_return": 0.0 if starting_equity == 0.0 else (ending_equity / starting_equity) - 1.0,
                    "equity_curve": ending_equity,
                    "cash": cash.balance,
                    "market_value": positions.market_value,
                    "turnover": turnover_rows[-1]["turnover"],
                    "pnl": ending_equity - starting_equity,
                }
            )

        timeseries = pd.DataFrame(time_rows)
        if not timeseries.empty:
            timeseries["drawdown"] = (timeseries["equity_curve"] / timeseries["equity_curve"].cummax()) - 1.0
        else:
            timeseries["drawdown"] = pd.Series(dtype=float)
        return EventDrivenBacktestResult(
            timeseries=timeseries,
            trades=pd.DataFrame(trades.trades),
            reconciliation=build_reconciliation_frame(reconciliation_rows),
            turnover=pd.DataFrame(turnover_rows),
            constraints=pd.DataFrame(constraint_rows),
            events=event_log,
            ledger=ledger,
        )

    def _empty_result(self) -> EventDrivenBacktestResult:
        return EventDrivenBacktestResult(
            timeseries=pd.DataFrame(), trades=pd.DataFrame(), reconciliation=pd.DataFrame(),
            turnover=pd.DataFrame(), constraints=pd.DataFrame(), events=[],
            ledger=PortfolioLedger(cash=CashLedger(balance=self.initial_capital)),
        )
