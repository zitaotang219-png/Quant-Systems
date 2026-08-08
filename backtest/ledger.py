"""Cash, position, and trade ledgers used to reconstruct portfolio equity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CashLedger:
    balance: float
    deposits: float = 0.0
    withdrawals: float = 0.0
    transaction_costs: float = 0.0

    def deposit(self, amount: float) -> None:
        self.balance += float(amount)
        self.deposits += float(amount)

    def withdraw(self, amount: float) -> None:
        self.balance -= float(amount)
        self.withdrawals += float(amount)

    def apply_trade(self, signed_notional: float, cost: float) -> None:
        # A buy consumes cash; a sell, including a short sale, adds cash.
        self.balance -= float(signed_notional) + float(cost)
        self.transaction_costs += float(cost)


@dataclass
class Position:
    symbol: str
    units: float = 0.0
    average_price: float = 0.0
    last_price: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(self, quantity: float, price: float) -> None:
        quantity = float(quantity)
        price = float(price)
        prior_units = self.units
        if abs(prior_units) < 1e-12 or prior_units * quantity > 0.0:
            new_units = prior_units + quantity
            if abs(new_units) > 1e-12:
                self.average_price = (
                    (abs(prior_units) * self.average_price) + (abs(quantity) * price)
                ) / abs(new_units)
            else:
                self.average_price = 0.0
        else:
            closed_units = min(abs(prior_units), abs(quantity))
            self.realized_pnl += closed_units * (price - self.average_price) * (1.0 if prior_units > 0 else -1.0)
            new_units = prior_units + quantity
            if prior_units * new_units < 0.0:
                self.average_price = price
            elif abs(new_units) < 1e-12:
                self.average_price = 0.0
        self.units = new_units
        self.last_price = price

    @property
    def market_value(self) -> float:
        return self.units * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return self.units * (self.last_price - self.average_price)


@dataclass
class PositionLedger:
    positions: dict[str, Position] = field(default_factory=dict)

    def apply_fill(self, symbol: str, quantity: float, price: float) -> Position:
        position = self.positions.setdefault(symbol, Position(symbol=symbol))
        position.apply_fill(quantity, price)
        return position

    def mark_to_market(self, prices: dict[str, float]) -> None:
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.last_price = float(prices[symbol])

    @property
    def market_value(self) -> float:
        return sum(position.market_value for position in self.positions.values())

    def snapshot(self) -> list[dict[str, float | str]]:
        return [asdict(position) | {"market_value": position.market_value, "unrealized_pnl": position.unrealized_pnl} for position in self.positions.values()]


@dataclass
class TradeLedger:
    trades: list[dict[str, float | str]] = field(default_factory=list)

    def record(
        self,
        *,
        timestamp: object,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        slippage: float,
        spread: float = 0.0,
        market_impact: float = 0.0,
        reason: str,
    ) -> None:
        self.trades.append(
            {
                "timestamp": str(timestamp),
                "symbol": symbol,
                "side": side,
                "quantity": abs(float(quantity)),
                "price": float(price),
                "notional": abs(float(quantity) * float(price)),
                "fee": float(fee),
                "slippage": float(slippage),
                "spread": float(spread),
                "market_impact": float(market_impact),
                "total_cost": float(fee + slippage + spread + market_impact),
                "reason": reason,
            }
        )
