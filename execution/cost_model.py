"""Extensible transaction-cost decomposition for executed notional."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TransactionCostBreakdown:
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    impact_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.slippage_cost + self.impact_cost

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["total_cost"] = self.total_cost
        return payload


@dataclass(frozen=True)
class TransactionCostModel:
    """Linear default model; individual components can later be made nonlinear."""

    commission_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0

    def estimate(self, notional: float) -> TransactionCostBreakdown:
        absolute_notional = abs(float(notional))
        scale = absolute_notional / 10_000.0
        return TransactionCostBreakdown(
            commission=scale * self.commission_bps,
            spread_cost=scale * self.spread_bps,
            slippage_cost=scale * self.slippage_bps,
            impact_cost=scale * self.market_impact_bps,
        )
