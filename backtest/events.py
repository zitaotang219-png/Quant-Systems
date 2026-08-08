"""Traceable events emitted by the daily event-driven portfolio backtester."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    timestamp: pd.Timestamp
    prices: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class SignalEvent:
    timestamp: pd.Timestamp
    target_weights: Mapping[str, float]


@dataclass(frozen=True)
class OrderEvent:
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    reason: str


@dataclass(frozen=True)
class FillEvent:
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    fee: float
    slippage: float
    reason: str


@dataclass(frozen=True)
class PositionEvent:
    timestamp: pd.Timestamp
    symbol: str
    units: float
    market_value: float


@dataclass(frozen=True)
class AccountingEvent:
    timestamp: pd.Timestamp
    starting_equity: float
    gross_pnl: float
    costs: float
    ending_equity: float


def event_to_dict(event: object) -> dict[str, Any]:
    """Convert an event to a JSON-safe payload for audit artifacts."""

    payload = asdict(event)
    for key, value in payload.items():
        if isinstance(value, pd.Timestamp):
            payload[key] = value.isoformat()
    payload["event_type"] = type(event).__name__
    return payload
