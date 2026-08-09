"""Historical market-cap weights with an explicit static-reference fallback."""

from __future__ import annotations

import pandas as pd


def build_lagged_market_cap_weights(panel: pd.DataFrame) -> pd.DataFrame:
    if "market_cap" not in panel.columns:
        return pd.DataFrame(columns=["date", "symbol", "weight"])
    frame = panel[["date", "symbol", "market_cap"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame = frame.sort_values(["symbol", "date"], kind="mergesort")
    frame["lagged_market_cap"] = frame.groupby("symbol", sort=False)["market_cap"].shift(1)
    frame["lagged_market_cap"] = frame["lagged_market_cap"].where(frame["lagged_market_cap"] > 0.0)
    totals = frame.groupby("date", sort=False)["lagged_market_cap"].transform("sum")
    frame["weight"] = (frame["lagged_market_cap"] / totals).fillna(0.0)
    return frame[["date", "symbol", "weight"]]
