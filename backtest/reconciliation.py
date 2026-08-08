"""Accounting reconciliation helpers for daily portfolio state."""

from __future__ import annotations

import pandas as pd


def build_reconciliation_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["date", "starting_equity", "gross_pnl", "fees", "slippage", "ending_equity", "difference"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame[columns]
