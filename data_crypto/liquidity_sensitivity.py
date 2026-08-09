"""Liquidity-threshold sensitivity analysis for point-in-time universe auditing."""

from __future__ import annotations

import json

import pandas as pd

from universes.point_in_time import PointInTimeUniverseBuilder


DEFAULT_ADV_THRESHOLDS = (0.0, 1_000_000.0, 5_000_000.0, 10_000_000.0)


def build_liquidity_sensitivity(
    panel: pd.DataFrame,
    asset_master: pd.DataFrame,
    *,
    minimum_historical_bars: int,
    minimum_data_completeness: float,
    thresholds: tuple[float, ...] = DEFAULT_ADV_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize membership changes without changing the production threshold."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["adv"] = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(frame["volume"], errors="coerce")
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        universe = PointInTimeUniverseBuilder(
            asset_master=asset_master,
            panel=frame,
            minimum_historical_bars=minimum_historical_bars,
            liquidity_threshold=float(threshold),
            minimum_data_completeness=minimum_data_completeness,
        ).build_over_time()
        eligible = universe.loc[universe["eligible"], ["date", "symbol"]]
        eligible_adv = frame.merge(eligible, on=["date", "symbol"], how="inner")["adv"]
        exclusions = universe.loc[~universe["eligible"]]
        liquidity_exclusions = exclusions.loc[exclusions["exclusion_reason"] == "insufficient_liquidity"]
        daily_size = eligible.groupby("date", sort=False)["symbol"].nunique()
        rows.append(
            {
                "threshold": float(threshold),
                "average_universe_size": float(daily_size.mean()) if not daily_size.empty else 0.0,
                "minimum_ADV": float(eligible_adv.min()) if not eligible_adv.empty else 0.0,
                "removed_assets": ",".join(sorted(liquidity_exclusions["symbol"].astype(str).unique())),
                "removal_reasons": json.dumps(
                    exclusions.groupby("exclusion_reason", sort=True).size().to_dict(), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)
