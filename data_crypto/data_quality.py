"""Data validation and cleaning for point-in-time crypto research inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")


@dataclass
class DataQualityResult:
    clean_panel: pd.DataFrame
    report: dict[str, Any]


def validate_crypto_panel(
    panel: pd.DataFrame,
    stale_price_days: int = 3,
    asset_master: pd.DataFrame | None = None,
) -> DataQualityResult:
    missing = [column for column in REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing required data-quality columns: {missing}")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_mask = frame.duplicated(subset=["date", "symbol"], keep="last")
    incomplete_mask = frame[list(REQUIRED_COLUMNS)].isna().any(axis=1) | frame["symbol"].eq("")
    non_positive_price = (frame[["open", "high", "low", "close"]] <= 0.0).any(axis=1)
    invalid_ohlc = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    )
    negative_volume = frame["volume"] < 0.0
    invalid_mask = duplicate_mask | incomplete_mask | non_positive_price | invalid_ohlc | negative_volume

    ordered = frame.loc[~invalid_mask].sort_values(["symbol", "date"], kind="mergesort").copy()
    unchanged = ordered.groupby("symbol", sort=False)["close"].transform(lambda values: values.eq(values.shift()))
    stale_run = unchanged.groupby(ordered["symbol"], sort=False).transform(
        lambda values: values.astype(int).groupby((~values).cumsum()).cumsum()
    )
    stale_mask = stale_run >= max(int(stale_price_days), 1)
    calendar = pd.Index(sorted(frame["date"].dropna().unique()))
    missing_bar_analysis = _classify_missing_bars(ordered, calendar, asset_master)

    report = {
        "row_count": int(len(frame)),
        "clean_row_count": int(len(ordered)),
        "duplicate_date_symbol_rows": int(duplicate_mask.sum()),
        "incomplete_bar_rows": int(incomplete_mask.sum()),
        "zero_or_negative_price_rows": int(non_positive_price.sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "negative_volume_rows": int(negative_volume.sum()),
        "missing_bar_analysis": missing_bar_analysis,
        "stale_price_rows": int(stale_mask.sum()),
        "stale_price_days_threshold": int(stale_price_days),
        "symbols": int(ordered["symbol"].nunique()),
        "date_min": str(ordered["date"].min()) if not ordered.empty else None,
        "date_max": str(ordered["date"].max()) if not ordered.empty else None,
    }
    return DataQualityResult(clean_panel=ordered.reset_index(drop=True), report=report)


def _classify_missing_bars(
    clean_panel: pd.DataFrame,
    calendar: pd.Index,
    asset_master: pd.DataFrame | None,
) -> dict[str, object]:
    """Separate lifecycle absences from unavailable bars inside an asset's life."""
    if calendar.empty:
        return {
            "true_missing_bars": 0,
            "expected_absence": {"before_listing": 0, "after_delisting": 0},
            "coverage_ratio": 1.0,
        }

    lifecycle = _normalize_lifecycle(asset_master, clean_panel, calendar)
    true_missing = 0
    before_listing = 0
    after_delisting = 0
    expected_bars = 0
    available_bars = 0
    for asset in lifecycle.itertuples(index=False):
        symbol = str(asset.symbol)
        listing = pd.Timestamp(asset.listing_date)
        delisting = pd.Timestamp(asset.delisting_date) if pd.notna(asset.delisting_date) else None
        in_lifecycle = calendar >= listing
        if delisting is not None:
            in_lifecycle &= calendar <= delisting
            after_delisting += int((calendar > delisting).sum())
        before_listing += int((calendar < listing).sum())
        expected_dates = calendar[in_lifecycle]
        available_dates = pd.Index(clean_panel.loc[clean_panel["symbol"] == symbol, "date"].dropna().unique())
        present = expected_dates.intersection(available_dates)
        expected_bars += len(expected_dates)
        available_bars += len(present)
        true_missing += len(expected_dates.difference(available_dates))

    return {
        "true_missing_bars": int(true_missing),
        "expected_absence": {
            "before_listing": int(before_listing),
            "after_delisting": int(after_delisting),
        },
        "coverage_ratio": float(available_bars / expected_bars) if expected_bars else 1.0,
    }


def _normalize_lifecycle(
    asset_master: pd.DataFrame | None,
    clean_panel: pd.DataFrame,
    calendar: pd.Index,
) -> pd.DataFrame:
    if asset_master is None:
        # Preserve standalone validation behavior by inferring lifecycle from observed data.
        bounds = clean_panel.groupby("symbol", sort=False)["date"].agg(["min", "max"]).reset_index()
        return bounds.rename(columns={"min": "listing_date", "max": "delisting_date"})

    required = {"symbol", "listing_date", "delisting_date"}
    missing = sorted(required.difference(asset_master.columns))
    if missing:
        raise KeyError(f"Asset master is missing lifecycle columns: {missing}")
    lifecycle = asset_master.loc[:, ["symbol", "listing_date", "delisting_date"]].copy()
    lifecycle["symbol"] = lifecycle["symbol"].astype(str).str.strip()
    lifecycle["listing_date"] = pd.to_datetime(lifecycle["listing_date"], utc=False, errors="coerce")
    lifecycle["delisting_date"] = pd.to_datetime(lifecycle["delisting_date"], utc=False, errors="coerce")
    if lifecycle["symbol"].duplicated().any() or lifecycle["listing_date"].isna().any():
        raise ValueError("Asset master lifecycle metadata is invalid.")
    return lifecycle.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def write_data_quality_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.generic)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")
