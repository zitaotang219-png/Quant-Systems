"""Asset lifecycle aware universe construction for historical research."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


ASSET_MASTER_COLUMNS = ("symbol", "exchange", "listing_date", "delisting_date", "status")


def load_asset_master(path: str | Path) -> pd.DataFrame:
    master = pd.read_csv(path)
    missing = [column for column in ASSET_MASTER_COLUMNS if column not in master.columns]
    if missing:
        raise KeyError(f"Asset master is missing required columns: {missing}")
    master = master.copy()
    master["symbol"] = master["symbol"].astype(str).str.strip()
    master["listing_date"] = pd.to_datetime(master["listing_date"], utc=False, errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], utc=False, errors="coerce")
    if master["symbol"].duplicated().any():
        raise ValueError("Asset master contains duplicate symbols.")
    if master["listing_date"].isna().any():
        raise ValueError("Asset master contains invalid listing_date values.")
    return master.sort_values("symbol", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class PointInTimeUniverseBuilder:
    asset_master: pd.DataFrame
    panel: pd.DataFrame
    minimum_historical_bars: int = 1
    liquidity_threshold: float = 0.0
    minimum_data_completeness: float = 1.0
    _universe_history: pd.DataFrame = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized_panel = _normalize_panel(self.panel)
        object.__setattr__(self, "panel", normalized_panel)
        object.__setattr__(self, "_universe_history", self._build_history())

    def build_universe(self, date: str | pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(date)
        rows = self._universe_history.loc[self._universe_history["date"] == as_of].copy()
        if rows.empty:
            raise ValueError(f"No market calendar row is available for {as_of.date()}.")
        return rows.reset_index(drop=True)

    def build_over_time(self, dates: pd.Series | None = None) -> pd.DataFrame:
        values = dates if dates is not None else self.panel["date"]
        unique_dates = pd.to_datetime(values, utc=False).dropna().unique()
        return self._universe_history.loc[self._universe_history["date"].isin(unique_dates)].reset_index(drop=True)

    def _build_history(self) -> pd.DataFrame:
        calendar = pd.Index(sorted(self.panel["date"].dropna().unique()), name="date")
        rows: list[pd.DataFrame] = []
        required = [column for column in ("open", "high", "low", "close", "volume") if column in self.panel.columns]
        history_bars = max(int(self.minimum_historical_bars), 1)
        for _, asset in self.asset_master.iterrows():
            symbol = str(asset["symbol"])
            asset_panel = self.panel.loc[self.panel["symbol"] == symbol].set_index("date").reindex(calendar)
            current_exists = asset_panel["symbol"].notna()
            complete = asset_panel[required].notna().all(axis=1) if required else pd.Series(False, index=calendar)
            observed_history = current_exists.cumsum()
            completeness = complete.astype(float).rolling(history_bars, min_periods=history_bars).mean().fillna(0.0)
            dollar_volume = pd.to_numeric(asset_panel["close"], errors="coerce") * pd.to_numeric(asset_panel["volume"], errors="coerce")
            median_liquidity = dollar_volume.rolling(history_bars, min_periods=history_bars).median()
            listing = pd.Timestamp(asset["listing_date"])
            delisting = asset["delisting_date"]
            reasons = pd.Series("", index=calendar, dtype="object")
            reasons.loc[calendar < listing] = "before_listing"
            if pd.notna(delisting):
                reasons.loc[calendar > pd.Timestamp(delisting)] = "after_delisting"
            reasons.loc[(reasons == "") & ~current_exists] = "missing_current_bar"
            reasons.loc[(reasons == "") & (observed_history < history_bars)] = "insufficient_history"
            reasons.loc[(reasons == "") & (completeness < self.minimum_data_completeness)] = "incomplete_history"
            if self.liquidity_threshold > 0.0:
                reasons.loc[(reasons == "") & (median_liquidity < self.liquidity_threshold)] = "insufficient_liquidity"
            rows.append(pd.DataFrame({"date": calendar, "symbol": symbol, "eligible": reasons.eq(""), "exclusion_reason": reasons.to_numpy()}))
        return pd.concat(rows, ignore_index=True)


def build_universe(
    date: str | pd.Timestamp,
    *,
    asset_master: pd.DataFrame,
    panel: pd.DataFrame,
    minimum_historical_bars: int = 1,
    liquidity_threshold: float = 0.0,
    minimum_data_completeness: float = 1.0,
) -> pd.DataFrame:
    return PointInTimeUniverseBuilder(
        asset_master=asset_master,
        panel=_normalize_panel(panel),
        minimum_historical_bars=minimum_historical_bars,
        liquidity_threshold=liquidity_threshold,
        minimum_data_completeness=minimum_data_completeness,
    ).build_universe(date)


def filter_panel_to_point_in_time_universe(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    eligible = universe.loc[universe["eligible"], ["date", "symbol"]].copy()
    eligible["date"] = pd.to_datetime(eligible["date"], utc=False)
    frame = _normalize_panel(panel)
    return frame.merge(eligible, on=["date", "symbol"], how="inner").sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def build_universe_report(universe: pd.DataFrame, asset_master: pd.DataFrame, panel: pd.DataFrame) -> str:
    if universe.empty:
        return "# Point-in-Time Universe Report\n\nNo eligible universe rows were produced.\n"
    ordered = universe.sort_values(["date", "symbol"], kind="mergesort")
    eligible = ordered.loc[ordered["eligible"]]
    size = eligible.groupby("date", sort=False)["symbol"].nunique()
    added_removed: list[str] = []
    previous: set[str] = set()
    for date, day in eligible.groupby("date", sort=False):
        current = set(day["symbol"].astype(str))
        added = sorted(current - previous)
        removed = sorted(previous - current)
        if added or removed:
            added_removed.append(f"- {pd.Timestamp(date).date()}: added={added or ['none']}; removed={removed or ['none']}")
        previous = current
    exclusions = ordered.loc[~ordered["eligible"]].groupby("exclusion_reason", sort=True).size().to_dict()
    delisted = asset_master.loc[asset_master["delisting_date"].notna(), ["symbol", "delisting_date"]]
    liquidity = (pd.to_numeric(panel["close"], errors="coerce") * pd.to_numeric(panel["volume"], errors="coerce")).describe()
    lines = [
        "# Point-in-Time Universe Report",
        "",
        "## Methodology",
        "",
        "An asset is eligible only on or after its listing date and through its delisting date, with required history, complete bars, and liquidity checks evaluated using data available on that date.",
        "",
        "## Universe Size",
        "",
        f"- Dates: {len(size)}",
        f"- Minimum eligible assets: {int(size.min())}",
        f"- Maximum eligible assets: {int(size.max())}",
        f"- Final eligible assets: {int(size.iloc[-1])}",
        "",
        "## Additions And Removals",
        "",
        *(added_removed or ["- No universe membership changes."]),
        "",
        "## Delisted Assets",
        "",
        *([f"- {row.symbol}: {pd.Timestamp(row.delisting_date).date()}" for row in delisted.itertuples(index=False)] or ["- None recorded."]),
        "",
        "## Exclusion Reasons",
        "",
        *([f"- {reason}: {count}" for reason, count in exclusions.items()] or ["- None."]),
        "",
        "## Liquidity Statistics",
        "",
        f"- Dollar-volume median: {float(liquidity.get('50%', 0.0)):.2f}",
        f"- Dollar-volume minimum: {float(liquidity.get('min', 0.0)):.2f}",
        f"- Dollar-volume maximum: {float(liquidity.get('max', 0.0)):.2f}",
        "",
    ]
    return "\n".join(lines)


def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
