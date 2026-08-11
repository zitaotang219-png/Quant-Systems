"""Audit report generation for point-in-time universe eligibility."""

from __future__ import annotations

import pandas as pd


LIMITATION = (
    "The current universe is a defined Crypto30 candidate universe with point-in-time "
    "eligibility filtering. It is not a complete exchange-wide historical universe."
)


def build_universe_audit(asset_master: pd.DataFrame, universe: pd.DataFrame) -> str:
    """Build a deterministic lifecycle and membership audit from existing inputs."""
    master = asset_master.copy()
    master["listing_date"] = pd.to_datetime(master["listing_date"], utc=False)
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], utc=False)
    history = universe.copy()
    history["date"] = pd.to_datetime(history["date"], utc=False)
    eligible = history.loc[history["eligible"]]
    first = eligible.groupby("symbol", sort=False)["date"].min()
    last = eligible.groupby("symbol", sort=False)["date"].max()
    lifecycle = master[["symbol", "listing_date", "delisting_date"]].copy()
    lifecycle["first_eligible_date"] = lifecycle["symbol"].map(first)
    lifecycle["last_eligible_date"] = lifecycle["symbol"].map(last)

    additions: list[str] = []
    previous: set[str] = set()
    for date, day in eligible.groupby("date", sort=True):
        current = set(day["symbol"].astype(str))
        added, removed = sorted(current - previous), sorted(previous - current)
        if added or removed:
            additions.append(f"- {date.date()}: additions={added or ['none']}; removals={removed or ['none']}")
        previous = current
    size = eligible.groupby("date", sort=True)["symbol"].nunique()
    exclusions = history.loc[~history["eligible"], "exclusion_reason"].replace(
        {"missing_current_bar": "missing_data", "incomplete_history": "missing_data"}
    ).value_counts().to_dict()

    lines = ["# Universe Audit", "", "## Asset Lifecycle Summary", "", "| Symbol | Listing date | Delisting date | First eligible date | Last eligible date |", "| --- | --- | --- | --- | --- |"]
    for row in lifecycle.sort_values("symbol", kind="mergesort").itertuples(index=False):
        lines.append(
            f"| {row.symbol} | {_date(row.listing_date)} | {_date(row.delisting_date)} | "
            f"{_date(row.first_eligible_date)} | {_date(row.last_eligible_date)} |"
        )
    lines.extend(["", "## Universe Statistics", "", f"- Candidate assets: {len(master)}", f"- Eligible dates: {len(size)}", f"- Minimum eligible assets: {int(size.min()) if not size.empty else 0}", f"- Maximum eligible assets: {int(size.max()) if not size.empty else 0}", "", "## Additions And Removals", "", *(additions or ["- None."]), "", "## Exclusion Reasons", ""])
    for reason in ("before_listing", "after_delisting", "insufficient_history", "insufficient_liquidity", "missing_data"):
        lines.append(f"- {reason}: {int(exclusions.get(reason, 0))}")
    lines.extend(["", "## Limitations", "", LIMITATION, ""])
    return "\n".join(lines)


def _date(value: object) -> str:
    return "" if pd.isna(value) else str(pd.Timestamp(value).date())
