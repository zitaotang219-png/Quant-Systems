from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data_crypto.data_quality import validate_crypto_panel
from data_crypto.liquidity_sensitivity import build_liquidity_sensitivity
from data_crypto.market_cap import build_lagged_market_cap_weights
from universes.point_in_time import PointInTimeUniverseBuilder


def _asset_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "exchange": ["TEST", "TEST"],
            "listing_date": ["2024-01-02", "2024-01-01"],
            "delisting_date": [None, "2024-01-02"],
            "status": ["active", "delisted"],
        }
    ).assign(
        listing_date=lambda frame: pd.to_datetime(frame["listing_date"]),
        delisting_date=lambda frame: pd.to_datetime(frame["delisting_date"]),
    )


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in pd.date_range("2024-01-01", periods=4, freq="D"):
        for symbol in ("AAA", "BBB"):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 100.0,
                }
            )
    return pd.DataFrame(rows)


def _builder() -> PointInTimeUniverseBuilder:
    return PointInTimeUniverseBuilder(asset_master=_asset_master(), panel=_panel())


def test_asset_not_available_before_listing() -> None:
    universe = _builder().build_universe("2024-01-01").set_index("symbol")
    assert not bool(universe.at["AAA", "eligible"])
    assert universe.at["AAA", "exclusion_reason"] == "before_listing"


def test_delisted_asset_available_before_delisting() -> None:
    universe = _builder().build_universe("2024-01-02").set_index("symbol")
    assert bool(universe.at["BBB", "eligible"])


def test_asset_removed_after_delisting() -> None:
    universe = _builder().build_universe("2024-01-03").set_index("symbol")
    assert not bool(universe.at["BBB", "eligible"])
    assert universe.at["BBB", "exclusion_reason"] == "after_delisting"


def test_universe_is_point_in_time() -> None:
    history = _builder().build_over_time()
    aaa = history.loc[history["symbol"] == "AAA"].set_index("date")
    assert not bool(aaa.at[pd.Timestamp("2024-01-01"), "eligible"])
    assert bool(aaa.at[pd.Timestamp("2024-01-02"), "eligible"])


def test_incomplete_bar_removed() -> None:
    panel = _panel()
    panel.loc[0, "close"] = float("nan")
    clean = validate_crypto_panel(panel).clean_panel
    assert len(clean) == len(panel) - 1
    assert clean["close"].notna().all()


def test_missing_bar_classification() -> None:
    panel = _panel()
    panel = panel.loc[~((panel["symbol"] == "AAA") & (panel["date"] == pd.Timestamp("2024-01-03")))].copy()
    analysis = validate_crypto_panel(panel, asset_master=_asset_master()).report["missing_bar_analysis"]
    assert analysis["true_missing_bars"] == 1
    assert analysis["expected_absence"] == {"before_listing": 1, "after_delisting": 2}
    assert analysis["coverage_ratio"] == pytest.approx(0.8)


def test_asset_metadata_provenance_exists() -> None:
    metadata_path = Path("crypto_data/asset_master_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert {"source", "retrieval_date", "version", "symbol_count", "description"}.issubset(metadata)
    assert metadata["symbol_count"] == 30


def test_liquidity_filter_behavior() -> None:
    panel = _panel()
    panel.loc[panel["symbol"] == "AAA", "volume"] = 10.0
    panel.loc[panel["symbol"] == "BBB", "volume"] = 1_000_000.0
    sensitivity = build_liquidity_sensitivity(
        panel,
        _asset_master(),
        minimum_historical_bars=1,
        minimum_data_completeness=1.0,
        thresholds=(0.0, 1_000_000.0),
    )
    high_threshold = sensitivity.loc[sensitivity["threshold"] == 1_000_000.0].iloc[0]
    assert "AAA" in high_threshold["removed_assets"]
    assert "insufficient_liquidity" in high_threshold["removal_reasons"]


def test_universe_methodology_document_exists() -> None:
    document = Path("docs/universe_methodology.md")
    content = document.read_text(encoding="utf-8")
    assert "defined Crypto30 candidate universe" in content


def test_market_cap_weights_are_lagged() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "market_cap": [100.0, 200.0, 300.0, 100.0],
        }
    )
    weights = build_lagged_market_cap_weights(panel)
    second_day = weights.loc[weights["date"] == pd.Timestamp("2024-01-02")].set_index("symbol")
    assert second_day.at["AAA", "weight"] == pytest.approx(0.25)
    assert second_day.at["BBB", "weight"] == pytest.approx(0.75)
