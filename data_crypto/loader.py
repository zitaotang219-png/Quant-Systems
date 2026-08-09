from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data_crypto.data_quality import DataQualityResult, validate_crypto_panel
from data_crypto.liquidity_sensitivity import build_liquidity_sensitivity
from universes.point_in_time import PointInTimeUniverseBuilder, filter_panel_to_point_in_time_universe, load_asset_master


REQUIRED_OHLCV_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")


@dataclass
class PointInTimePanelResult:
    panel: pd.DataFrame
    universe: pd.DataFrame
    asset_master: pd.DataFrame
    data_quality: DataQualityResult
    liquidity_sensitivity: pd.DataFrame


def load_crypto_panel_csv(path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    return _normalize_crypto_panel(panel, source=Path(path))


def load_crypto_benchmark_csv(path: str | Path, benchmark_name: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise KeyError("Crypto benchmark CSV must contain a 'date' column.")
    if "close" not in frame.columns and "return" not in frame.columns:
        raise KeyError("Crypto benchmark CSV must contain either 'close' or 'return'.")

    benchmark = frame.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=False)
    if "close" in benchmark.columns:
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    if "open" in benchmark.columns:
        benchmark["open"] = pd.to_numeric(benchmark["open"], errors="coerce")
    if "return" in benchmark.columns:
        benchmark["return"] = pd.to_numeric(benchmark["return"], errors="coerce")
    if benchmark_name:
        benchmark["name"] = str(benchmark_name)
    elif "name" not in benchmark.columns:
        benchmark["name"] = "crypto_benchmark"
    benchmark = benchmark.sort_values("date", kind="mergesort").reset_index(drop=True)
    return benchmark


def load_crypto_universe_csv(path: str | Path) -> list[dict[str, str]]:
    universe = pd.read_csv(path)
    if "symbol" not in universe.columns:
        raise KeyError("Crypto universe CSV must contain a 'symbol' column.")

    records: list[dict[str, Any]] = []
    for _, row in universe.iterrows():
        symbol = str(row["symbol"]).strip()
        if not symbol:
            continue
        record = {
            str(column): row[column]
            for column in universe.columns
            if pd.notna(row[column])
        }
        record["symbol"] = symbol
        record["name"] = str(row.get("name", symbol)).strip() or symbol
        records.append(record)
    if not records:
        raise ValueError("Crypto universe CSV did not contain any usable symbols.")
    return records


def build_crypto_panel_from_directory(
    input_dir: str | Path,
    universe_csv: str | Path | None = None,
    file_pattern: str = "*.csv",
) -> pd.DataFrame:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Crypto panel directory does not exist: {input_path}")

    allowed_symbols: set[str] | None = None
    if universe_csv is not None:
        allowed_symbols = {record["symbol"] for record in load_crypto_universe_csv(universe_csv)}

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(input_path.glob(file_pattern)):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        if "symbol" not in frame.columns:
            frame = frame.copy()
            frame["symbol"] = csv_path.stem
        if allowed_symbols is not None:
            symbols = frame["symbol"].astype(str)
            frame = frame.loc[symbols.isin(allowed_symbols)].copy()
            if frame.empty:
                continue
        frames.append(frame)

    if not frames:
        raise ValueError(f"No crypto CSV files matched under {input_path}.")
    panel = pd.concat(frames, ignore_index=True)
    return _normalize_crypto_panel(panel, source=input_path)


def prepare_point_in_time_crypto_panel(
    panel: pd.DataFrame,
    *,
    asset_master_path: str | Path,
    minimum_historical_bars: int = 1,
    liquidity_threshold: float = 0.0,
    minimum_data_completeness: float = 1.0,
) -> PointInTimePanelResult:
    master = load_asset_master(asset_master_path)
    quality = validate_crypto_panel(panel, asset_master=master)
    builder = PointInTimeUniverseBuilder(
        asset_master=master,
        panel=quality.clean_panel,
        minimum_historical_bars=minimum_historical_bars,
        liquidity_threshold=liquidity_threshold,
        minimum_data_completeness=minimum_data_completeness,
    )
    universe = builder.build_over_time()
    filtered = filter_panel_to_point_in_time_universe(quality.clean_panel, universe)
    if filtered.empty:
        raise ValueError("Point-in-time universe filtering removed every panel row.")
    return PointInTimePanelResult(
        panel=filtered,
        universe=universe,
        asset_master=master,
        data_quality=quality,
        liquidity_sensitivity=build_liquidity_sensitivity(
            quality.clean_panel,
            master,
            minimum_historical_bars=minimum_historical_bars,
            minimum_data_completeness=minimum_data_completeness,
        ),
    )


def _normalize_crypto_panel(panel: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in panel.columns]
    if missing:
        raise KeyError(f"Crypto panel from {source} is missing required columns: {missing}")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    if "name" in frame.columns:
        frame["name"] = frame["name"].astype(str).str.strip()
    else:
        frame["name"] = frame["symbol"]

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    frame = frame.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])
    if frame.empty:
        raise ValueError(f"Crypto panel from {source} became empty after normalization.")
    return frame
