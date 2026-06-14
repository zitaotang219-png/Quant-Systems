from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_WINDOWS = {
    "short": 5,
    "medium": 10,
    "long": 20,
}


def build_crypto_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"Crypto panel is missing required columns for feature engineering: {missing}")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    by_symbol = frame.groupby("symbol", sort=False)
    previous_close = by_symbol["close"].shift(1)
    trading_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)

    frame["daily_return"] = by_symbol["close"].pct_change(fill_method=None)
    frame["log_return"] = np.log(frame["close"] / previous_close)
    frame["intraday_return"] = (frame["close"] / frame["open"]) - 1.0
    frame["amplitude"] = (frame["high"] - frame["low"]) / previous_close
    frame["body_ratio"] = (frame["close"] - frame["open"]).abs() / trading_range
    frame["body_signed"] = (frame["close"] - frame["open"]) / trading_range
    frame["upper_shadow_ratio"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / trading_range
    frame["lower_shadow_ratio"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / trading_range
    frame["close_location_in_range"] = ((frame["close"] - frame["low"]) / trading_range) - 0.5

    frame["volume_change_1d"] = by_symbol["volume"].pct_change(fill_method=None)
    frame["volume_change_5d"] = by_symbol["volume"].pct_change(FEATURE_WINDOWS["short"], fill_method=None)
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["dollar_volume_change_5d"] = by_symbol["dollar_volume"].pct_change(FEATURE_WINDOWS["short"], fill_method=None)
    frame["log_volume"] = np.log(frame["volume"].replace(0.0, np.nan))
    frame["volume_zscore_20"] = by_symbol["log_volume"].transform(
        lambda series: _rolling_zscore(series, FEATURE_WINDOWS["long"])
    )

    frame["volatility_10"] = by_symbol["daily_return"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["medium"], min_periods=FEATURE_WINDOWS["medium"]).std()
    )
    frame["volatility_20"] = by_symbol["daily_return"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["long"], min_periods=FEATURE_WINDOWS["long"]).std()
    )
    frame["realized_skew_20"] = by_symbol["daily_return"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["long"], min_periods=FEATURE_WINDOWS["long"]).skew()
    )
    frame["realized_kurt_20"] = by_symbol["daily_return"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["long"], min_periods=FEATURE_WINDOWS["long"]).kurt()
    )

    frame["return_zscore_20"] = by_symbol["daily_return"].transform(
        lambda series: _rolling_zscore(series, FEATURE_WINDOWS["long"])
    )
    frame["volume_change_zscore_20"] = by_symbol["volume_change_1d"].transform(
        lambda series: _rolling_zscore(series, FEATURE_WINDOWS["long"])
    )
    frame["price_volume_divergence"] = frame["return_zscore_20"] - frame["volume_change_zscore_20"]
    frame["price_volume_confirmation"] = frame["return_zscore_20"] + frame["volume_change_zscore_20"]

    basket_return = frame.groupby("date", sort=False)["daily_return"].transform("mean")
    frame["basket_return_1d"] = basket_return
    frame["relative_strength_1d"] = frame["daily_return"] - frame["basket_return_1d"]
    frame["relative_strength_5d"] = by_symbol["relative_strength_1d"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["short"], min_periods=FEATURE_WINDOWS["short"]).sum()
    )
    frame["relative_strength_10d"] = by_symbol["relative_strength_1d"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["medium"], min_periods=FEATURE_WINDOWS["medium"]).sum()
    )

    frame["rolling_beta_20"] = _rolling_beta(frame["daily_return"], frame["basket_return_1d"], frame["symbol"], 20)
    frame["idiosyncratic_return_1d"] = frame["daily_return"] - (frame["rolling_beta_20"] * frame["basket_return_1d"])
    frame["residual_momentum_5d"] = by_symbol["idiosyncratic_return_1d"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["short"], min_periods=FEATURE_WINDOWS["short"]).sum()
    )
    frame["residual_momentum_10d"] = by_symbol["idiosyncratic_return_1d"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["medium"], min_periods=FEATURE_WINDOWS["medium"]).sum()
    )
    frame["residual_volatility_20"] = by_symbol["idiosyncratic_return_1d"].transform(
        lambda series: series.rolling(FEATURE_WINDOWS["long"], min_periods=FEATURE_WINDOWS["long"]).std()
    )
    frame["market_neutral_close_move"] = frame["daily_return"] - frame["basket_return_1d"]

    return _sanitize_feature_frame(frame)


def engineered_crypto_feature_columns(panel: pd.DataFrame) -> list[str]:
    numeric_columns = panel.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"open", "high", "low", "close", "volume"}
    return [column for column in numeric_columns if column not in excluded]


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=window).mean()
    rolling_std = series.rolling(window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def _rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "symbol": symbols.astype(str),
            "asset": asset_returns.astype(float),
            "market": market_returns.astype(float),
        }
    )
    grouped = frame.groupby("symbol", sort=False, group_keys=False)

    def _beta(group: pd.DataFrame) -> pd.Series:
        covariance = group["asset"].rolling(window, min_periods=window).cov(group["market"])
        variance = group["market"].rolling(window, min_periods=window).var().replace(0.0, np.nan)
        return covariance / variance

    return grouped.apply(_beta).reset_index(level=0, drop=True).reindex(frame.index).astype(float)


def _sanitize_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        return features
    features.loc[:, numeric_columns] = features.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    return features

