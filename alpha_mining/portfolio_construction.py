from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import RegimeConfig, SelectedFactor


def factor_blend_weights(
    factors: list[SelectedFactor],
    weight_scheme: str,
    regime_label: str | None = None,
    regime_config: RegimeConfig | None = None,
) -> dict[str, float]:
    if not factors:
        return {}
    if weight_scheme == "fitness":
        raw = np.array([max(float(factor.fitness), 0.0) for factor in factors], dtype=float)
        if raw.sum() > 0.0:
            normalized = raw / raw.sum()
            base_weights = {
                factor.expression: float(weight)
                for factor, weight in zip(factors, normalized, strict=False)
            }
            return _apply_regime_multipliers(base_weights, factors, regime_label, regime_config)
    if weight_scheme == "manual":
        base_weights = _normalized_metric_weights(factors, "manual_weight")
        if base_weights:
            return _apply_regime_multipliers(base_weights, factors, regime_label, regime_config)
    if weight_scheme == "regression":
        base_weights = _normalized_metric_weights(factors, "regression_weight")
        if base_weights:
            return _apply_regime_multipliers(base_weights, factors, regime_label, regime_config)
    equal_weight = 1.0 / float(len(factors))
    base_weights = {factor.expression: equal_weight for factor in factors}
    return _apply_regime_multipliers(base_weights, factors, regime_label, regime_config)


def _normalized_metric_weights(factors: list[SelectedFactor], metric_name: str) -> dict[str, float]:
    raw = np.array(
        [max(float(factor.metrics.get(metric_name, 0.0)), 0.0) for factor in factors],
        dtype=float,
    )
    total = float(raw.sum())
    if total <= 1e-12:
        return {}
    normalized = raw / total
    return {
        factor.expression: float(weight)
        for factor, weight in zip(factors, normalized, strict=False)
    }


def combine_factor_columns(
    factor_columns: dict[str, pd.Series],
    selected_factors: list[SelectedFactor],
    weight_scheme: str,
    dates: pd.Series | None = None,
    regime_by_date: pd.Series | None = None,
    regime_config: RegimeConfig | None = None,
) -> pd.Series:
    if not factor_columns:
        return pd.Series(dtype=float)

    combined = pd.Series(0.0, index=next(iter(factor_columns.values())).index, dtype=float)
    if dates is None or regime_by_date is None or regime_config is None or not regime_config.enabled:
        blend_weights = factor_blend_weights(selected_factors, weight_scheme)
        for factor in selected_factors:
            values = factor_columns[factor.expression].astype(float).fillna(0.0)
            combined = combined.add(values * float(blend_weights.get(factor.expression, 0.0)), fill_value=0.0)
        return combined

    date_series = pd.to_datetime(dates, utc=False)
    regime_series = regime_by_date.copy()
    regime_series.index = pd.to_datetime(regime_series.index, utc=False)
    factor_frame = pd.DataFrame(
        {
            factor.expression: factor_columns[factor.expression].astype(float).fillna(0.0)
            for factor in selected_factors
        }
    )
    for date_value, day_index in date_series.groupby(date_series).groups.items():
        regime_label = str(regime_series.get(pd.Timestamp(date_value), "bull_low_vol"))
        blend_weights = factor_blend_weights(
            selected_factors,
            weight_scheme,
            regime_label=regime_label,
            regime_config=regime_config,
        )
        day_values = factor_frame.loc[day_index]
        day_combined = pd.Series(0.0, index=day_values.index, dtype=float)
        for factor in selected_factors:
            day_combined = day_combined.add(
                day_values[factor.expression] * float(blend_weights.get(factor.expression, 0.0)),
                fill_value=0.0,
            )
        combined.loc[day_index] = day_combined.to_numpy(dtype=float)
    return combined


def _apply_regime_multipliers(
    base_weights: dict[str, float],
    factors: list[SelectedFactor],
    regime_label: str | None,
    regime_config: RegimeConfig | None,
) -> dict[str, float]:
    if regime_config is None or not regime_config.enabled or regime_label is None:
        return base_weights

    adjusted: dict[str, float] = {}
    for factor in factors:
        base_weight = float(base_weights.get(factor.expression, 0.0))
        multiplier = _resolve_regime_multiplier(factor.expression, regime_label, regime_config)
        adjusted[factor.expression] = max(base_weight * multiplier, 0.0)
    total = float(sum(adjusted.values()))
    if total <= 1e-12:
        return base_weights
    return {expression: weight / total for expression, weight in adjusted.items()}


def _resolve_regime_multiplier(expression: str, regime_label: str, regime_config: RegimeConfig) -> float:
    explicit = regime_config.factor_regime_weight_overrides.get(expression, {})
    if regime_label in explicit:
        return float(explicit[regime_label])
    if not regime_config.use_style_multipliers:
        return 1.0
    style = _infer_factor_style(expression)
    style_override = regime_config.factor_regime_weight_overrides.get("__style__", {})
    if style in style_override and regime_label in {"bull_low_vol", "bull_high_vol"}:
        return float(style_override[style])
    return float(_DEFAULT_STYLE_REGIME_MULTIPLIERS.get(style, {}).get(regime_label, 1.0))


def _infer_factor_style(expression: str) -> str:
    lowered = expression.lower()
    trend_beta_tokens = (
        "beta",
        "momentum",
        "relative_strength",
        "trend",
        "breakout",
        "close_move",
        "basket_return",
    )
    liquidity_tokens = (
        "volume",
        "dollar_volume",
        "turnover",
        "liquidity",
        "taker_buy",
        "price_volume",
    )
    defensive_tokens = (
        "volatility",
        "rolling_std",
        "amplitude",
        "idiosyncratic",
        "residual",
        "shadow",
        "drawdown",
    )
    mean_reversion_tokens = (
        "zscore",
        "delay",
        "body_signed",
        "body_ratio",
        "return_zscore",
        "close_location",
    )
    if any(token in lowered for token in trend_beta_tokens):
        return "trend_beta"
    if any(token in lowered for token in liquidity_tokens):
        return "liquidity_rotation"
    if any(token in lowered for token in defensive_tokens):
        return "defensive"
    if any(token in lowered for token in mean_reversion_tokens):
        return "mean_reversion"
    return "neutral"


_DEFAULT_STYLE_REGIME_MULTIPLIERS = {
    "trend_beta": {
        "bull_low_vol": 1.30,
        "bull_high_vol": 1.10,
        "bear_low_vol": 0.85,
        "bear_high_vol": 0.60,
    },
    "liquidity_rotation": {
        "bull_low_vol": 1.15,
        "bull_high_vol": 1.10,
        "bear_low_vol": 0.95,
        "bear_high_vol": 0.85,
    },
    "defensive": {
        "bull_low_vol": 0.85,
        "bull_high_vol": 1.05,
        "bear_low_vol": 1.15,
        "bear_high_vol": 1.30,
    },
    "mean_reversion": {
        "bull_low_vol": 0.90,
        "bull_high_vol": 1.00,
        "bear_low_vol": 1.15,
        "bear_high_vol": 1.25,
    },
    "neutral": {
        "bull_low_vol": 1.00,
        "bull_high_vol": 1.00,
        "bear_low_vol": 1.00,
        "bear_high_vol": 1.00,
    },
}


def build_weight_frame(
    panel: pd.DataFrame,
    score_series: pd.Series,
    *,
    weighting_scheme: str,
    long_quantile: float,
    short_quantile: float,
    position_limit: float,
    gross_leverage: float,
    signal_vol_window: int,
    signal_clip: float,
    smoothing: float,
    market_neutral: bool,
    turnover_limit: float,
    regime_by_date: pd.Series | None = None,
    benchmark_follow_enabled: bool = False,
    benchmark_follow_btc_symbol: str = "BTCUSDT",
    benchmark_follow_btc_weight: float = 0.5,
    regime_benchmark_blend: dict[str, float] | None = None,
    regime_benchmark_direction: dict[str, float] | None = None,
    initial_previous_weights: pd.Series | None = None,
) -> pd.DataFrame:
    frame = panel[["date", "symbol"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["score"] = score_series.astype(float)
    frame["asset_return"] = _panel_returns(panel)
    frame["volatility"] = _rolling_volatility(
        frame["asset_return"],
        frame["date"],
        panel["symbol"],
        window=signal_vol_window,
    )
    frame["weight"] = 0.0

    previous_weights = initial_previous_weights.astype(float).copy() if initial_previous_weights is not None else None
    for _, day_index in frame.groupby("date", sort=False).groups.items():
        day_date = pd.Timestamp(frame.loc[day_index, "date"].iloc[0])
        regime_label = None
        if regime_by_date is not None and not regime_by_date.empty:
            regime_label = str(regime_by_date.get(day_date, "bull_low_vol"))
        day_scores = frame.loc[day_index, "score"]
        day_vol = frame.loc[day_index, "volatility"]
        target = construct_day_weights(
            score=day_scores,
            volatility=day_vol,
            weighting_scheme=weighting_scheme,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            position_limit=position_limit,
            gross_leverage=gross_leverage,
            signal_clip=signal_clip,
            smoothing=smoothing,
            market_neutral=market_neutral,
            previous_weights=previous_weights,
        )
        target = _apply_benchmark_follow_overlay(
            target=target,
            symbols=frame.loc[day_index, "symbol"].astype(str),
            regime_label=regime_label,
            position_limit=position_limit,
            gross_leverage=gross_leverage,
            market_neutral=market_neutral,
            benchmark_follow_enabled=benchmark_follow_enabled,
            benchmark_follow_btc_symbol=benchmark_follow_btc_symbol,
            benchmark_follow_btc_weight=benchmark_follow_btc_weight,
            regime_benchmark_blend=regime_benchmark_blend,
            regime_benchmark_direction=regime_benchmark_direction,
        )
        if previous_weights is not None and turnover_limit > 0.0:
            target = apply_turnover_limit(previous_weights, target, turnover_limit)
        previous_weights = target
        frame.loc[day_index, "weight"] = target.reindex(day_index).fillna(0.0).to_numpy(dtype=float)

    return frame[["date", "symbol", "weight"]]


def latest_weights_from_history(
    history: pd.DataFrame,
    score_series: pd.Series,
    *,
    weighting_scheme: str,
    long_quantile: float,
    short_quantile: float,
    position_limit: float,
    gross_leverage: float,
    signal_vol_window: int,
    signal_clip: float,
    smoothing: float,
    market_neutral: bool,
    regime_by_date: pd.Series | None = None,
    benchmark_follow_enabled: bool = False,
    benchmark_follow_btc_symbol: str = "BTCUSDT",
    benchmark_follow_btc_weight: float = 0.5,
    regime_benchmark_blend: dict[str, float] | None = None,
    regime_benchmark_direction: dict[str, float] | None = None,
    previous_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    ordered = history.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    latest_date = pd.Timestamp(ordered["date"].max())
    latest_index = ordered.index[ordered["date"] == latest_date]
    if len(latest_index) == 0:
        return {}

    volatility = _rolling_volatility(
        _panel_returns(ordered),
        ordered["date"],
        ordered["symbol"],
        window=signal_vol_window,
    )
    previous = None
    if previous_weights is not None:
        previous = pd.Series(
            {index: float(previous_weights.get(str(ordered.loc[index, "symbol"]), 0.0)) for index in latest_index},
            dtype=float,
        )

    target = construct_day_weights(
        score=score_series.loc[latest_index].astype(float),
        volatility=volatility.loc[latest_index].astype(float),
        weighting_scheme=weighting_scheme,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        position_limit=position_limit,
        gross_leverage=gross_leverage,
        signal_clip=signal_clip,
        smoothing=smoothing,
        market_neutral=market_neutral,
        previous_weights=previous,
    )
    regime_label = None
    if regime_by_date is not None and not regime_by_date.empty:
        regime_label = str(regime_by_date.get(latest_date, "bull_low_vol"))
    target = _apply_benchmark_follow_overlay(
        target=target,
        symbols=ordered.loc[latest_index, "symbol"].astype(str),
        regime_label=regime_label,
        position_limit=position_limit,
        gross_leverage=gross_leverage,
        market_neutral=market_neutral,
        benchmark_follow_enabled=benchmark_follow_enabled,
        benchmark_follow_btc_symbol=benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=benchmark_follow_btc_weight,
        regime_benchmark_blend=regime_benchmark_blend,
        regime_benchmark_direction=regime_benchmark_direction,
    )
    latest_weights = pd.DataFrame(
        {
            "symbol": ordered.loc[latest_index, "symbol"].astype(str).to_numpy(),
            "weight": target.reindex(latest_index).fillna(0.0).to_numpy(dtype=float),
        }
    )
    latest_weights = latest_weights.loc[latest_weights["weight"].abs() > 1e-12]
    return {str(row["symbol"]): float(row["weight"]) for _, row in latest_weights.iterrows()}


def latest_weights_from_snapshot(
    *,
    symbols: pd.Series,
    score: pd.Series,
    volatility: pd.Series,
    latest_date: pd.Timestamp,
    weighting_scheme: str,
    long_quantile: float,
    short_quantile: float,
    position_limit: float,
    gross_leverage: float,
    signal_clip: float,
    smoothing: float,
    market_neutral: bool,
    regime_by_date: pd.Series | None = None,
    benchmark_follow_enabled: bool = False,
    benchmark_follow_btc_symbol: str = "BTCUSDT",
    benchmark_follow_btc_weight: float = 0.5,
    regime_benchmark_blend: dict[str, float] | None = None,
    regime_benchmark_direction: dict[str, float] | None = None,
    previous_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if score.empty or volatility.empty or symbols.empty:
        return {}

    latest_index = score.index
    previous = None
    if previous_weights is not None:
        previous = pd.Series(
            {index: float(previous_weights.get(str(symbols.loc[index]), 0.0)) for index in latest_index},
            dtype=float,
        )

    target = construct_day_weights(
        score=score.astype(float),
        volatility=volatility.astype(float),
        weighting_scheme=weighting_scheme,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        position_limit=position_limit,
        gross_leverage=gross_leverage,
        signal_clip=signal_clip,
        smoothing=smoothing,
        market_neutral=market_neutral,
        previous_weights=previous,
    )
    regime_label = None
    if regime_by_date is not None and not regime_by_date.empty:
        regime_label = str(regime_by_date.get(pd.Timestamp(latest_date), "bull_low_vol"))
    target = _apply_benchmark_follow_overlay(
        target=target,
        symbols=symbols.astype(str),
        regime_label=regime_label,
        position_limit=position_limit,
        gross_leverage=gross_leverage,
        market_neutral=market_neutral,
        benchmark_follow_enabled=benchmark_follow_enabled,
        benchmark_follow_btc_symbol=benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=benchmark_follow_btc_weight,
        regime_benchmark_blend=regime_benchmark_blend,
        regime_benchmark_direction=regime_benchmark_direction,
    )
    latest_weights = pd.DataFrame(
        {
            "symbol": symbols.astype(str).to_numpy(),
            "weight": target.reindex(latest_index).fillna(0.0).to_numpy(dtype=float),
        }
    )
    latest_weights = latest_weights.loc[latest_weights["weight"].abs() > 1e-12]
    return {str(row["symbol"]): float(row["weight"]) for _, row in latest_weights.iterrows()}


def compute_signal_volatility(panel: pd.DataFrame, window: int) -> pd.Series:
    return _rolling_volatility(
        _panel_returns(panel),
        panel["date"],
        panel["symbol"],
        window=window,
    )


def construct_day_weights(
    *,
    score: pd.Series,
    volatility: pd.Series,
    weighting_scheme: str,
    long_quantile: float,
    short_quantile: float,
    position_limit: float,
    gross_leverage: float,
    signal_clip: float,
    smoothing: float,
    market_neutral: bool,
    previous_weights: pd.Series | None,
) -> pd.Series:
    aligned_score = score.astype(float).replace([np.inf, -np.inf], np.nan)
    aligned_vol = volatility.astype(float).replace([np.inf, -np.inf], np.nan)
    if weighting_scheme == "bucket":
        target = _bucket_weights(
            aligned_score,
            long_quantile,
            short_quantile,
            position_limit=position_limit,
            gross_leverage=gross_leverage,
        )
    else:
        ranked = aligned_score.rank(method="average", pct=True) - 0.5
        safe_vol = aligned_vol.where(aligned_vol.abs() > 1e-8, np.nan)
        signal = (ranked / safe_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if signal_clip > 0.0:
            signal = signal.clip(lower=-float(signal_clip), upper=float(signal_clip))
        if market_neutral and not signal.empty:
            signal = signal - float(signal.mean())
        target = _weights_from_signal(signal, position_limit=position_limit, gross_leverage=gross_leverage)

    if previous_weights is not None and smoothing > 0.0:
        previous = previous_weights.reindex(target.index).fillna(0.0).astype(float)
        target = (float(smoothing) * previous) + ((1.0 - float(smoothing)) * target)
        if market_neutral and not target.empty:
            target = target - float(target.mean())
        target = _weights_from_signal(target, position_limit=position_limit, gross_leverage=gross_leverage)

    return target.astype(float)


def apply_turnover_limit(
    previous_weights: pd.Series,
    target_weights: pd.Series,
    turnover_limit: float,
) -> pd.Series:
    if turnover_limit <= 0.0:
        return target_weights.astype(float)
    symbols = previous_weights.index.union(target_weights.index)
    previous = previous_weights.reindex(symbols).fillna(0.0).astype(float)
    target = target_weights.reindex(symbols).fillna(0.0).astype(float)
    turnover = float((target - previous).abs().sum())
    if turnover <= turnover_limit or turnover <= 1e-12:
        return target
    scale = turnover_limit / turnover
    return previous + (target - previous) * scale


def _panel_returns(panel: pd.DataFrame) -> pd.Series:
    if "daily_return" in panel.columns:
        return panel["daily_return"].astype(float)
    ordered = pd.DataFrame(
        {
            "date": pd.to_datetime(panel["date"], utc=False),
            "symbol": panel["symbol"].astype(str),
            "close": pd.to_numeric(panel["close"], errors="coerce"),
        },
        index=panel.index,
    ).sort_values(["symbol", "date"], kind="mergesort")
    returns = ordered.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)
    return returns.reindex(panel.index).astype(float)


def _rolling_volatility(returns: pd.Series, dates: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, utc=False),
            "symbol": symbols.astype(str),
            "returns": returns.astype(float),
        },
        index=returns.index,
    )
    ordered = frame.sort_values(["symbol", "date"], kind="mergesort")
    volatility = ordered.groupby("symbol", sort=False)["returns"].transform(
        lambda series: series.rolling(window=window, min_periods=max(5, min(window, 10))).std()
    )
    return volatility.reindex(returns.index).astype(float)


def _bucket_weights(
    score: pd.Series,
    long_quantile: float,
    short_quantile: float,
    *,
    position_limit: float,
    gross_leverage: float,
) -> pd.Series:
    clean = score.replace([np.inf, -np.inf], np.nan)
    ranked = clean.rank(method="average", pct=True)
    long_mask = ranked >= (1.0 - long_quantile)
    short_mask = ranked <= short_quantile
    weights = pd.Series(0.0, index=score.index, dtype=float)
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    if long_count > 0:
        weights.loc[long_mask] = 1.0 / long_count
    if short_count > 0:
        weights.loc[short_mask] = -1.0 / short_count
    return _weights_from_signal(
        weights,
        position_limit=position_limit,
        gross_leverage=gross_leverage,
    )


def _weights_from_signal(signal: pd.Series, position_limit: float, gross_leverage: float) -> pd.Series:
    clean = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if clean.abs().sum() <= 1e-12:
        return pd.Series(0.0, index=clean.index, dtype=float)
    if position_limit <= 0.0:
        position_limit = gross_leverage

    remaining = float(max(gross_leverage, 0.0))
    result = pd.Series(0.0, index=clean.index, dtype=float)
    active = clean.loc[clean.abs() > 1e-12].copy()
    while not active.empty and remaining > 1e-12:
        scaled = active / active.abs().sum() * remaining
        capped = scaled.abs() > position_limit
        if not capped.any():
            result.loc[scaled.index] = scaled
            remaining = 0.0
            break
        capped_values = scaled.loc[capped].apply(lambda value: np.sign(value) * position_limit)
        result.loc[capped_values.index] = capped_values
        remaining = float(max(gross_leverage - result.abs().sum(), 0.0))
        active = active.loc[~capped]

    return result.astype(float)


def _apply_benchmark_follow_overlay(
    target: pd.Series,
    symbols: pd.Series,
    regime_label: str | None,
    position_limit: float,
    gross_leverage: float,
    market_neutral: bool,
    benchmark_follow_enabled: bool,
    benchmark_follow_btc_symbol: str,
    benchmark_follow_btc_weight: float,
    regime_benchmark_blend: dict[str, float] | None,
    regime_benchmark_direction: dict[str, float] | None,
) -> pd.Series:
    if not benchmark_follow_enabled or regime_label is None:
        return target.astype(float)

    blend = float((regime_benchmark_blend or {}).get(regime_label, 0.0))
    if blend <= 1e-12:
        return target.astype(float)

    direction = float((regime_benchmark_direction or {}).get(regime_label, 1.0))
    overlay_weight = float(min(max(abs(benchmark_follow_btc_weight), 0.0), max(gross_leverage, 0.0)))
    if overlay_weight <= 1e-12:
        return target.astype(float)

    overlay = pd.Series(0.0, index=target.index, dtype=float)
    day_symbols = symbols.reindex(target.index).fillna("").astype(str).str.upper()
    btc_mask = day_symbols == str(benchmark_follow_btc_symbol).upper()
    if not bool(btc_mask.any()):
        return target.astype(float)
    overlay.loc[btc_mask] = overlay_weight * direction
    combined = (((1.0 - blend) * target.astype(float)) + (blend * overlay)).astype(float)
    if market_neutral and abs(direction) < 1e-12:
        combined = combined - float(combined.mean())
    # Reapply the configured single-name limit after blending in the benchmark overlay.
    return _weights_from_signal(combined, position_limit=position_limit, gross_leverage=gross_leverage)
