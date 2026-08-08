from __future__ import annotations

import argparse
import json
import threading
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from alpha_mining.config import SelectedFactor
from alpha_mining.dsl import parse_expression
from alpha_mining.evaluator import _prepare_panel
from alpha_mining.pipeline import _build_evaluator
from alpha_mining.portfolio_construction import factor_blend_weights
from alpha_mining.run_crypto_workflow import (
    build_crypto_workflow_config,
    build_equal_weight_benchmark,
    build_market_cap_benchmark,
    load_crypto_universe_from_args,
    summarize_market_baseline,
    to_jsonable,
    write_json,
)
from data_crypto.loader import load_crypto_panel_csv, load_crypto_universe_csv, load_crypto_benchmark_csv
from execution_crypto.paper import run_crypto_paper_trading_with_factors
from features_crypto.engineer import build_crypto_panel_features


DEFAULT_PANEL_CSV = Path("crypto_data") / "binance_crypto30_daily" / "panel.csv"
DEFAULT_UNIVERSE_CSV = Path("crypto_data") / "binance_crypto30_daily" / "universe.csv"
DEFAULT_BTC_BENCHMARK_CSV = Path("crypto_data") / "binance_crypto30_daily" / "benchmark.csv"
DEFAULT_MARKET_CAP_BENCHMARK_CSV = Path("crypto_data") / "binance_crypto30_daily" / "market_cap_benchmark.csv"
DEFAULT_OUTPUT_DIR = Path("reports") / "crypto_dashboard_runs"
DEFAULT_UI_DIR = Path("ui_crypto")
DEFAULT_AUTO_WEIGHT_LOOKBACK_BARS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local crypto paper trading dashboard.")
    parser.add_argument("--panel-csv", default=str(DEFAULT_PANEL_CSV), help="Crypto panel CSV.")
    parser.add_argument("--universe-csv", default=str(DEFAULT_UNIVERSE_CSV), help="Crypto universe CSV.")
    parser.add_argument("--btc-benchmark-csv", default=str(DEFAULT_BTC_BENCHMARK_CSV), help="BTC benchmark CSV.")
    parser.add_argument("--market-cap-benchmark-csv", default=str(DEFAULT_MARKET_CAP_BENCHMARK_CSV), help="Market-cap benchmark CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for dashboard run artifacts.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the dashboard server.")
    parser.add_argument("--port", type=int, default=5055, help="Port for the dashboard server.")
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> Flask:
    ui_dir = Path(DEFAULT_UI_DIR).resolve()
    app = Flask(__name__, static_folder=str(ui_dir), static_url_path="")

    panel_raw = load_crypto_panel_csv(args.panel_csv)
    panel = build_crypto_panel_features(panel_raw)
    universe = load_crypto_universe_csv(args.universe_csv)
    btc_benchmark = load_crypto_benchmark_csv(args.btc_benchmark_csv, benchmark_name="BTC")
    market_cap_benchmark = load_crypto_benchmark_csv(
        args.market_cap_benchmark_csv,
        benchmark_name="crypto_market_cap_weighted",
    )
    equal_weight_benchmark = build_equal_weight_benchmark(panel, benchmark_name="crypto_equal_weight")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    job_store: dict[str, dict[str, Any]] = {}
    job_lock = threading.Lock()

    @app.get("/")
    def index() -> Any:
        return send_from_directory(str(ui_dir), "index.html")

    @app.get("/api/config")
    def api_config() -> Any:
        dates = pd.to_datetime(panel["date"], utc=False)
        unique_dates = sorted(dates.dropna().unique())
        default_paper_bars = 45
        default_start = pd.Timestamp(unique_dates[max(len(unique_dates) - default_paper_bars, 0)]).date() if unique_dates else None
        default_end = pd.Timestamp(unique_dates[-1]).date() if unique_dates else None
        payload = {
            "symbol_count": int(panel["symbol"].astype(str).nunique()),
            "symbols": sorted(panel["symbol"].astype(str).unique().tolist()),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "default_initial_capital": 100000.0,
            "default_paper_bars": default_paper_bars,
            "default_paper_start_date": str(default_start) if default_start is not None else None,
            "default_paper_end_date": str(default_end) if default_end is not None else None,
            "default_factors": _default_factor_examples(),
            "defaults": {
                "position_limit": 0.15,
                "gross_leverage": 1.20,
                "smoothing": 0.35,
                "market_neutral": False,
                "market_adaptation": "standard",
                "bull_follow_strength": 0.65,
                "factor_weight_scheme": "equal",
                "bull_trend_preference": "standard",
                "auto_weight_lookback_bars": DEFAULT_AUTO_WEIGHT_LOOKBACK_BARS,
            },
        }
        return jsonify(payload)

    @app.post("/api/run-paper")
    def api_run_paper() -> Any:
        payload = request.get_json(silent=True) or {}
        factor_lines = payload.get("factor_expressions", [])
        if isinstance(factor_lines, str):
            factor_lines = _split_factor_input(factor_lines)
        selected_factors = _build_selected_factors(factor_lines)
        if not selected_factors:
            return jsonify({"error": "Please provide at least one factor expression."}), 400

        initial_capital = float(payload.get("initial_capital", 100000.0))
        paper_bars = int(payload.get("paper_trading_bars", 45))
        paper_start_date = payload.get("paper_start_date") or None
        paper_end_date = payload.get("paper_end_date") or None
        portfolio_overrides = {
            "position_limit": float(payload.get("position_limit", 0.15)),
            "gross_leverage": float(payload.get("gross_leverage", 1.20)),
            "smoothing": float(payload.get("smoothing", 0.35)),
            "market_neutral": bool(payload.get("market_neutral", False)),
            "market_adaptation": str(payload.get("market_adaptation", "standard")).strip().lower() or "standard",
            "bull_follow_strength": float(payload.get("bull_follow_strength", 0.65)),
            "factor_weight_scheme": str(payload.get("factor_weight_scheme", "equal")).strip().lower() or "equal",
            "bull_trend_preference": str(payload.get("bull_trend_preference", "standard")).strip().lower() or "standard",
            "auto_weight_lookback_bars": int(payload.get("auto_weight_lookback_bars", DEFAULT_AUTO_WEIGHT_LOOKBACK_BARS)),
        }
        run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        with job_lock:
            job_store[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "run_dir": str(run_dir),
                "selected_factors": [factor.expression for factor in selected_factors],
                "result": None,
                "error": None,
            }

        worker = threading.Thread(
            target=_run_dashboard_job,
            args=(
                job_store,
                job_lock,
                job_id,
                panel,
                btc_benchmark,
                equal_weight_benchmark,
                market_cap_benchmark,
                selected_factors,
                initial_capital,
                paper_bars,
                paper_start_date,
                paper_end_date,
                portfolio_overrides,
                run_dir,
            ),
            daemon=True,
        )
        worker.start()
        return jsonify({"job_id": job_id, "status": "queued", "progress": 0.0, "message": "Queued"}), 202

    @app.get("/api/jobs/<job_id>")
    def api_job_status(job_id: str) -> Any:
        with job_lock:
            job = job_store.get(job_id)
            if job is None:
                return jsonify({"error": "Job not found."}), 404
            return jsonify(to_jsonable(job))

    return app


def _build_selected_factors(factor_lines: list[str]) -> list[SelectedFactor]:
    selected: list[SelectedFactor] = []
    for raw_line in factor_lines:
        line = str(raw_line).strip()
        if not line:
            continue
        direction, explicit_direction, expression, manual_weight = _parse_factor_entry(line)
        node = parse_expression(expression)
        selected.append(
            SelectedFactor(
                expression=expression,
                node=node,
                direction=direction,
                fitness=0.0,
                metrics={
                    "explicit_direction": explicit_direction,
                    "manual_weight": manual_weight,
                    "weight_source": "manual" if manual_weight != 1.0 else "default",
                },
                complexity=node.complexity(),
                finite_ratio=1.0,
                values=None,
            )
        )
    return selected


def _run_dashboard_job(
    job_store: dict[str, dict[str, Any]],
    job_lock: threading.Lock,
    job_id: str,
    panel: pd.DataFrame,
    btc_benchmark: pd.DataFrame,
    equal_weight_benchmark: pd.DataFrame,
    market_cap_benchmark: pd.DataFrame,
    selected_factors: list[SelectedFactor],
    initial_capital: float,
    paper_bars: int,
    paper_start_date: str | None,
    paper_end_date: str | None,
    portfolio_overrides: dict[str, Any],
    run_dir: Path,
) -> None:
    def set_job_state(**updates: Any) -> None:
        with job_lock:
            if job_id in job_store:
                job_store[job_id].update(updates)

    def progress_callback(stage: str, current: int, total: int, message: str) -> None:
        safe_total = max(int(total), 1)
        stage_base = {
            "prepare": 0.08,
            "calibration": 0.18,
            "paper_trading": 0.12,
            "finalize": 0.96,
            "completed": 1.0,
        }
        if stage == "paper_trading":
            progress = min(0.12 + (max(int(current), 0) / safe_total) * 0.82, 0.95)
        else:
            progress = stage_base.get(stage, 0.0)
        set_job_state(status="running", progress=float(progress), message=message)

    try:
        set_job_state(status="running", progress=0.03, message="Building workflow config")
        config = build_crypto_workflow_config(
            output_dir=run_dir,
            symbols=sorted(panel["symbol"].astype(str).unique()),
            panel=panel,
            fitness_override=None,
            quick=True,
        )
        portfolio_cls = config.portfolio.__class__
        config = config.__class__(
            gp=config.gp,
            evaluation=config.evaluation,
            fitness=config.fitness,
            regime=config.regime,
            portfolio=portfolio_cls(
                selected_factor_count=max(len(selected_factors), 4),
                min_selected_factor_count=min(max(len(selected_factors), 1), 4),
                max_pairwise_correlation=config.portfolio.max_pairwise_correlation,
                factor_weight_scheme=_resolve_factor_weight_scheme(portfolio_overrides["factor_weight_scheme"]),
                weighting_scheme=config.portfolio.weighting_scheme,
                position_limit=portfolio_overrides["position_limit"],
                turnover_limit=config.portfolio.turnover_limit,
                gross_leverage=portfolio_overrides["gross_leverage"],
                signal_vol_window=config.portfolio.signal_vol_window,
                signal_clip=config.portfolio.signal_clip,
                smoothing=portfolio_overrides["smoothing"],
                market_neutral=portfolio_overrides["market_neutral"],
                benchmark_follow_enabled=portfolio_overrides["market_adaptation"] != "off",
                benchmark_follow_btc_symbol=config.portfolio.benchmark_follow_btc_symbol,
                benchmark_follow_btc_weight=float(
                    min(max(portfolio_overrides["bull_follow_strength"], 0.0), portfolio_overrides["gross_leverage"])
                ),
                regime_benchmark_blend=_resolve_regime_blend(
                    portfolio_overrides["market_adaptation"],
                    portfolio_overrides["bull_follow_strength"],
                ),
                regime_benchmark_direction=dict(config.portfolio.regime_benchmark_direction),
            ),
            registry=config.registry,
            live_mode=config.live_mode,
            fast_filter_keep=config.fast_filter_keep,
            deep_eval_keep=config.deep_eval_keep,
            walk_forward_enabled=config.walk_forward_enabled,
            walk_forward_train_fraction=config.walk_forward_train_fraction,
            walk_forward_validation_fraction=config.walk_forward_validation_fraction,
            walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
            walk_forward_min_folds=config.walk_forward_min_folds,
            deduplicate_expressions=config.deduplicate_expressions,
            save_registry=False,
            universe_symbols=config.universe_symbols,
        )
        config = _apply_trend_preference_to_config(config, portfolio_overrides["bull_trend_preference"])
        calibration_panel = _build_calibration_panel(
            panel=panel,
            paper_bars=paper_bars,
            paper_start_date=paper_start_date,
            paper_end_date=paper_end_date,
            lookback_bars=portfolio_overrides["auto_weight_lookback_bars"],
        )
        if config.portfolio.factor_weight_scheme in {"fitness", "regression"}:
            set_job_state(status="running", progress=0.12, message="Calibrating factor directions and weights")
            selected_factors = _calibrate_selected_factors(
                selected_factors=selected_factors,
                calibration_panel=calibration_panel,
                config=config,
                factor_weight_scheme=config.portfolio.factor_weight_scheme,
            )
        set_job_state(status="running", progress=0.20, message="Starting paper trading")
        paper_result = run_crypto_paper_trading_with_factors(
            panel=panel,
            config=config,
            selected_factors=selected_factors,
            initial_cash=initial_capital,
            output_dir=run_dir / "paper",
            paper_trading_bars=paper_bars,
            paper_start_date=paper_start_date,
            paper_end_date=paper_end_date,
            progress_callback=progress_callback,
        )
        set_job_state(status="running", progress=0.97, message="Summarizing results")
        paper_summary = _summarize_paper_result(paper_result, initial_capital)
        benchmark_payload = {
            "btc": summarize_market_baseline(btc_benchmark, paper_result.daily_metrics["date"], initial_capital),
            "equal_weight": summarize_market_baseline(equal_weight_benchmark, paper_result.daily_metrics["date"], initial_capital),
            "market_cap": summarize_market_baseline(market_cap_benchmark, paper_result.daily_metrics["date"], initial_capital),
        }
        response = {
            "run_dir": str(run_dir),
            "selected_factors": _serialize_selected_factors(selected_factors, config.portfolio.factor_weight_scheme),
            "portfolio_config": {
                "position_limit": config.portfolio.position_limit,
                "gross_leverage": config.portfolio.gross_leverage,
                "smoothing": config.portfolio.smoothing,
                "market_neutral": config.portfolio.market_neutral,
                "market_adaptation": portfolio_overrides["market_adaptation"],
                "bull_follow_strength": portfolio_overrides["bull_follow_strength"],
                "factor_weight_scheme": config.portfolio.factor_weight_scheme,
                "bull_trend_preference": portfolio_overrides["bull_trend_preference"],
                "auto_weight_lookback_bars": portfolio_overrides["auto_weight_lookback_bars"],
            },
            "paper_summary": paper_summary,
            "benchmarks": benchmark_payload,
            "daily_metrics": _frame_records(paper_result.daily_metrics),
            "trades": _frame_records(paper_result.trades),
            "positions": _frame_records(paper_result.positions),
            "equity_curve": _frame_records(paper_result.equity_curve),
            "benchmark_curves": {
                "btc": _build_baseline_curve(btc_benchmark, paper_result.daily_metrics["date"], initial_capital),
                "equal_weight": _build_baseline_curve(equal_weight_benchmark, paper_result.daily_metrics["date"], initial_capital),
                "market_cap": _build_baseline_curve(market_cap_benchmark, paper_result.daily_metrics["date"], initial_capital),
            },
        }
        write_json(run_dir / "dashboard_response.json", to_jsonable(response))
        set_job_state(status="completed", progress=1.0, message="Completed", result=response, error=None)
    except Exception as exc:
        set_job_state(
            status="failed",
            progress=1.0,
            message=f"Failed: {exc}",
            error={"message": str(exc), "traceback": traceback.format_exc()},
        )


def _split_factor_input(raw_text: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    depth = 0
    normalized = str(raw_text).replace("\r\n", "\n").replace("\r", "\n")
    for char in normalized:
        if char == "(":
            depth += 1
            current.append(char)
            continue
        if char == ")":
            depth = max(depth - 1, 0)
            current.append(char)
            continue
        if depth == 0 and char in {",", "\n", "，", ";"}:
            token = "".join(current).strip()
            if token:
                entries.append(token)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        entries.append(tail)
    return entries


def _parse_factor_entry(raw_line: str) -> tuple[int, bool, str, float]:
    direction = 1
    explicit_direction = False
    expression = str(raw_line).strip()
    manual_weight = 1.0
    parts = [part.strip() for part in expression.split("|") if str(part).strip()]
    if parts:
        expression = parts[0]
        for extra in parts[1:]:
            lowered = extra.lower()
            if lowered.startswith("weight="):
                try:
                    manual_weight = float(extra.split("=", 1)[1].strip())
                except Exception:
                    manual_weight = 1.0
    lowered = expression.lower()
    if lowered.startswith("short:"):
        direction = -1
        explicit_direction = True
        expression = expression.split(":", 1)[1].strip()
    elif lowered.startswith("long:"):
        direction = 1
        explicit_direction = True
        expression = expression.split(":", 1)[1].strip()
    manual_weight = float(abs(manual_weight)) if np.isfinite(manual_weight) else 1.0
    if manual_weight <= 1e-12:
        manual_weight = 1.0
    return direction, explicit_direction, expression, manual_weight


def _default_factor_examples() -> list[str]:
    selected_path = Path("reports") / "crypto_alpha_workflow_full" / "selected_factors_summary.csv"
    if selected_path.exists():
        try:
            frame = pd.read_csv(selected_path)
            expressions = frame.get("expression")
            if expressions is not None:
                return [str(item) for item in expressions.dropna().astype(str).head(8).tolist()]
        except Exception:
            pass
    return [
        "long: rank(correlation(close, volatility_10, 10)) | weight=1.0",
        "long: zscore(dollar_volume) | weight=0.8",
        "long: rolling_std(volume_change_1d, 30) | weight=0.7",
        "short: rank(rolling_mean(rank(correlation(body_signed, price_volume_divergence, 3)), 10)) | weight=0.6",
    ]


def _resolve_factor_weight_scheme(raw_value: str) -> str:
    normalized = str(raw_value).strip().lower()
    if normalized in {"equal", "fitness", "manual", "regression"}:
        return normalized
    return "equal"


def _resolve_regime_blend(market_adaptation: str, bull_follow_strength: float) -> dict[str, float]:
    strength = float(min(max(bull_follow_strength, 0.0), 0.95))
    if market_adaptation == "off":
        return {
            "bull_low_vol": 0.0,
            "bull_high_vol": 0.0,
            "bear_low_vol": 0.0,
            "bear_high_vol": 0.0,
        }
    if market_adaptation == "strong":
        return {
            "bull_low_vol": strength,
            "bull_high_vol": min(max(strength - 0.15, 0.0), 0.90),
            "bear_low_vol": min(strength * 0.20, 0.25),
            "bear_high_vol": 0.0,
        }
    return {
        "bull_low_vol": strength,
        "bull_high_vol": min(max(strength - 0.25, 0.0), 0.75),
        "bear_low_vol": min(strength * 0.15, 0.20),
        "bear_high_vol": 0.0,
    }


def _apply_trend_preference_to_config(config: Any, bull_trend_preference: str) -> Any:
    regime_cls = config.regime.__class__
    overrides = dict(config.regime.factor_regime_weight_overrides)
    if bull_trend_preference == "off":
        return config.__class__(
            gp=config.gp,
            evaluation=config.evaluation,
            fitness=config.fitness,
            regime=regime_cls(
                enabled=config.regime.enabled,
                alpha=config.regime.alpha,
                threshold=config.regime.threshold,
                trend_window=config.regime.trend_window,
                volatility_window=config.regime.volatility_window,
                zscore_window=config.regime.zscore_window,
                return_column=config.regime.return_column,
                use_style_multipliers=False,
                factor_regime_weight_overrides=overrides,
            ),
            portfolio=config.portfolio,
            registry=config.registry,
            live_mode=config.live_mode,
            fast_filter_keep=config.fast_filter_keep,
            deep_eval_keep=config.deep_eval_keep,
            walk_forward_enabled=config.walk_forward_enabled,
            walk_forward_train_fraction=config.walk_forward_train_fraction,
            walk_forward_validation_fraction=config.walk_forward_validation_fraction,
            walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
            walk_forward_min_folds=config.walk_forward_min_folds,
            deduplicate_expressions=config.deduplicate_expressions,
            save_registry=config.save_registry,
            universe_symbols=config.universe_symbols,
        )
    multipliers = {
        "standard": {"trend_beta": 1.25, "defensive": 0.90, "mean_reversion": 0.92, "liquidity_rotation": 1.05},
        "strong": {"trend_beta": 1.45, "defensive": 0.80, "mean_reversion": 0.82, "liquidity_rotation": 1.12},
    }
    chosen = multipliers.get(str(bull_trend_preference).lower(), multipliers["standard"])
    overrides["__style__"] = chosen
    return config.__class__(
        gp=config.gp,
        evaluation=config.evaluation,
        fitness=config.fitness,
        regime=regime_cls(
            enabled=config.regime.enabled,
            alpha=config.regime.alpha,
            threshold=config.regime.threshold,
            trend_window=config.regime.trend_window,
            volatility_window=config.regime.volatility_window,
            zscore_window=config.regime.zscore_window,
            return_column=config.regime.return_column,
            use_style_multipliers=True,
            factor_regime_weight_overrides=overrides,
        ),
        portfolio=config.portfolio,
        registry=config.registry,
        live_mode=config.live_mode,
        fast_filter_keep=config.fast_filter_keep,
        deep_eval_keep=config.deep_eval_keep,
        walk_forward_enabled=config.walk_forward_enabled,
        walk_forward_train_fraction=config.walk_forward_train_fraction,
        walk_forward_validation_fraction=config.walk_forward_validation_fraction,
        walk_forward_backtest_fraction=config.walk_forward_backtest_fraction,
        walk_forward_min_folds=config.walk_forward_min_folds,
        deduplicate_expressions=config.deduplicate_expressions,
        save_registry=config.save_registry,
        universe_symbols=config.universe_symbols,
    )


def _build_calibration_panel(
    *,
    panel: pd.DataFrame,
    paper_bars: int,
    paper_start_date: str | None,
    paper_end_date: str | None,
    lookback_bars: int,
) -> pd.DataFrame:
    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    unique_dates = sorted(pd.to_datetime(ordered["date"], utc=False).dropna().unique())
    if not unique_dates:
        return ordered.iloc[0:0].copy()
    if paper_start_date is not None:
        paper_start = pd.Timestamp(paper_start_date)
    else:
        paper_start = pd.Timestamp(unique_dates[max(len(unique_dates) - int(max(paper_bars, 2)), 0)])
    pre_paper_dates = [pd.Timestamp(date) for date in unique_dates if pd.Timestamp(date) < paper_start]
    if not pre_paper_dates:
        return ordered.iloc[0:0].copy()
    selected_dates = pre_paper_dates[-int(max(lookback_bars, 30)) :]
    return ordered.loc[ordered["date"].isin(selected_dates)].copy().reset_index(drop=True)


def _calibrate_selected_factors(
    *,
    selected_factors: list[SelectedFactor],
    calibration_panel: pd.DataFrame,
    config: Any,
    factor_weight_scheme: str,
) -> list[SelectedFactor]:
    if calibration_panel.empty or len(selected_factors) == 0:
        return selected_factors
    evaluator = _build_evaluator(config)
    calibrated: list[SelectedFactor] = []
    for factor in selected_factors:
        evaluation = evaluator.evaluate(factor.node, calibration_panel)
        explicit_direction = bool(factor.metrics.get("explicit_direction", False))
        direction = int(factor.direction)
        fitness = float(factor.fitness)
        metrics = dict(factor.metrics)
        metrics.update(dict(evaluation.metrics))
        if not explicit_direction:
            direction = int(evaluation.direction)
            metrics["direction_source"] = "auto"
            fitness = float(evaluation.fitness)
        else:
            metrics["direction_source"] = "manual"
            if int(factor.direction) == int(evaluation.direction):
                fitness = float(evaluation.fitness)
            else:
                fitness = 0.0
                metrics["fitness_penalty_reason"] = "manual_direction_conflicts_with_history"
        metrics["weight_source"] = factor_weight_scheme if factor_weight_scheme in {"fitness", "regression"} else metrics.get("weight_source", "default")
        calibrated.append(
            SelectedFactor(
                expression=factor.expression,
                node=factor.node,
                direction=direction,
                fitness=fitness,
                metrics=metrics,
                complexity=factor.complexity,
                finite_ratio=float(evaluation.finite_ratio),
                values=evaluation.values,
            )
        )
    if factor_weight_scheme == "regression":
        calibrated = _apply_regression_weights(calibrated, calibration_panel, config)
    return calibrated


def _apply_regression_weights(
    factors: list[SelectedFactor],
    calibration_panel: pd.DataFrame,
    config: Any,
) -> list[SelectedFactor]:
    if calibration_panel.empty or not factors:
        return factors
    prepared = _prepare_panel(
        calibration_panel,
    )
    feature_map: dict[str, pd.Series] = {}
    explicit_map: dict[str, bool] = {}
    direction_map: dict[str, int] = {}
    for factor in factors:
        raw_values = factor.node.evaluate(prepared).astype(float)
        oriented = raw_values * float(factor.direction)
        feature_map[factor.expression] = _cross_sectional_standardize(prepared["date"], oriented)
        explicit_map[factor.expression] = bool(factor.metrics.get("explicit_direction", False))
        direction_map[factor.expression] = int(factor.direction)

    feature_frame = pd.DataFrame(feature_map).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = pd.to_numeric(prepared["future_return"], errors="coerce").fillna(0.0)
    if feature_frame.empty or feature_frame.shape[1] == 0:
        return factors
    x = feature_frame.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    ridge_alpha = 1.0
    xtx = x.T @ x
    beta = np.linalg.solve(xtx + (ridge_alpha * np.eye(xtx.shape[0])), x.T @ y)

    updated: list[SelectedFactor] = []
    for index, factor in enumerate(factors):
        coeff = float(beta[index])
        metrics = dict(factor.metrics)
        direction = int(factor.direction)
        if explicit_map[factor.expression]:
            regression_weight = max(coeff, 0.0)
            metrics["direction_source"] = "manual"
        else:
            if coeff < 0.0:
                direction = -1 * int(direction_map[factor.expression] or 1)
                coeff = abs(coeff)
            regression_weight = max(coeff, 0.0)
            metrics["direction_source"] = "regression"
        metrics["regression_beta"] = float(beta[index])
        metrics["regression_weight"] = float(regression_weight)
        metrics["weight_source"] = "regression"
        updated.append(
            SelectedFactor(
                expression=factor.expression,
                node=factor.node,
                direction=direction,
                fitness=factor.fitness,
                metrics=metrics,
                complexity=factor.complexity,
                finite_ratio=factor.finite_ratio,
                values=factor.values,
            )
        )
    return updated


def _cross_sectional_standardize(dates: pd.Series, values: pd.Series) -> pd.Series:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, utc=False),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    )
    standardized = frame.groupby("date", sort=False)["value"].transform(
        lambda series: ((series - series.mean()) / series.std()) if float(series.std() or 0.0) > 1e-12 else 0.0
    )
    return standardized.fillna(0.0).clip(-5.0, 5.0).astype(float)


def _serialize_selected_factors(selected_factors: list[SelectedFactor], weight_scheme: str) -> list[dict[str, Any]]:
    blend_weights = factor_blend_weights(selected_factors, weight_scheme)
    rows: list[dict[str, Any]] = []
    for factor in selected_factors:
        rows.append(
            {
                "expression": factor.expression,
                "direction": int(factor.direction),
                "direction_label": "short" if int(factor.direction) < 0 else "long",
                "fitness": float(factor.fitness),
                "final_weight": float(blend_weights.get(factor.expression, 0.0)),
                "manual_weight": float(factor.metrics.get("manual_weight", 0.0)),
                "regression_weight": float(factor.metrics.get("regression_weight", 0.0)),
                "regression_beta": float(factor.metrics.get("regression_beta", 0.0)),
                "validation_rank_ic_mean": float(factor.metrics.get("validation_rank_ic_mean", 0.0)),
                "sharpe": float(factor.metrics.get("sharpe", 0.0)),
                "weight_source": str(factor.metrics.get("weight_source", weight_scheme)),
                "direction_source": str(factor.metrics.get("direction_source", "manual")),
            }
        )
    return rows


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=False).astype(str)
    return json.loads(normalized.to_json(orient="records"))


def _summarize_paper_result(paper_result: Any, initial_cash: float) -> dict[str, float]:
    if paper_result.daily_metrics.empty:
        return {
            "days": 0,
            "final_equity": float(initial_cash),
            "total_return": 0.0,
            "mean_turnover": 0.0,
            "num_trades": 0,
        }
    final_equity = float(paper_result.daily_metrics["equity_after"].iloc[-1])
    return {
        "days": int(len(paper_result.daily_metrics)),
        "final_equity": final_equity,
        "total_return": (final_equity / float(initial_cash)) - 1.0,
        "mean_turnover": float(paper_result.daily_metrics["turnover"].mean()),
        "num_trades": int(len(paper_result.trades)),
    }


def _build_baseline_curve(
    benchmark_df: pd.DataFrame,
    date_values: pd.Series | list[Any],
    initial_capital: float,
) -> list[dict[str, Any]]:
    benchmark = benchmark_df.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=False)
    aligned_dates = pd.to_datetime(pd.Series(date_values), utc=False).dropna().unique()
    benchmark = benchmark.loc[benchmark["date"].isin(aligned_dates)].copy()
    benchmark = benchmark.sort_values("date", kind="mergesort").reset_index(drop=True)
    if benchmark.empty:
        return []
    if "return" in benchmark.columns:
        daily_return = pd.to_numeric(benchmark["return"], errors="coerce").fillna(0.0)
    else:
        daily_return = pd.to_numeric(benchmark["close"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
    equity = float(initial_capital) * (1.0 + daily_return).cumprod()
    curve = pd.DataFrame({"date": benchmark["date"].astype(str), "equity": equity.astype(float)})
    return json.loads(curve.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    app = create_app(args)
    print(f"Crypto dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
