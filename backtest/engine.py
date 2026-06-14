from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest

from backtest.formatting import to_backtesting_format
from backtest.strategy import BaseStrategy, build_backtestingpy_adapter


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float,
        transaction_cost_bps: float,
        slippage_bps: float,
        risk_fraction: float,
    ) -> None:
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.risk_fraction = risk_fraction

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        output_dir: str | Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
        signal_frame = _generate_signals(df, strategy).reset_index(drop=True)
        bt_frame = to_backtesting_format(signal_frame)
        bt_frame = bt_frame.rename(columns={"raw_signal": "Signal"})

        adapter_cls = build_backtestingpy_adapter(self.risk_fraction)
        backtest = Backtest(
            bt_frame,
            adapter_cls,
            cash=self.initial_capital,
            commission=self.transaction_cost_bps / 10000.0,
            spread=self.slippage_bps / 10000.0,
            exclusive_orders=True,
            finalize_trades=True,
            trade_on_close=False,
        )
        stats = backtest.run()

        result = _attach_backtest_outputs(signal_frame, stats, self.initial_capital)
        metrics = _compute_metrics(result, stats)
        artifacts = _build_artifacts(backtest, stats, output_dir)
        return result, metrics, artifacts

    def run_cross_sectional(
        self,
        panel: pd.DataFrame,
        strategy: Any,
        output_dir: str | Path | None = None,
        strategy_backtest_kwargs: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
        research_metrics: dict[str, float] = {}
        if hasattr(strategy, "factor"):
            try:
                from alpha_mining import FactorEvaluator

                evaluation = FactorEvaluator(
                    transaction_cost_bps=self.transaction_cost_bps + self.slippage_bps,
                ).evaluate(strategy.factor, panel)
                research_metrics = {
                    "ic_mean": float(evaluation.metrics.get("ic_mean", 0.0)),
                    "rank_ic_mean": float(evaluation.metrics.get("rank_ic_mean", 0.0)),
                    "fitness": evaluation.fitness,
                    "validation_rank_ic_mean": float(evaluation.metrics.get("validation_rank_ic_mean", 0.0)),
                    "test_rank_ic_mean": float(evaluation.metrics.get("test_rank_ic_mean", 0.0)),
                }
            except Exception:
                research_metrics = {}

        portfolio = strategy.backtest(
            panel=panel,
            initial_capital=self.initial_capital,
            transaction_cost_bps=self.transaction_cost_bps + self.slippage_bps,
            **(strategy_backtest_kwargs or {}),
        )
        result = portfolio.timeseries.copy()
        metrics = _compute_cross_sectional_metrics(result)
        metrics.update(research_metrics)
        artifacts = _build_cross_sectional_artifacts(strategy, portfolio, output_dir)
        return result, metrics, artifacts


def run_backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    risk_fraction: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    engine = BacktestEngine(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_fraction=risk_fraction,
    )
    result, metrics, _ = engine.run(df, strategy)
    return result, metrics


def run_cross_sectional_backtest(
    panel: pd.DataFrame,
    strategy: Any,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    risk_fraction: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    engine = BacktestEngine(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_fraction=risk_fraction,
    )
    result, metrics, _ = engine.run_cross_sectional(panel, strategy)
    return result, metrics


def _attach_backtest_outputs(
    df: pd.DataFrame,
    stats: pd.Series,
    initial_capital: float,
) -> pd.DataFrame:
    equity_curve = stats["_equity_curve"].copy()
    equity_curve.index.name = "date"
    equity_curve = equity_curve.reset_index()
    if "DrawdownPct" in equity_curve.columns:
        equity_curve["drawdown"] = -equity_curve["DrawdownPct"].astype(float)
    else:
        equity_curve["drawdown"] = (
            equity_curve["Equity"].astype(float) / equity_curve["Equity"].astype(float).cummax()
        ) - 1.0

    result = df.copy().reset_index(drop=True)
    result["asset_return"] = result["close"].pct_change().fillna(0.0)
    result["benchmark_curve"] = initial_capital * (1.0 + result["asset_return"]).cumprod()
    result = result.merge(
        equity_curve[["date", "Equity", "drawdown"]],
        on="date",
        how="left",
    )
    result = result.rename(columns={"Equity": "equity_curve"})
    result["equity_curve"] = result["equity_curve"].ffill().bfill()
    result["drawdown"] = result["drawdown"].ffill().fillna(0.0)
    result["strategy_return_net"] = result["equity_curve"].pct_change().fillna(0.0)
    return result


def _generate_signals(df: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
    signals = df.copy()
    signals["raw_signal"] = strategy.generate_signal_series(signals)
    signals["position"] = signals["raw_signal"].ffill().fillna(0.0)
    signals["target_position"] = signals["position"].shift(1).fillna(0.0)
    return signals


def _compute_metrics(
    df: pd.DataFrame,
    stats: pd.Series,
) -> dict[str, float]:
    turnover_metrics = _compute_turnover_metrics(df, stats)
    total_return = _safe_ratio(stats.get("Return [%]"))
    benchmark_return = _safe_ratio(stats.get("Buy & Hold Return [%]"))
    max_drawdown = -_safe_ratio(stats.get("Max. Drawdown [%]"))
    sharpe = _safe_float(stats.get("Sharpe Ratio"))
    win_rate = _safe_ratio(stats.get("Win Rate [%]"))
    final_equity = _safe_float(stats.get("Equity Final [$]"), fallback=df["equity_curve"].iloc[-1])

    mean_daily = df["strategy_return_net"].mean()
    std_daily = df["strategy_return_net"].std()
    fallback_sharpe = 0.0 if std_daily == 0 or pd.isna(std_daily) else (mean_daily / std_daily) * math.sqrt(252)

    return {
        "total_return": total_return,
        "annual_return": _safe_ratio(stats.get("Return (Ann.) [%]")),
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "annual_volatility": _safe_ratio(stats.get("Volatility (Ann.) [%]")),
        "alpha": _safe_ratio(stats.get("Alpha [%]")),
        "beta": _safe_float(stats.get("Beta")),
        "max_drawdown": max_drawdown,
        "avg_drawdown": -_safe_ratio(stats.get("Avg. Drawdown [%]")),
        "sharpe": sharpe if sharpe != 0.0 else fallback_sharpe,
        "sortino": _safe_float(stats.get("Sortino Ratio")),
        "calmar": _safe_float(stats.get("Calmar Ratio")),
        "cagr": _safe_ratio(stats.get("CAGR [%]")),
        "win_rate": win_rate,
        "best_trade": _safe_ratio(stats.get("Best Trade [%]")),
        "worst_trade": _safe_ratio(stats.get("Worst Trade [%]")),
        "avg_trade": _safe_ratio(stats.get("Avg. Trade [%]")),
        "profit_factor": _safe_float(stats.get("Profit Factor")),
        "expectancy": _safe_ratio(stats.get("Expectancy [%]")),
        "final_equity": final_equity if final_equity != 0.0 else float(df["equity_curve"].iloc[-1]),
        "exposure_time": _safe_ratio(stats.get("Exposure Time [%]")),
        "num_trades": _safe_float(stats.get("# Trades")),
        "avg_trade_duration_days": _duration_to_days(stats.get("Avg. Trade Duration")),
        "max_trade_duration_days": _duration_to_days(stats.get("Max. Trade Duration")),
        "sqn": _safe_float(stats.get("SQN")),
        "turnover_ratio": turnover_metrics["turnover_ratio"],
        "annualized_turnover_ratio": turnover_metrics["annualized_turnover_ratio"],
        "sharpe_turnover_ratio": _safe_divide(
            sharpe if sharpe != 0.0 else fallback_sharpe,
            turnover_metrics["turnover_ratio"],
        ),
        "alpha_per_turnover_unit": _safe_divide(
            _safe_ratio(stats.get("Alpha [%]")),
            turnover_metrics["turnover_ratio"],
        ),
        "gross_turnover_dollars": turnover_metrics["gross_turnover_dollars"],
    }


def _build_artifacts(
    backtest: Backtest,
    stats: pd.Series,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "stats": _serialize_stats(stats),
        "trades": _serialize_trades(stats),
    }

    if output_dir is None:
        return artifacts

    output_path = Path(output_dir)
    plot_path = output_path / "backtest_report.html"
    try:
        backtest.plot(
            results=stats,
            filename=str(plot_path),
            open_browser=False,
            plot_return=True,
            plot_drawdown=True,
            plot_volume=True,
            plot_pl=True,
            plot_trades=True,
            smooth_equity=False,
            relative_equity=True,
        )
        artifacts["plot_path"] = str(plot_path)
    except Exception as exc:
        artifacts["plot_error"] = str(exc)

    return artifacts


def _build_cross_sectional_artifacts(
    strategy: Any,
    portfolio: Any,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    artifacts = {
        "weights": portfolio.weights.to_dict(orient="records"),
    }

    if hasattr(strategy, "factor"):
        artifacts["factor_expression"] = strategy.factor.describe()

    if output_dir is None:
        return artifacts

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_path / "cross_sectional_equity.csv"
    weights_path = output_path / "cross_sectional_weights.csv"
    portfolio.timeseries.to_csv(timeseries_path, index=False)
    portfolio.weights.to_csv(weights_path, index=False)
    artifacts["timeseries_path"] = str(timeseries_path)
    artifacts["weights_path"] = str(weights_path)
    return artifacts


def _serialize_stats(stats: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in stats.items():
        if str(key).startswith("_"):
            continue
        payload[str(key)] = _serialize_value(value)
    return payload


def _serialize_trades(stats: pd.Series) -> list[dict[str, Any]]:
    trades = stats.get("_trades")
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return []
    serializable = trades.copy()
    for column in serializable.columns:
        serializable[column] = serializable[column].map(_serialize_value)
    return serializable.to_dict(orient="records")


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "__class__") and value.__class__.__name__.endswith("Strategy"):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _compute_cross_sectional_metrics(result: pd.DataFrame) -> dict[str, float]:
    if result.empty:
        return {
            "pnl": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "turnover": 0.0,
            "max_drawdown": 0.0,
        }

    daily = result.copy()
    mean_daily = float(daily["net_return"].mean())
    std_daily = float(daily["net_return"].std())
    sharpe = 0.0 if std_daily == 0.0 or np.isnan(std_daily) else mean_daily / std_daily * math.sqrt(252)
    first_equity = float(daily["equity_curve"].iloc[0])
    first_return = float(daily["net_return"].iloc[0]) if "net_return" in daily.columns else 0.0
    initial_equity = first_equity / (1.0 + first_return) if np.isfinite(first_return) and first_return > -1.0 else first_equity
    total_return = float(daily["equity_curve"].iloc[-1] / initial_equity - 1.0) if initial_equity != 0.0 else 0.0
    annual_return = float((1.0 + total_return) ** (252.0 / max(len(daily), 1)) - 1.0)

    return {
        "pnl": float(daily["pnl"].sum()),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "turnover": float(daily["turnover"].mean() * 252.0),
        "max_drawdown": float(-daily["drawdown"].min()),
    }


def _duration_to_days(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds() / 86400.0)
    return float(pd.to_timedelta(value).total_seconds() / 86400.0)


def _compute_turnover_metrics(df: pd.DataFrame, stats: pd.Series) -> dict[str, float]:
    trades = stats.get("_trades")
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return {
            "turnover_ratio": 0.0,
            "annualized_turnover_ratio": 0.0,
            "gross_turnover_dollars": 0.0,
        }

    trade_values = (trades["Size"].abs() * trades["EntryPrice"].abs()) + (
        trades["Size"].abs() * trades["ExitPrice"].abs()
    )
    gross_turnover = float(trade_values.sum())
    avg_equity = float(df["equity_curve"].mean()) if not df.empty else 0.0
    turnover_ratio = 0.0 if avg_equity == 0.0 else gross_turnover / avg_equity

    num_days = max(len(df), 1)
    annualized_turnover_ratio = turnover_ratio * (252.0 / num_days)

    return {
        "turnover_ratio": float(turnover_ratio),
        "annualized_turnover_ratio": float(annualized_turnover_ratio),
        "gross_turnover_dollars": gross_turnover,
    }


def _safe_ratio(value: object) -> float:
    numeric = _safe_float(value)
    return numeric / 100.0


def _safe_float(value: object, fallback: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return float(fallback)
    return float(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator)
