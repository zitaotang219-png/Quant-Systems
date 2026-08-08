"""Paper trading backed by the same event-driven accounting engine as research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_mining.config import AlphaMiningConfig, SelectedFactor
from alpha_mining.portfolio_construction import build_weight_frame, combine_factor_columns
from alpha_mining.regime import build_regime_frame
from alpha_mining.registry import FactorRegistry
from backtest.event_driven import EventDrivenBacktestResult, EventDrivenPortfolioBacktester
from backtest.ledger import PortfolioLedger
from backtest.portfolio_constraints import PortfolioConstraints
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION
from execution.cost_model import TransactionCostModel


@dataclass
class PaperBroker:
    """Read-only view of the ledger produced by the shared execution simulator."""

    ledger: PortfolioLedger

    @property
    def cash(self) -> float:
        return self.ledger.cash.balance

    @property
    def positions(self):
        return self.ledger.positions

    @property
    def trade_log(self) -> list[dict[str, float | str]]:
        return self.ledger.trades.trades


@dataclass
class PaperTradingResult:
    equity_curve: pd.DataFrame
    daily_metrics: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    broker: PaperBroker
    accounting: EventDrivenBacktestResult
    target_weights: pd.DataFrame


def split_crypto_research_and_paper_panel(
    panel: pd.DataFrame,
    paper_trading_bars: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = _prepare_ordered_panel(panel)
    unique_dates = sorted(ordered["date"].dropna().unique())
    if len(unique_dates) <= paper_trading_bars:
        raise ValueError("Not enough bars to split research and paper periods.")
    paper_dates = unique_dates[-paper_trading_bars:]
    paper_start = pd.Timestamp(paper_dates[0])
    return (
        ordered.loc[ordered["date"] < paper_start].copy().reset_index(drop=True),
        ordered.loc[ordered["date"].isin(paper_dates)].copy().reset_index(drop=True),
    )


def run_crypto_paper_trading(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    initial_cash: float = 1.0,
    output_dir: str | Path = "outputs",
    paper_trading_bars: int = 90,
) -> PaperTradingResult:
    frozen_factors, metadata = load_frozen_factors(config)
    if not frozen_factors:
        raise ValueError("No frozen factors found. Crypto paper trading must load frozen factors only.")
    return run_crypto_paper_trading_with_factors(
        panel=panel,
        config=config,
        selected_factors=frozen_factors,
        initial_cash=initial_cash,
        output_dir=output_dir,
        paper_trading_bars=paper_trading_bars,
        training_metadata=metadata,
    )


def run_crypto_paper_trading_with_factors(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    selected_factors: list[SelectedFactor],
    initial_cash: float = 1.0,
    output_dir: str | Path = "outputs",
    paper_trading_bars: int = 90,
    training_metadata: dict[str, Any] | None = None,
    paper_start_date: str | pd.Timestamp | None = None,
    paper_end_date: str | pd.Timestamp | None = None,
    progress_callback: Any | None = None,
) -> PaperTradingResult:
    if not selected_factors:
        raise ValueError("Crypto paper trading requires at least one selected factor.")

    ordered = _prepare_ordered_panel(panel, config.universe_symbols)
    paper_dates = _resolve_paper_dates(
        ordered,
        paper_trading_bars=paper_trading_bars,
        paper_start_date=paper_start_date,
        paper_end_date=paper_end_date,
    )
    _validate_training_range(training_metadata or {}, pd.Timestamp(paper_dates[0]))
    if progress_callback is not None:
        progress_callback(
            stage="prepare",
            current=0,
            total=max(len(paper_dates) - 1, 1),
            message=f"Preparing paper window {paper_dates[0]} -> {paper_dates[-1]}",
        )

    target_weights = _build_target_weights(ordered, selected_factors, config)
    paper_panel = ordered.loc[ordered["date"].isin(paper_dates)].copy()
    paper_weights = target_weights.loc[target_weights["date"].isin(paper_dates)].copy()
    simulator = EventDrivenPortfolioBacktester(
        initial_capital=initial_cash,
        cost_model=TransactionCostModel(
            commission_bps=config.evaluation.transaction_cost_bps,
            slippage_bps=config.evaluation.slippage_bps,
        ),
        constraints=PortfolioConstraints(
            max_position_size=config.portfolio.position_limit,
            max_leverage=config.portfolio.gross_leverage,
            max_gross_exposure=config.portfolio.gross_leverage,
            max_net_exposure=config.portfolio.gross_leverage,
        ),
        convention=DEFAULT_TRADING_CONVENTION,
    )
    accounting = simulator.run(paper_panel, paper_weights)
    daily_metrics = _build_daily_metrics(accounting)
    positions = _build_position_rows(accounting.trades)
    equity_curve = accounting.timeseries[["date", "equity_curve"]].rename(columns={"equity_curve": "equity"}).copy()
    _save_logs(output_dir, accounting, equity_curve, daily_metrics, positions)
    if progress_callback is not None:
        progress_callback(
            stage="completed",
            current=max(len(paper_dates) - 1, 1),
            total=max(len(paper_dates) - 1, 1),
            message="Paper trading complete",
        )
    return PaperTradingResult(
        equity_curve=equity_curve,
        daily_metrics=daily_metrics,
        trades=accounting.trades,
        positions=positions,
        broker=PaperBroker(ledger=accounting.ledger),
        accounting=accounting,
        target_weights=paper_weights,
    )


def load_frozen_factors(config: AlphaMiningConfig) -> tuple[list[SelectedFactor], dict[str, Any]]:
    registry = FactorRegistry(config.registry_dir())
    return registry.load(config), registry.load_metadata(config)


def _prepare_ordered_panel(panel: pd.DataFrame, universe_symbols: tuple[str, ...] = ()) -> pd.DataFrame:
    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    if universe_symbols:
        ordered = ordered.loc[ordered["symbol"].astype(str).isin(tuple(str(symbol) for symbol in universe_symbols))]
    return ordered.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _resolve_paper_dates(
    ordered: pd.DataFrame,
    *,
    paper_trading_bars: int,
    paper_start_date: str | pd.Timestamp | None,
    paper_end_date: str | pd.Timestamp | None,
) -> list[pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(ordered["date"], utc=False).dropna().unique())
    if paper_start_date is None and paper_end_date is None:
        if len(unique_dates) <= paper_trading_bars:
            raise ValueError("Not enough bars to split research and paper periods.")
        return [pd.Timestamp(date) for date in unique_dates[-paper_trading_bars:]]
    start = pd.Timestamp(paper_start_date) if paper_start_date is not None else pd.Timestamp(unique_dates[-paper_trading_bars])
    end = pd.Timestamp(paper_end_date) if paper_end_date is not None else pd.Timestamp(unique_dates[-1])
    resolved = [pd.Timestamp(date) for date in unique_dates if start <= pd.Timestamp(date) <= end]
    if len(resolved) < 2:
        raise ValueError("Paper trading window must contain at least 2 bars.")
    return resolved


def _validate_training_range(metadata: dict[str, Any], paper_start: pd.Timestamp) -> None:
    training_end = pd.to_datetime(metadata.get("data_range", {}).get("date_max"), utc=False, errors="coerce")
    if pd.notna(training_end) and training_end >= paper_start:
        raise ValueError(
            "Frozen factor training range overlaps the crypto paper window. "
            f"Training end date {training_end} must be earlier than paper start date {paper_start}."
        )


def _build_target_weights(
    ordered: pd.DataFrame,
    selected_factors: list[SelectedFactor],
    config: AlphaMiningConfig,
) -> pd.DataFrame:
    factor_columns = {
        factor.expression: factor.node.evaluate(ordered).astype(float) * float(factor.direction)
        for factor in selected_factors
    }
    regime_by_date = None
    if config.regime.enabled:
        regime_frame = build_regime_frame(ordered, config.regime)
        if not regime_frame.empty:
            regime_by_date = regime_frame.set_index("date")["regime"]
    combined = combine_factor_columns(
        factor_columns=factor_columns,
        selected_factors=selected_factors,
        weight_scheme=config.portfolio.factor_weight_scheme,
        dates=ordered["date"],
        regime_by_date=regime_by_date,
        regime_config=config.regime,
    )
    return build_weight_frame(
        panel=ordered,
        score_series=combined,
        weighting_scheme=config.portfolio.weighting_scheme,
        long_quantile=config.evaluation.long_quantile,
        short_quantile=config.evaluation.short_quantile,
        position_limit=config.portfolio.position_limit,
        gross_leverage=config.portfolio.gross_leverage,
        signal_vol_window=config.portfolio.signal_vol_window,
        signal_clip=config.portfolio.signal_clip,
        smoothing=config.portfolio.smoothing,
        market_neutral=config.portfolio.market_neutral,
        turnover_limit=config.portfolio.turnover_limit,
        regime_by_date=regime_by_date,
        benchmark_follow_enabled=config.portfolio.benchmark_follow_enabled,
        benchmark_follow_btc_symbol=config.portfolio.benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=config.portfolio.benchmark_follow_btc_weight,
        regime_benchmark_blend=dict(config.portfolio.regime_benchmark_blend),
        regime_benchmark_direction=dict(config.portfolio.regime_benchmark_direction),
    )


def _build_daily_metrics(accounting: EventDrivenBacktestResult) -> pd.DataFrame:
    if accounting.timeseries.empty:
        return pd.DataFrame()
    reconciliation = accounting.reconciliation.rename(columns={"starting_equity": "equity_before", "ending_equity": "equity_after"})
    return accounting.timeseries.merge(reconciliation, on="date", how="left")


def _build_position_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "units", "entry_price", "notional"])
    entries = trades.loc[trades["reason"] == "entry"].copy()
    entries["units"] = entries["quantity"].where(entries["side"] == "BUY", -entries["quantity"])
    return entries.rename(columns={"price": "entry_price"})[["timestamp", "symbol", "units", "entry_price", "notional"]]


def _save_logs(
    output_dir: str | Path,
    accounting: EventDrivenBacktestResult,
    equity: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    positions: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    accounting.trades.to_csv(root / "paper_trades.csv", index=False)
    equity.to_csv(root / "paper_equity.csv", index=False)
    daily_metrics.to_csv(root / "paper_daily_metrics.csv", index=False)
    positions.to_csv(root / "paper_positions.csv", index=False)
    accounting.reconciliation.to_csv(root / "paper_accounting_reconciliation.csv", index=False)
    accounting.turnover.to_csv(root / "paper_turnover_report.csv", index=False)
    accounting.constraints.to_csv(root / "paper_constraint_report.csv", index=False)
