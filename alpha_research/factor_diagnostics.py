"""Deterministic post-selection diagnostics for symbolic alpha factors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.trading_convention import DEFAULT_TRADING_CONVENTION, TradingConvention


HORIZONS = (1, 3, 5, 10, 20)


def generate_factor_diagnostics(
    *,
    factors: list[Any],
    panel: pd.DataFrame,
    output_dir: str | Path,
    window_dir: str | Path | None = None,
    trading_convention: TradingConvention = DEFAULT_TRADING_CONVENTION,
) -> dict[str, pd.DataFrame]:
    """Write reliability diagnostics without changing factor selection or trading."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame = panel.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["forward_return"] = trading_convention.forward_return(frame)
    signals = _signal_frame(factors, frame)
    timeseries = _timeseries(frame, signals)
    statistics, daily_ic = _statistics(timeseries)
    decay = _decay(frame, signals)
    quantile_returns = _quantile_returns(frame, signals)
    similarity, clusters = _similarity(signals, daily_ic)
    families = _families(factors, statistics)
    stability = _stability(factors, window_dir)
    timeseries.to_csv(root / "factor_timeseries.csv", index=False)
    statistics.to_csv(root / "factor_statistics.csv", index=False)
    decay.to_csv(root / "factor_decay.csv", index=False)
    quantile_returns.to_csv(root / "factor_quantile_returns.csv", index=False)
    similarity.to_csv(root / "factor_similarity_matrix.csv", index=False)
    clusters.to_csv(root / "factor_clusters.csv", index=False)
    families.to_csv(root / "factor_family_summary.csv", index=False)
    stability.to_csv(root / "factor_window_stability.csv", index=False)
    (root / "alpha_validation_report.md").write_text(_report(statistics, decay, similarity, families, stability), encoding="utf-8")
    return {"timeseries": timeseries, "statistics": statistics, "decay": decay, "quantile_returns": quantile_returns, "similarity": similarity, "clusters": clusters, "families": families, "stability": stability}


def _signal_frame(factors: list[Any], panel: pd.DataFrame) -> pd.DataFrame:
    values = {str(factor.expression): factor.node.evaluate(panel).astype(float).replace([np.inf, -np.inf], np.nan) for factor in factors}
    return pd.DataFrame(values, index=panel.index)


def _timeseries(panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    base = panel[["date", "symbol", "forward_return"]].copy()
    return pd.concat([base, signals], axis=1).melt(id_vars=["date", "symbol", "forward_return"], var_name="factor", value_name="signal_value")


def _daily_rank_ic(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (factor, date), day in frame.groupby(["factor", "date"], sort=True):
        clean=day[["signal_value","forward_return"]].replace([np.inf,-np.inf],np.nan).dropna()
        rows.append({"factor":factor,"date":date,"rank_ic":clean["signal_value"].corr(clean["forward_return"],method="spearman") if len(clean)>=3 else np.nan,"observations":len(clean)})
    return pd.DataFrame(rows)


def _statistics(timeseries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily=_daily_rank_ic(timeseries); rows=[]
    for factor, day in daily.groupby("factor", sort=True):
        ic=day["rank_ic"].dropna(); std=float(ic.std(ddof=1)) if len(ic)>1 else 0.0; mean=float(ic.mean()) if len(ic) else 0.0
        standard_error=std/np.sqrt(len(ic)) if std>1e-12 and len(ic)>1 else 0.0; rows.append({"factor":factor,"mean_IC":mean,"IC_std":std,"ICIR":mean/std if std>1e-12 else 0.0,"IC_t_stat":mean/standard_error if standard_error else 0.0,"mean_IC_ci_lower":mean-1.96*standard_error,"mean_IC_ci_upper":mean+1.96*standard_error,"positive_IC_ratio":float((ic>0).mean()) if len(ic) else 0.0,"observations":int(day["observations"].sum())})
    return pd.DataFrame(rows), daily


def _decay(panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for horizon in HORIZONS:
        future_open=panel.groupby("symbol",sort=False)["open"].shift(-1)
        future_close=panel.groupby("symbol",sort=False)["close"].shift(-horizon)
        returns=(future_close/future_open)-1.0
        for factor in signals.columns:
            values=pd.DataFrame({"date":panel["date"],"signal_value":signals[factor],"forward_return":returns})
            daily=_daily_rank_ic(values.assign(factor=factor))
            rows.append({"factor":factor,"horizon_days":horizon,"mean_IC":float(daily["rank_ic"].mean()),"observations":int(daily["observations"].sum())})
    return pd.DataFrame(rows)


def _quantile_returns(panel: pd.DataFrame, signals: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Observational equal-weight factor portfolios, limited to selected factors."""
    rows = []
    for factor in signals.columns:
        frame = pd.DataFrame({"date": panel["date"], "signal": signals[factor], "forward_return": panel["forward_return"]})
        for date, day in frame.groupby("date", sort=True):
            clean = day.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < quantiles:
                continue
            ranks = clean["signal"].rank(method="first", pct=True)
            low = clean.loc[ranks <= 1.0 / quantiles, "forward_return"].mean()
            high = clean.loc[ranks > (quantiles - 1.0) / quantiles, "forward_return"].mean()
            rows.append({"factor": factor, "date": date, "low_quantile_return": float(low), "high_quantile_return": float(high), "high_minus_low_return": float(high - low), "observations": int(len(clean))})
    return pd.DataFrame(rows)

def _similarity(signals: pd.DataFrame, daily_ic: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    names=list(signals.columns); signal_corr=signals.corr(method="spearman"); ic_wide=daily_ic.pivot(index="date",columns="factor",values="rank_ic"); return_corr=ic_wide.corr()
    rows=[]; clusters=[]
    for i,left in enumerate(names):
        cluster=i
        for right in names[i+1:]:
            rows.append({"factor_left":left,"factor_right":right,"signal_correlation":float(signal_corr.loc[left,right]),"return_correlation":float(return_corr.loc[left,right])})
            if abs(float(signal_corr.loc[left,right]))>=0.8: cluster=min(cluster,names.index(right))
        clusters.append({"factor":left,"cluster_id":int(cluster)})
    return pd.DataFrame(rows),pd.DataFrame(clusters)


def _families(factors: list[Any], statistics: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for factor in factors:
        expression=str(factor.expression).lower(); family="risk"
        if "momentum" in expression or "relative_strength" in expression: family="momentum"
        elif "dollar_volume" in expression: family="liquidity"
        elif "volume" in expression: family="volume"
        elif "volatility" in expression or "skew" in expression or "kurt" in expression: family="volatility"
        elif "reversion" in expression or "delta" in expression: family="mean_reversion"
        rows.append({"factor":factor.expression,"family":family})
    detail=pd.DataFrame(rows).merge(statistics[["factor","mean_IC"]],on="factor",how="left")
    return detail.groupby("family",as_index=False).agg(factor_count=("factor","count"),mean_IC=("mean_IC","mean"))


def _stability(factors: list[Any], window_dir: str | Path | None) -> pd.DataFrame:
    selected=[str(f.expression) for f in factors]; windows=[]
    if window_dir:
        for path in sorted(Path(window_dir).glob("window_*/validated_pool.csv")):
            windows.append(set(pd.read_csv(path)["expression"].astype(str)))
    total=len(windows)
    return pd.DataFrame([{"factor":f,"selection_frequency":sum(f in window for window in windows),"stability_score":sum(f in window for window in windows)/total if total else 0.0,"window_count":total} for f in selected])


def _report(stats: pd.DataFrame, decay: pd.DataFrame, similarity: pd.DataFrame, families: pd.DataFrame, stability: pd.DataFrame) -> str:
    warnings=[]
    if not stats.empty and (stats["IC_t_stat"].abs()<2).any(): warnings.append("Some factors have IC t-statistics below 2; treat apparent signal strength cautiously.")
    if not similarity.empty and (similarity["signal_correlation"].abs()>=0.8).any(): warnings.append("Highly correlated factor signals indicate redundancy risk.")
    if not stability.empty and (stability["stability_score"]<0.5).any(): warnings.append("Some selected factors appear in fewer than half of validation pools.")
    return "# Alpha Validation Report\n\n## Search Bias Warnings\n\n"+"\n".join(f"- {item}" for item in (warnings or ["- No threshold-based warnings triggered."]))+"\n\n## Interpretation\n\nDiagnostics quantify reliability after GP selection; they do not improve, reweight, or approve factors.\n"
