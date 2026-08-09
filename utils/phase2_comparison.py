"""Compare fixed-universe and point-in-time workflow artifacts for Phase 2 auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def generate_phase2_comparison(
    baseline_dir: str | Path,
    point_in_time_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a reproducible comparison without changing either workflow result."""
    baseline = Path(baseline_dir)
    point_in_time = Path(point_in_time_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline_manifest = _read_json(baseline / "experiment_manifest.json")
    point_manifest = _read_json(point_in_time / "experiment_manifest.json")
    universe = _compare_universes(baseline, point_in_time)
    factors = _compare_factors(baseline, point_in_time)
    portfolio = _compare_portfolios(baseline, point_in_time)
    performance = _compare_performance(baseline, point_in_time)
    metadata = {
        "baseline_dir": str(baseline),
        "point_in_time_dir": str(point_in_time),
        "baseline_seed": baseline_manifest.get("experiment", {}).get("random_seed"),
        "point_in_time_seed": point_manifest.get("experiment", {}).get("random_seed"),
        "same_panel_hash": _input_hash(baseline_manifest, "panel.csv") == _input_hash(point_manifest, "panel.csv"),
        "baseline_config_hash": baseline_manifest.get("config_hash"),
        "point_in_time_config_hash": point_manifest.get("config_hash"),
    }

    universe.to_csv(output / "universe_comparison.csv", index=False)
    _write_json(output / "factor_comparison.json", factors)
    _write_json(output / "portfolio_comparison.json", portfolio)
    _write_json(output / "performance_comparison.json", performance)
    _write_json(output / "comparison_metadata.json", metadata)
    (output / "phase2_comparison_report.md").write_text(
        _build_report(metadata, universe, factors, portfolio, performance), encoding="utf-8"
    )
    return {
        "metadata": metadata,
        "factors": factors,
        "portfolio": portfolio,
        "performance": performance,
    }


def _compare_universes(baseline: Path, point_in_time: Path) -> pd.DataFrame:
    fixed = pd.read_csv(baseline / "panel.csv", parse_dates=["date"])
    point_universe = pd.read_csv(point_in_time / "point_in_time_universe.csv", parse_dates=["date"])
    fixed_size = fixed.groupby("date", sort=True)["symbol"].nunique().rename("fixed_universe_size")
    pit_size = (
        point_universe.loc[point_universe["eligible"]]
        .groupby("date", sort=True)["symbol"]
        .nunique()
        .rename("point_in_time_universe_size")
    )
    comparison = pd.concat([fixed_size, pit_size], axis=1).fillna(0).reset_index()
    comparison["universe_size_difference"] = comparison["point_in_time_universe_size"] - comparison["fixed_universe_size"]
    return comparison


def _compare_factors(baseline: Path, point_in_time: Path) -> dict[str, Any]:
    fixed = pd.read_csv(baseline / "selected_factors_summary.csv")
    pit = pd.read_csv(point_in_time / "selected_factors_summary.csv")
    fixed_set = set(fixed["expression"].astype(str))
    pit_set = set(pit["expression"].astype(str))
    overlap = sorted(fixed_set & pit_set)
    union = fixed_set | pit_set
    return {
        "baseline_factor_count": int(len(fixed_set)),
        "point_in_time_factor_count": int(len(pit_set)),
        "overlap_count": int(len(overlap)),
        "overlap_ratio": float(len(overlap) / len(union)) if union else 1.0,
        "overlapping_expressions": overlap,
    }


def _compare_portfolios(baseline: Path, point_in_time: Path) -> dict[str, Any]:
    fixed = pd.read_csv(baseline / "backtest" / "cross_sectional_weights.csv", parse_dates=["date"])
    pit = pd.read_csv(point_in_time / "backtest" / "cross_sectional_weights.csv", parse_dates=["date"])
    fixed_summary = _weight_summary(fixed)
    pit_summary = _weight_summary(pit)
    aligned = fixed_summary.join(pit_summary, how="inner", lsuffix="_baseline", rsuffix="_point_in_time")
    overlap = []
    for row in aligned.itertuples():
        fixed_symbols = set(row.active_symbols_baseline)
        pit_symbols = set(row.active_symbols_point_in_time)
        overlap.append(len(fixed_symbols & pit_symbols) / len(fixed_symbols | pit_symbols) if fixed_symbols | pit_symbols else 1.0)
    fixed_metrics = _read_json(baseline / "metrics.json")
    pit_metrics = _read_json(point_in_time / "metrics.json")
    return {
        "shared_backtest_dates": int(len(aligned)),
        "mean_holdings_overlap": float(sum(overlap) / len(overlap)) if overlap else 0.0,
        "baseline_average_gross_exposure": float(aligned["gross_exposure_baseline"].mean()) if not aligned.empty else 0.0,
        "point_in_time_average_gross_exposure": float(aligned["gross_exposure_point_in_time"].mean()) if not aligned.empty else 0.0,
        "baseline_average_net_exposure": float(aligned["net_exposure_baseline"].mean()) if not aligned.empty else 0.0,
        "point_in_time_average_net_exposure": float(aligned["net_exposure_point_in_time"].mean()) if not aligned.empty else 0.0,
        "baseline_turnover": float(fixed_metrics.get("turnover", 0.0)),
        "point_in_time_turnover": float(pit_metrics.get("turnover", 0.0)),
        "turnover_difference": float(pit_metrics.get("turnover", 0.0)) - float(fixed_metrics.get("turnover", 0.0)),
    }


def _weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    frame = weights.copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    return frame.groupby("date", sort=True).agg(
        active_symbols=("symbol", lambda values: tuple(sorted(frame.loc[values.index][frame.loc[values.index, "weight"].ne(0.0), "symbol"].astype(str)))),
        gross_exposure=("weight", lambda values: float(values.abs().sum())),
        net_exposure=("weight", "sum"),
    )


def _compare_performance(baseline: Path, point_in_time: Path) -> dict[str, dict[str, float]]:
    fixed = _read_json(baseline / "metrics.json")
    pit = _read_json(point_in_time / "metrics.json")
    keys = ("total_return", "sharpe", "max_drawdown", "total_cost")
    return {
        key: {
            "baseline": float(fixed.get(key, 0.0)),
            "point_in_time": float(pit.get(key, 0.0)),
            "difference": float(pit.get(key, 0.0)) - float(fixed.get(key, 0.0)),
        }
        for key in keys
    }


def _input_hash(manifest: dict[str, Any], filename: str) -> str | None:
    for item in manifest.get("data", {}).get("inputs", []):
        if str(item.get("path", "")).replace("\\", "/").endswith(filename):
            return item.get("sha256")
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_report(
    metadata: dict[str, Any],
    universe: pd.DataFrame,
    factors: dict[str, Any],
    portfolio: dict[str, Any],
    performance: dict[str, dict[str, float]],
) -> str:
    lines = [
        "# Phase 2 Fixed-Universe vs Point-in-Time Comparison",
        "",
        "This comparison measures research impact from point-in-time eligibility. It does not treat higher or lower Sharpe as a success criterion.",
        "",
        "## Reproducibility Controls",
        "",
        f"- Baseline seed: {metadata['baseline_seed']}",
        f"- Point-in-time seed: {metadata['point_in_time_seed']}",
        f"- Same input panel hash: {metadata['same_panel_hash']}",
        "",
        "## Universe",
        "",
        f"- Fixed universe size: {int(universe['fixed_universe_size'].max())}",
        f"- Point-in-time universe size range: {int(universe['point_in_time_universe_size'].min())} to {int(universe['point_in_time_universe_size'].max())}",
        f"- Dates with a different universe size: {int(universe['universe_size_difference'].ne(0).sum())}",
        "",
        "## Factors",
        "",
        f"- Baseline selected factors: {factors['baseline_factor_count']}",
        f"- Point-in-time selected factors: {factors['point_in_time_factor_count']}",
        f"- Selected-factor overlap: {factors['overlap_count']} ({factors['overlap_ratio']:.2%})",
        "",
        "## Portfolio",
        "",
        f"- Mean holdings overlap: {portfolio['mean_holdings_overlap']:.2%}",
        f"- Turnover: baseline {portfolio['baseline_turnover']:.6f}; point-in-time {portfolio['point_in_time_turnover']:.6f}; difference {portfolio['turnover_difference']:.6f}",
        f"- Average gross exposure: baseline {portfolio['baseline_average_gross_exposure']:.6f}; point-in-time {portfolio['point_in_time_average_gross_exposure']:.6f}",
        "",
        "## Performance",
        "",
    ]
    for key, values in performance.items():
        lines.append(f"- {key}: baseline {values['baseline']:.6f}; point-in-time {values['point_in_time']:.6f}; difference {values['difference']:.6f}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Differences quantify the research effect of removing lifecycle-ineligible observations. They are not optimization targets and do not by themselves establish superior expected performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 1 fixed-universe and Phase 2 point-in-time workflow results.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--point-in-time-dir", required=True)
    parser.add_argument("--output-dir", default="reports/phase2_comparison")
    args = parser.parse_args()
    generate_phase2_comparison(args.baseline_dir, args.point_in_time_dir, args.output_dir)


if __name__ == "__main__":
    main()
