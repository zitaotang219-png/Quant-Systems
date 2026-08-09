# Alpha Mining Research Report (Original Fixed-Universe Baseline)

## Purpose And Scope

This report describes the original Alpha Mining research workflow before the Phase 2 point-in-time universe change. It is intended as self-contained context for an external GPT code/research review.

Baseline artifact: `reports/phase1_full_verify`  
Dataset: `crypto_data/binance_crypto30_daily/panel.csv`  
Seed: `42`  
Universe: defined Crypto30 candidate set, using the available rows in the local historical panel.

This is an offline research and backtest system. It is not live trading and does not place orders.

## Main Workflow

Entrypoint: `alpha_mining/run_crypto_workflow.py`

1. Load the local multi-asset daily OHLCV panel and the configured Crypto30 universe.
2. Engineer cross-sectional and time-series features in `features_crypto/engineer.py`.
3. Split research data into three rolling train/validation windows.
4. Generate symbolic factor expressions using genetic programming.
5. Fast-filter and deeply evaluate candidates using rank IC, portfolio performance, turnover, drawdown, stability, and regime-aware metrics.
6. Deduplicate and diversify the candidate pool, then select a final factor set.
7. Recheck each selected factor's long/short direction on validation data.
8. Freeze the selected factors and run one final out-of-sample backtest using the shared accounting engine.
9. Persist factor registry, configuration, metrics, ledgers, and reproducibility manifest.

The original baseline does not reselect factors during the final backtest. Its reported final period is therefore separate from the factor-mining period.

## Data And Time Splits

Local sample coverage: 2023-01-01 to 2026-05-12, 30 symbols, 34,645 rows in the Phase 1 baseline panel.

Research window: 2024-01-01 to 2024-12-31  
Final backtest window: 2025-06-01 to 2026-01-31

Rolling research windows:

| Window | Train | Validation | Fitness profile |
| --- | --- | --- | --- |
| 1 | 2024-01-01 to 2024-06-30 | 2024-07-01 to 2024-08-31 | Defensive |
| 2 | 2024-03-01 to 2024-08-31 | 2024-09-01 to 2024-10-31 | Aggressive |
| 3 | 2024-05-01 to 2024-10-31 | 2024-11-01 to 2024-12-31 | Balanced |

The rolling workflow uses one purge bar and three embargo bars. It produced 42, 40, and 43 new candidates in the three windows, then retained a final pool of 96 candidates.

## Factor Language And Features

Factor DSL: `alpha_mining/dsl.py`  
GP generator: `alpha_mining/gp_generator.py`

The symbolic language supports fields, constants, arithmetic, delay/delta, percentage changes, rolling mean/std, time-series rank, cross-sectional rank/z-score, and rolling correlation. Trees are constrained by configurable depth and allowed window choices.

Examples of engineered input fields include daily and intraday returns, OHLC shape measures, dollar volume, volume z-scores, realized volatility/skew/kurtosis, relative strength, residual momentum, rolling beta, and idiosyncratic return.

For the baseline quick configuration, GP used population size 40, three generations, maximum depth 5, seed 42, and periods/windows from small daily lookback sets. Raw `open`, `high`, `low`, and `close` fields were disallowed as direct terminal inputs, while engineered features remained available.

## Evaluation And Selection

Evaluator: `alpha_mining/evaluator.py`  
Selection/pipeline: `alpha_mining/pipeline.py`

Candidate evaluation includes finite-value coverage, cross-sectional rank IC, portfolio-level Sharpe, cumulative return, excess return versus equal weight, turnover, maximum drawdown, and path stability. Candidates can be rejected for insufficient valid values, weak IC, high turnover, or excessive drawdown.

The baseline selected factors are diversified by expression/value correlation before final selection. Final validation also checks long and short implementations and retains the preferable direction when supported by validation results.

## Baseline Selected Factors

Eight factors were selected:

1. `rolling_mean(zscore(residual_momentum_10d), 30)`
2. `rolling_mean(dollar_volume, 30)`
3. `rank(rank(rolling_std(delta(body_ratio, 5), 20)))`
4. `residual_momentum_10d`
5. `rolling_mean(rank(zscore(volatility_20)), 30)`
6. `rolling_mean(price_volume_confirmation, 30)`
7. `zscore(rolling_mean(rank(zscore(dollar_volume_change_5d)), 20))`
8. `zscore(rank(realized_skew_20))`

Exact per-factor fitness and IC statistics are in `reports/phase1_full_verify/selected_factors_summary.csv`.

## Portfolio And Backtest Interface

Portfolio construction: `alpha_mining/portfolio_construction.py`  
Backtest entrypoint: `alpha_mining/pipeline.py::backtest_selected_factors`  
Accounting engine: `backtest/engine.py`

Selected factor values are blended into cross-sectional target weights, subject to position limits, gross leverage, smoothing, turnover limits, market-neutral behavior, signal volatility controls, and a BTC benchmark-follow overlay. The final backtest uses the shared Phase 1 accounting path, including execution conventions, transaction costs, ledgers, reconciliation, trade log, and turnover report.

Baseline configuration highlights: 8 selected factors, 8 bps commission, 8 bps slippage, 30% long and 30% short quantiles, 0.08 position limit, 0.70 gross leverage, 1.2 turnover limit, and USD 100,000 initial capital.

## Baseline Final Backtest Results

| Metric | Result |
| --- | ---: |
| Final equity | 63,745.32 |
| Total return | -36.25% |
| Annual return | -37.19% |
| Sharpe | -6.326 |
| Maximum drawdown | 36.39% |
| Turnover | 292.940 |
| Total transaction cost | 36,773.48 |
| Backtest trading days | 244 |

Benchmark total returns: BTC -25.62%, equal-weight basket -41.50%, and market-cap benchmark -21.44%.

The baseline underperformed BTC and the market-cap benchmark, but outperformed the equal-weight basket by 5.24 percentage points. These values are descriptive only and should not be interpreted as evidence of investability.

## Reproducibility And Artifacts

The run stores `experiment_manifest.json`, `config.yaml`, `workflow_config.json`, factor registry files, selected-factor CSVs, final metrics, accounting reconciliation, event log, trade ledger, turnover report, and regime outputs under `reports/phase1_full_verify/`.

The manifest records code revision, dirty state, runtime package versions, seed, CLI command, data input hashes, configuration hash, and selected-factor count.

## Important Limitations For Review

- The original baseline uses a defined Crypto30 candidate set, not an exchange-wide historical universe.
- Before Phase 2, universe membership was not explicitly reconstructed point-in-time; later listings and the MATIC delisting motivated the lifecycle layer.
- The dataset lacks historical market-cap observations. The fallback benchmark is therefore a static reference-weight benchmark rather than a fully point-in-time market-cap index.
- The final return and Sharpe are materially negative. Any future work should diagnose robustness, factor economic rationale, turnover/cost sensitivity, and benchmark assumptions rather than optimize headline Sharpe.
- The research and final backtest are separated in time, but all conclusions remain conditional on the local data source, candidate-universe definition, and transaction-cost assumptions.

## Suggested External GPT Review Questions

1. Does the factor DSL or evaluator create any residual look-ahead risk under the stated signal/execution convention?
2. Are the rolling split, purge, embargo, and factor-direction checks sufficient for the candidate-generation process?
3. Are the fitness weights and rejection thresholds economically justified, or likely to overfit this dataset?
4. Does the portfolio construction path correctly translate factor values into tradable target weights under the stated cost and turnover assumptions?
5. Which robustness tests should be prioritized before treating the research outputs as decision-supporting evidence?
