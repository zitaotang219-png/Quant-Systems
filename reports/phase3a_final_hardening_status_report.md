# Phase 3A Final Hardening Status Report

## Purpose

Phase 3A is an observational hardening phase for the existing vanilla Genetic Programming (GP) alpha search. It does not change GP selection, crossover, mutation, DSL operators, fitness formulas, portfolio construction, or execution behavior.

The objective is to make the existing search process auditable before considering any future GP algorithm changes.

## Implemented And Verified

### GP Telemetry And Provenance

- `FactorEvaluator` rejection reasons are read from `metrics["reject_reason"]` and are persisted in candidate telemetry.
- Every raw GP individual has a deterministic ID derived from generation and population slot. Workflow exports namespace IDs by rolling window.
- Candidate provenance includes expression hash, creation operator, parent IDs, `parent_1_id`, `parent_2_id`, fitness, complexity, depth, rejection reason, and duplicate provenance.
- Parent IDs are captured when a child is created. Initial population, elitism, crossover, subtree mutation, point mutation, and reproduction can be reconstructed from the audit graph.
- Raw individual occurrences remain visible even when identical expressions are exactly deduplicated for evaluation.

### Search Funnel And Hypothesis Accounting

- `gp_research/stage_events.csv` is the persisted source for the search funnel.
- `gp_research/search_funnel.csv` is derived from events, grouped by rolling window and stage.
- Funnel stages include raw GP population, exact expression deduplication, fast filter, fast keep, deep evaluation, deep evaluation pass, fallback expansion, new candidate pool, rolling revalidation, rolling pool trimming, final candidate pool, initial factor selection, validation refinement, and final research selection.
- Funnel rows contain window, stage, input/pass/reject counts, unique-expression counts, cumulative unique-expression counts, evaluation counts where applicable, and rejection reasons.
- `gp_research/hypothesis_statistics.json` separates raw individual occurrences, expression evaluation calls, global unique expressions, unique `(window, expression)` pairs, per-window counts, and selected-factor count.

### Operator Effectiveness

Generation statistics now distinguish operator attempts from effective changes:

- Elite count.
- Crossover, subtree-mutation, and point-mutation attempted/effective counts.
- Reproduction count.
- Same-as-parent offspring, duplicate-in-generation offspring, novel-offspring count, and novel-offspring rate.

These are observational measures only. GP operators and their probabilities are unchanged.

### Protocol And Holdout Protection

- Every workflow run writes `gp_research/baseline_protocol.json`.
- The protocol captures data paths/fingerprints, PIT universe settings, rolling windows, purge/embargo, GP/evaluator/fitness configuration, transaction-cost assumptions, runtime metadata, git information, config hash, data hash, and seed.
- It explicitly declares the final holdout as spent:
  - Status: `spent`
  - Period: `2025-06-01/2026-01-31`
  - Policy: this period must not be used to choose, tune, accept, or reject future GP-search modifications.
- In `--research-only` mode, the final holdout panel is not materialized and the final backtest path is not invoked.

### Regression Coverage

The test suite includes:

- Rejection-reason propagation.
- Frozen vanilla-GP golden fixture with exact expressions, ordering, and fitness values.
- Deterministic IDs, real parent IDs, and window namespacing.
- Funnel reconciliation.
- Global versus window-level hypothesis counting.
- Operator effectiveness for unchanged and changed offspring.
- Research-only final-holdout non-materialization.
- Telemetry and research-selection invariance under fixed seed.
- Experiment and fitness-optimization ledger durability tests from prior Phase 3A steps.

## Verification Result

Latest code verification:

```text
python -m pytest -q
52 passed
```

`python -m compileall -q alpha_mining tests utils` and `git diff --check` also passed.

Known warnings are pre-existing pandas/scipy warnings in factor diagnostics and evaluator groupby behavior; they did not fail tests.

## Production Baseline Status

The required default-config, real-data `--research-only` production baseline has **not yet completed**.

A first default run exceeded the environment's 10-minute foreground command limit. A subsequent background run consumed approximately 37 minutes of CPU but was stopped because the final exact-deduplication funnel semantic correction occurred after that run started. Its outputs are therefore stale and must not be treated as Phase 3A evidence.

Do not use these incomplete/stale directories as evidence:

- `reports/phase3a_vanilla_production_baseline/`
- `reports/phase3a_vanilla_production_baseline_20260827T125157922302Z/`

The production baseline must be rerun from the final code with:

```powershell
python -m alpha_mining.run_crypto_workflow `
  --panel-csv crypto_data/binance_crypto30_daily/panel.csv `
  --output-dir reports/phase3a_vanilla_production_baseline `
  --research-only `
  --seed 42
```

Expected final artifacts:

- `gp_research/generation_statistics.csv`
- `gp_research/candidate_audit.csv`
- `gp_research/search_funnel.csv`
- `gp_research/hypothesis_statistics.json`
- `gp_research/baseline_protocol.json`
- `phase3a_report.md`

## Current Interpretation

No quantitative conclusion about duplicate rates, diversity collapse, validation survival, or operator waste should be made until the fresh production baseline completes. The code and tests now provide the instrumentation required to measure those issues without changing vanilla GP behavior.

## Closure Decision

Phase 3A code hardening and automated tests are complete. Phase 3A is **not formally closed** until the fresh default-config production research-only baseline completes and its generated artifacts are reviewed.
