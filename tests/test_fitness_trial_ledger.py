from __future__ import annotations

import json

import numpy as np

from alpha_mining.config import FitnessConfig
from alpha_mining.optimize_crypto_fitness import build_optimization_summary, sample_fitness_config
from utils.fitness_trial_ledger import FitnessTrialLedger


def _record_trial(ledger: FitnessTrialLedger, started: dict, index: int, *, failed: bool = False) -> dict:
    return ledger.record_trial(
        optimization_run_id=started["optimization_run_id"],
        parent_experiment_id=started["parent_experiment_id"],
        trial_index=index,
        fitness_configuration={"sharpe_weight": float(index)},
        candidate_pool_identity={"sha256": "pool-hash", "factor_count": 4},
        seed=17,
        status="failed" if failed else "completed",
        objective_value=-1_000_000_000.0 if failed else float(index),
        validation_metrics={} if failed else {"sharpe": float(index)},
        selected_factor_count=0 if failed else 4,
        failure_reason="expected failure" if failed else None,
    )


def _started(ledger: FitnessTrialLedger) -> dict:
    return ledger.start_run(
        parent_experiment_id="experiment-parent",
        configuration={"seed": 17, "trials": 3},
        artifact_locations={"output_directory": "reports/test"},
    )


def test_each_attempt_is_durable_and_non_winners_are_retained(tmp_path) -> None:
    ledger = FitnessTrialLedger(tmp_path)
    started = _started(ledger)
    first = _record_trial(ledger, started, 1)
    persisted_after_first = [json.loads(line) for line in ledger.trials_path.read_text(encoding="utf-8").splitlines()]
    assert [record["trial_id"] for record in persisted_after_first] == [first["trial_id"]]
    second = _record_trial(ledger, started, 2)
    third = _record_trial(ledger, started, 3)
    trials = ledger.trials_for_run(started["optimization_run_id"])
    assert len(trials) == 3
    assert {trial["trial_id"] for trial in trials} == {first["trial_id"], second["trial_id"], third["trial_id"]}
    assert second["trial_id"] != third["trial_id"]


def test_failed_trials_and_crash_prefix_remain_durable(tmp_path) -> None:
    ledger = FitnessTrialLedger(tmp_path)
    started = _started(ledger)
    _record_trial(ledger, started, 1)
    failed = _record_trial(ledger, started, 2, failed=True)
    try:
        _record_trial(ledger, started, 3)
        raise RuntimeError("simulated crash after trial 3")
    except RuntimeError:
        pass
    recovered = FitnessTrialLedger(tmp_path).trials_for_run(started["optimization_run_id"])
    assert [trial["trial_index"] for trial in recovered] == [1, 2, 3]
    assert recovered[1]["status"] == "failed"
    assert recovered[1]["failure_reason"] == failed["failure_reason"]


def test_summary_reconciles_trials_and_winner_link(tmp_path) -> None:
    ledger = FitnessTrialLedger(tmp_path)
    started = _started(ledger)
    loser = _record_trial(ledger, started, 1)
    winner = _record_trial(ledger, started, 2)
    _record_trial(ledger, started, 3, failed=True)
    trials = ledger.trials_for_run(started["optimization_run_id"])
    summary = build_optimization_summary(
        optimization_started=started,
        trials=trials,
        winning_trial_id=winner["trial_id"],
        winning_configuration={"sharpe_weight": 2.0},
        artifact_locations={"results": "results.csv"},
    )
    assert summary["total_attempted_trials"] == len(trials)
    assert summary["completed_trials"] == 2
    assert summary["failed_trials"] == 1
    assert summary["winning_trial_id"] != loser["trial_id"]
    assert summary["winning_trial_id"] in {trial["trial_id"] for trial in trials}
    assert summary["optimization_run_id"] != summary["parent_experiment_id"]
    assert summary["winning_trial_id"] != summary["optimization_run_id"]


def test_trial_accounting_does_not_consume_fitness_search_rng(tmp_path) -> None:
    base = FitnessConfig()
    baseline_rng = np.random.default_rng(23)
    expected = [sample_fitness_config(baseline_rng, base) for _ in range(3)]

    observed_rng = np.random.default_rng(23)
    ledger = FitnessTrialLedger(tmp_path)
    started = _started(ledger)
    observed = []
    for index in range(1, 4):
        observed.append(sample_fitness_config(observed_rng, base))
        _record_trial(ledger, started, index)

    assert observed == expected
