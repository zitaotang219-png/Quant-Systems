from __future__ import annotations

import json

import pandas as pd

from alpha_mining.search_funnel import build_search_funnel
from utils.experiment_ledger import ExperimentLedger, research_configuration_fingerprint


def test_configuration_fingerprint_semantics() -> None:
    base={"seed":42,"gp":{"population_size":10},"output_dir":"one","timestamp_utc":"x"}
    assert research_configuration_fingerprint(base)==research_configuration_fingerprint({**base,"output_dir":"two","timestamp_utc":"y"})
    assert research_configuration_fingerprint(base)!=research_configuration_fingerprint({**base,"seed":43})


def test_append_only_started_completed_and_failed_lifecycle(tmp_path) -> None:
    ledger=ExperimentLedger(tmp_path); started=ledger.start(configuration={"seed":42},metadata={"cli":"test"}); ledger.complete(started,results={"selected_factors":["x"]})
    failed=ledger.start(configuration={"seed":43},metadata={})
    try: raise ValueError("boom")
    except ValueError as error: ledger.fail(failed,error)
    records=[json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [item["status"] for item in records]==["started","completed","started","failed"]
    assert records[-1]["experiment_id"]==failed["experiment_id"]


def test_funnel_reconciles_stage_events() -> None:
    events=pd.DataFrame([
        {"window":"window_1","stage":"raw_gp_population","status":"passed","individual_id":"a","expression_hash":"h1","reject_reason":""},
        {"window":"window_1","stage":"fast_filter","status":"passed","individual_id":"a","expression_hash":"h1","reject_reason":""},
        {"window":"window_1","stage":"fast_filter","status":"rejected","individual_id":"b","expression_hash":"h2","reject_reason":"variance"},
        {"window":"window_1","stage":"exact_expression_dedup","status":"passed","individual_id":"a","expression_hash":"h1","reject_reason":""},
    ])
    funnel=build_search_funnel(events).set_index("stage")
    assert funnel.at["fast_filter","input_count"]==2
    assert funnel.at["fast_filter","pass_count"]==1
    assert funnel.at["fast_filter","reject_count"]==1
    assert funnel.at["fast_filter","rejection_reasons"]=="variance"
