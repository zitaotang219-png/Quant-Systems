from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from alpha_mining.config import GPConfig
from alpha_mining.gp_generator import GPGenerator
from utils.experiment_manifest import ExperimentManifest, config_hash, create_experiment_directory
from utils.random_state import set_global_seed


def _run_seeded_research(seed: int) -> tuple[list[str], list[str], dict[str, float]]:
    set_global_seed(seed)
    generator = GPGenerator(
        GPConfig(
            population_size=12,
            generations=1,
            max_depth=4,
            init_max_depth=3,
            seed=seed,
            field_names=("volume", "momentum_5", "volatility_10"),
            disallowed_raw_field_names=(),
        )
    )
    generated = [generator.random_tree().describe() for _ in range(12)]
    selected = sorted(dict.fromkeys(generated))[:4]
    metrics = {
        "python_random": random.random(),
        "numpy_random": float(np.random.random()),
        "factor_count": float(len(selected)),
    }
    return generated, selected, metrics


def test_same_seed_same_output() -> None:
    first_generated, first_selected, first_metrics = _run_seeded_research(1729)
    second_generated, second_selected, second_metrics = _run_seeded_research(1729)

    assert first_generated == second_generated
    assert first_selected == second_selected
    assert first_metrics == second_metrics


def test_manifest_created(tmp_path: Path) -> None:
    input_file = tmp_path / "panel.csv"
    input_file.write_text("date,symbol,close\n2026-01-01,BTCUSDT,100\n", encoding="utf-8")
    run_dir = create_experiment_directory(tmp_path / "research_run")
    manifest = ExperimentManifest(
        run_dir,
        random_seed=42,
        cli_command="python -m alpha_mining.run_crypto_workflow --seed 42",
        input_paths=[input_file],
        dataset_metadata={"row_count": 1, "symbol_count": 1},
    )
    manifest.set_config({"seed": 42, "parameters": {"lookback": 20}})
    manifest.set_research_results(generated_factor_count=12, selected_factors=["rank(volume)"])
    manifest_path = manifest.save(status="completed")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.exists()
    assert (run_dir / "config.yaml").exists()
    assert payload["git"]["commit_hash"]
    assert payload["experiment"]["random_seed"] == 42
    assert payload["runtime"]["packages"]["numpy"] == np.__version__
    assert payload["data"]["inputs"][0]["sha256"]
    assert payload["research"]["selected_factor_count"] == 1
    assert payload["config_hash"] == config_hash({"seed": 42, "parameters": {"lookback": 20}})


def test_config_hash_changes_when_config_changes() -> None:
    baseline = {"seed": 42, "portfolio": {"gross_leverage": 0.8, "symbols": ("BTC", "ETH")}}
    same_values_different_order = {"portfolio": {"symbols": ["BTC", "ETH"], "gross_leverage": 0.8}, "seed": 42}
    changed = {"seed": 42, "portfolio": {"gross_leverage": 0.9, "symbols": ("BTC", "ETH")}}

    assert config_hash(baseline) == config_hash(same_values_different_order)
    assert config_hash(baseline) != config_hash(changed)


def test_experiment_directory_does_not_overwrite(tmp_path: Path) -> None:
    requested = tmp_path / "experiment"
    first = create_experiment_directory(requested)
    marker = first / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    second = create_experiment_directory(requested)

    assert second != first
    assert marker.read_text(encoding="utf-8") == "original"
