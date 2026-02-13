from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from stacksats.api import BacktestResult
from stacksats.strategy_types import StrategyArtifactSet


def test_backtest_payload_includes_provenance() -> None:
    spd = pd.DataFrame(
        {
            "window": ["2024-01-01 → 2025-01-01"],
            "uniform_percentile": [50.0],
            "dynamic_percentile": [55.0],
            "dynamic_sats_per_dollar": [5000.0],
            "uniform_sats_per_dollar": [4800.0],
            "min_sats_per_dollar": [4000.0],
            "max_sats_per_dollar": [6000.0],
            "excess_percentile": [5.0],
        }
    ).set_index("window")
    payload = BacktestResult(
        spd_table=spd,
        exp_decay_percentile=55.0,
        win_rate=100.0,
        score=77.5,
        strategy_id="my-strategy",
        strategy_version="1.2.3",
        config_hash="abc123",
        run_id="run-001",
    ).to_json()
    assert payload["provenance"]["strategy_id"] == "my-strategy"
    assert payload["provenance"]["version"] == "1.2.3"
    assert payload["provenance"]["config_hash"] == "abc123"
    assert payload["provenance"]["run_id"] == "run-001"


def test_strategy_artifact_set_json_contract() -> None:
    artifact = StrategyArtifactSet(
        strategy_id="my-strategy",
        version="1.0.0",
        config_hash="abc123",
        run_id="run-42",
        output_dir="output/my-strategy/1.0.0/run-42",
        files={"weights_csv": "weights.csv"},
    )
    payload = json.loads(json.dumps(asdict(artifact)))
    required = {"strategy_id", "version", "config_hash", "run_id", "output_dir", "files"}
    assert required.issubset(payload.keys())
