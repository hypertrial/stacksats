from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd

from stacksats.strategy_types import (
    BacktestConfig,
    ExportConfig,
    StrategyArtifactSet,
    StrategyMetadata,
    StrategySpec,
    ValidationConfig,
    strategy_context_from_features_df,
)


def test_strategy_context_defaults() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    ctx = strategy_context_from_features_df(
        pd.DataFrame({"price_usd": [1, 2, 3]}, index=idx),
        idx.min(),
        idx.max(),
        idx.max(),
    )
    assert ctx.btc_price_col == "price_usd"
    assert ctx.mvrv_col == "mvrv"


def test_config_defaults() -> None:
    assert BacktestConfig().output_dir == "output"
    assert ValidationConfig().min_win_rate == 50.0
    export_config = ExportConfig()
    assert export_config.output_dir == "output"
    start = pd.to_datetime(export_config.range_start)
    end = pd.to_datetime(export_config.range_end)
    assert end >= start
    assert (end - start).days in {364, 365}


def test_strategy_artifact_set_fields() -> None:
    artifacts = StrategyArtifactSet(
        strategy_id="my-strategy",
        version="1.0.0",
        config_hash="abc123",
        run_id="run-1",
        output_dir="output/my-strategy/1.0.0/run-1",
        files={"weights_csv": "weights.csv"},
    )
    assert artifacts.files["weights_csv"] == "weights.csv"


def test_strategy_metadata_and_spec_dataclasses() -> None:
    metadata = StrategyMetadata(strategy_id="my-strategy", version="1.0.0", description="demo")
    spec = StrategySpec(
        metadata=metadata,
        intent_mode="profile",
        params={"alpha": 1},
        required_feature_columns=("price_vs_ma",),
    )

    assert spec.metadata.strategy_id == "my-strategy"
    assert spec.params["alpha"] == 1


def test_strategy_context_is_frozen() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    ctx = strategy_context_from_features_df(
        pd.DataFrame({"price_usd": [1, 2]}, index=idx),
        idx.min(),
        idx.max(),
        idx.max(),
    )
    try:
        ctx.btc_price_col = "other"
    except FrozenInstanceError:
        return
    raise AssertionError("StrategyContext should be immutable.")
