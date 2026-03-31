from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import polars as pl

from stacksats.strategy_types import (
    AgentServiceConfig,
    BacktestConfig,
    BaseStrategy,
    DecideDailyConfig,
    ExportConfig,
    StrategyArtifactSet,
    StrategyMetadata,
    StrategySpec,
    ValidationConfig,
    strategy_context_from_features_df,
)


def _features_df(days: int = 3) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=offset) for offset in range(days)]
    return pl.DataFrame({"date": dates, "price_usd": list(range(1, days + 1))})


def test_strategy_context_defaults() -> None:
    frame = _features_df()
    ctx = strategy_context_from_features_df(
        frame,
        frame["date"][0],
        frame["date"][-1],
        frame["date"][-1],
    )
    assert ctx.btc_price_col == "price_usd"
    assert ctx.mvrv_col == "mvrv"


def test_config_defaults() -> None:
    assert BacktestConfig().output_dir == "output"
    assert ValidationConfig().min_win_rate == 50.0
    export_config = ExportConfig()
    assert export_config.output_dir == "output"
    start = datetime.strptime(export_config.range_start, "%Y-%m-%d")
    end = datetime.strptime(export_config.range_end, "%Y-%m-%d")
    assert end >= start
    assert (end - start).days in {364, 365}


def test_default_run_daily_validation_config_is_strict() -> None:
    config = BaseStrategy().default_run_daily_validation_config()
    assert config.min_win_rate == 50.0
    assert config.strict is True


def test_default_decide_daily_config_uses_documented_defaults() -> None:
    config = BaseStrategy().default_decide_daily_config()
    assert isinstance(config, DecideDailyConfig)
    assert config.total_window_budget_usd == 1000.0
    assert config.btc_price_col == "price_usd"
    assert config.output_dir == "output"
    assert config.force is False
    assert config.state_db_path.endswith(".stacksats/run_state.sqlite3")


def test_agent_service_config_uses_documented_defaults() -> None:
    config = AgentServiceConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.output_dir == "output"
    assert config.auth_token_env == "STACKSATS_AGENT_API_TOKEN"
    assert config.btc_price_col_default == "price_usd"
    assert config.registry_path.endswith(".stacksats/agent_service_registry.json")
    assert config.state_db_path.endswith(".stacksats/run_state.sqlite3")


def test_strategy_artifact_set_fields() -> None:
    artifacts = StrategyArtifactSet(
        schema_version="1.0.0",
        strategy_id="my-strategy",
        version="1.0.0",
        config_hash="abc123",
        run_id="run-1",
        output_dir="output/my-strategy/1.0.0/run-1",
        files={"weights_csv": "weights.csv"},
    )
    assert artifacts.schema_version == "1.0.0"
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
    frame = _features_df(days=2)
    ctx = strategy_context_from_features_df(
        frame,
        frame["date"][0],
        frame["date"][-1],
        frame["date"][-1],
    )
    try:
        ctx.btc_price_col = "other"
    except FrozenInstanceError:
        return
    raise AssertionError("StrategyContext should be immutable.")
