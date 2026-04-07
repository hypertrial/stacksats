"""Tests for the installable strategy-first StackSats package API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import stacksats
import stacksats.export_weights as export_weights
import stacksats.export_weights as pkg_export_weights
import stacksats.strategies.mvrv as compat_mvrv
import stacksats.strategy_time_series as strategy_time_series
from stacksats import (
    AgentServiceConfig,
    BacktestConfig,
    ComparisonConfig,
    ComparisonResult,
    ComparisonRow,
    DailyDecisionResult,
    DecideDailyConfig,
    ExecutionReceiptEvent,
    ExecutionReceiptHistoryResult,
    ExecutionStatusResult,
    MVRVStrategy,
    MergedMetricsDataset,
    MetricCatalog,
    MomentumStrategy,
    RunDailyPaperStrategy,
    SimpleZScoreStrategy,
    UniformStrategy,
    WeightTimeSeries,
    WeightTimeSeriesBatch,
    get_strategy_catalog_entry,
    load_metric_catalog,
    list_strategies,
    open_merged_metrics,
)
from tests.test_helpers import btc_frame

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "snapshots" / "public_contract_snapshots.json"


def _sample_btc_df():
    return btc_frame(start="2022-01-01", days=520, price_start=20000.0, price_step=50.0).with_columns(
        mvrv=np.linspace(0.8, 2.2, 520)
    )


def test_export_module_identity():
    """Top-level export module should alias packaged module object."""
    assert export_weights is pkg_export_weights


def test_backtest_uniform_strategy():
    """Users can backtest a custom strategy through strategy methods."""
    btc_df = _sample_btc_df()
    result = UniformStrategy().backtest(
        BacktestConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            strategy_label="uniform-test",
        ),
        btc_df=btc_df,
    )

    assert result.spd_table.height > 0
    assert np.isfinite(result.win_rate)
    assert np.isfinite(result.score)


def test_backtest_default_strategy():
    """Built-in MVRV strategy is compatible with strategy methods."""
    btc_df = _sample_btc_df()

    result = MVRVStrategy().backtest(
        BacktestConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            strategy_label="mvrv-test",
        ),
        btc_df=btc_df,
    )

    assert result.spd_table.height > 0
    assert np.isfinite(result.exp_decay_percentile)


def test_removed_timeseries_aliases_are_not_exported() -> None:
    assert WeightTimeSeries is not None
    assert WeightTimeSeriesBatch is not None
    assert not hasattr(stacksats, "TimeSeries")
    assert not hasattr(stacksats, "TimeSeriesBatch")
    assert not hasattr(stacksats, "StrategyTimeSeries")
    assert not hasattr(stacksats, "StrategyTimeSeriesBatch")
    assert not hasattr(strategy_time_series, "TimeSeries")
    assert not hasattr(strategy_time_series, "TimeSeriesBatch")
    assert not hasattr(strategy_time_series, "StrategyTimeSeries")
    assert not hasattr(strategy_time_series, "StrategyTimeSeriesBatch")


def test_eda_api_is_exported() -> None:
    assert MergedMetricsDataset is stacksats.MergedMetricsDataset
    assert MetricCatalog is stacksats.MetricCatalog
    assert open_merged_metrics is stacksats.open_merged_metrics
    assert load_metric_catalog is stacksats.load_metric_catalog


def test_public_api_snapshot_matches_contract() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert sorted(stacksats.__all__) == snapshot["public_api_all"]


def test_stable_strategies_are_top_level_exports() -> None:
    expected = [
        entry.class_name
        for entry in list_strategies(tier="stable")
        if entry.public_export
    ]
    observed = [name for name in expected if hasattr(stacksats, name)]
    assert observed == expected
    assert UniformStrategy is stacksats.UniformStrategy
    assert RunDailyPaperStrategy is stacksats.RunDailyPaperStrategy
    assert SimpleZScoreStrategy is stacksats.SimpleZScoreStrategy
    assert MomentumStrategy is stacksats.MomentumStrategy
    assert MVRVStrategy is stacksats.MVRVStrategy


def test_strategy_catalog_helpers_are_top_level_exports() -> None:
    entry = get_strategy_catalog_entry("simple-zscore")
    assert entry.strategy_id == "simple-zscore"
    assert any(item.strategy_id == "mvrv-plus" for item in list_strategies(public_only=False))


def test_mvrv_compat_module_reexports_stable_strategy() -> None:
    assert compat_mvrv.MVRVStrategy is MVRVStrategy
    assert compat_mvrv.__all__ == ["MVRVStrategy"]


def test_comparison_types_are_top_level_exports() -> None:
    assert ComparisonConfig is stacksats.ComparisonConfig
    assert ComparisonResult is stacksats.ComparisonResult
    assert ComparisonRow is stacksats.ComparisonRow


def test_decision_types_are_top_level_exports() -> None:
    assert DailyDecisionResult is stacksats.DailyDecisionResult
    assert DecideDailyConfig is stacksats.DecideDailyConfig
    assert AgentServiceConfig is stacksats.AgentServiceConfig
    assert ExecutionReceiptEvent is stacksats.ExecutionReceiptEvent
    assert ExecutionReceiptHistoryResult is stacksats.ExecutionReceiptHistoryResult
    assert ExecutionStatusResult is stacksats.ExecutionStatusResult
    assert callable(stacksats.create_agent_service_app)


def test_experimental_strategies_are_not_top_level_exports() -> None:
    assert not hasattr(stacksats, "ExampleMVRVStrategy")
    assert not hasattr(stacksats, "MVRVPlusStrategy")


def test_templates_are_not_cataloged_or_top_level_exports() -> None:
    catalog_ids = {entry.strategy_id for entry in list_strategies(public_only=False)}
    assert "minimal-propose-template" not in catalog_ids
    assert "minimal-profile-template" not in catalog_ids
    assert not hasattr(stacksats, "MinimalProposeTemplateStrategy")
    assert not hasattr(stacksats, "MinimalProfileTemplateStrategy")
