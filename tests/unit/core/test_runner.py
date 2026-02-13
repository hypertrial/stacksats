from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import (
    BacktestConfig,
    BaseStrategy,
    ExportConfig,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)


class UniformBaseStrategy(BaseStrategy):
    strategy_id = "test-uniform"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="absolute")
        return TargetProfile(
            values=pd.Series(np.ones(len(features_df.index)), index=features_df.index),
            mode="absolute",
        )


class BadStrategy(BaseStrategy):
    strategy_id = "bad"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del ctx, signals
        return pd.Series(np.nan, index=features_df.index)


def _btc_df() -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=1500, freq="D")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000, 60000, len(idx)),
            "CapMVRVCur": np.linspace(0.9, 2.1, len(idx)),
        },
        index=idx,
    )


def test_runner_backtest_with_uniform_strategy() -> None:
    runner = StrategyRunner()
    result = runner.backtest(
        UniformBaseStrategy(),
        BacktestConfig(start_date="2022-01-01", end_date="2024-01-01"),
        btc_df=_btc_df(),
    )
    assert result.win_rate >= 0.0
    assert result.strategy_id == "test-uniform"
    assert result.run_id


def test_runner_raises_profile_validation_error() -> None:
    runner = StrategyRunner()
    with pytest.raises(ValueError, match="target profile must contain finite numeric values"):
        runner.backtest(
            BadStrategy(),
            BacktestConfig(start_date="2022-01-01", end_date="2024-01-01"),
            btc_df=_btc_df(),
        )


def test_runner_validate_empty_range_returns_failure_result() -> None:
    runner = StrategyRunner()
    result = runner.validate(
        UniformBaseStrategy(),
        ValidationConfig(start_date="2050-01-01", end_date="2050-02-01"),
        btc_df=_btc_df(),
    )
    assert result.passed is False
    assert any("No data available" in msg for msg in result.messages)


def test_runner_export_writes_artifacts(tmp_path) -> None:
    runner = StrategyRunner()
    df = runner.export(
        UniformBaseStrategy(),
        ExportConfig(
            range_start="2023-01-01",
            range_end="2024-12-31",
            output_dir=str(tmp_path),
        ),
        btc_df=_btc_df(),
        current_date=pd.Timestamp("2024-01-15"),
    )
    assert not df.empty
    artifact_paths = list(tmp_path.glob("**/artifacts.json"))
    assert artifact_paths, "Expected artifacts.json in strategy-addressable output path."
    payload = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    assert payload["strategy_id"] == "test-uniform"
    assert "run_id" in payload


def test_runner_uses_injected_data_provider_when_no_btc_df() -> None:
    class FakeProvider:
        def __init__(self):
            self.called = False

        def load(self, *, backtest_start: str):
            self.called = True
            return _btc_df()

    provider = FakeProvider()
    runner = StrategyRunner(data_provider=provider)
    result = runner.backtest(
        UniformBaseStrategy(),
        BacktestConfig(start_date="2022-01-01", end_date="2024-01-01"),
    )
    assert provider.called is True
    assert result.score >= 0.0
