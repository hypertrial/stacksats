from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
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
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.is_empty():
            return TargetProfile(
                values=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
                mode="absolute",
            )
        return TargetProfile(
            values=pl.DataFrame({
                "date": features_df["date"],
                "value": pl.Series([1.0] * features_df.height),
            }),
            mode="absolute",
        )


class BadStrategy(BaseStrategy):
    strategy_id = "bad"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx, signals
        return TargetProfile(
            values=pl.DataFrame({
                "date": features_df["date"],
                "value": pl.Series([float("nan")] * features_df.height),
            }),
            mode="absolute",
        )


def _btc_df() -> pl.DataFrame:
    dates = pl.datetime_range(
        dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 1) + dt.timedelta(days=1499),
        interval="1d", eager=True
    ).to_list()
    return pl.DataFrame({
        "date": dates,
        "price_usd": np.linspace(10000, 60000, 1500),
        "mvrv": np.linspace(0.9, 2.1, 1500),
    })


@pytest.mark.slow
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
    batch = runner.export(
        UniformBaseStrategy(),
        ExportConfig(
            range_start="2023-01-01",
            range_end="2024-12-31",
            output_dir=str(tmp_path),
        ),
        btc_df=_btc_df(),
        current_date=dt.datetime(2024, 1, 15),
    )
    assert batch.row_count > 0
    flattened = batch.to_dataframe()
    assert not flattened.is_empty()
    assert {"start_date", "end_date", "date", "price_usd", "weight"}.issubset(flattened.columns)
    artifact_paths = list(tmp_path.glob("**/artifacts.json"))
    assert artifact_paths, "Expected artifacts.json in strategy-addressable output path."
    payload = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    assert payload["strategy_id"] == "test-uniform"
    assert "run_id" in payload


@pytest.mark.slow
def test_runner_uses_injected_data_provider_when_no_btc_df() -> None:
    class FakeProvider:
        def __init__(self):
            self.called = False
            self.end_date = None

        def load(self, *, backtest_start: str, end_date: str | None = None):
            self.called = True
            self.end_date = end_date
            return _btc_df()

    provider = FakeProvider()
    runner = StrategyRunner(data_provider=provider)
    result = runner.backtest(
        UniformBaseStrategy(),
        BacktestConfig(start_date="2022-01-01", end_date="2024-01-01"),
    )
    assert provider.called is True
    assert provider.end_date == "2024-01-01"
    assert result.score >= 0.0


@pytest.mark.slow
def test_runner_backtest_does_not_require_params_serialization_for_runtime_execution() -> None:
    class RuntimeOnlyParamStrategy(BaseStrategy):
        strategy_id = "runtime-only-param"
        version = "1.0.0"
        runtime_model = object()

        def build_target_profile(
            self,
            ctx: StrategyContext,
            features_df: pl.DataFrame,
            signals: dict[str, pl.Series],
        ) -> TargetProfile:
            del ctx, signals
            return TargetProfile(
                values=pl.DataFrame({
                    "date": features_df["date"],
                    "value": pl.Series([1.0] * features_df.height),
                }),
                mode="absolute",
            )

    runner = StrategyRunner()
    result = runner.backtest(
        RuntimeOnlyParamStrategy(),
        BacktestConfig(start_date="2022-01-01", end_date="2024-01-01"),
        btc_df=_btc_df(),
    )

    assert result.strategy_id == "runtime-only-param"


def test_runner_backtest_materializes_features_per_window_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    seen_current_dates: list[dt.datetime] = []
    btc_df = _btc_df()
    original_materialize = runner._materialize_strategy_features

    def _wrapped_materialize(strategy, btc_df_arg, *, start_date, end_date, current_date):
        seen_current_dates.append(current_date)
        return original_materialize(
            strategy,
            btc_df_arg,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        )

    monkeypatch.setattr(runner, "_materialize_strategy_features", _wrapped_materialize)

    runner.backtest(
        UniformBaseStrategy(),
        BacktestConfig(start_date="2023-01-01", end_date="2024-01-01"),
        btc_df=btc_df,
    )

    assert seen_current_dates
    assert len(set(seen_current_dates)) > 1
