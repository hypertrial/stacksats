from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from stacksats import (
    BacktestConfig,
    BaseStrategy,
    DayState,
    StrategyContext,
    StrategyRunner,
    TargetProfile,
    ValidationConfig,
)
from tests.test_helpers import btc_frame

REPO_ROOT = Path(__file__).resolve().parents[3]
FIRST_STRATEGY_RUN_DOC = REPO_ROOT / "docs" / "start" / "first-strategy-run.md"
MINIMAL_STRATEGY_EXAMPLES_DOC = REPO_ROOT / "docs" / "start" / "minimal-strategy-examples.md"
TASKS_DOC = REPO_ROOT / "docs" / "tasks.md"
VALIDATE_COMMAND_DOC = REPO_ROOT / "docs" / "run" / "validate.md"


def _onboarding_btc_df() -> pl.DataFrame:
    return btc_frame(
        start="2022-01-01",
        days=1461,
        price_start=20000.0,
        price_step=40.0,
    )


class FirstStrategyRunStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    description = "First custom strategy."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
    ) -> dict[str, pl.Series]:
        del ctx
        value_signal = (-features_df["mvrv_zscore"]).clip(-4, 4)
        trend_signal = (-features_df["price_vs_ma"]).clip(-1, 1)
        return {"value": value_signal, "trend": trend_signal}

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx
        preference = (0.7 * signals["value"]) + (0.3 * signals["trend"])
        return TargetProfile(
            values=pl.DataFrame({"date": features_df["date"], "value": preference}),
            mode="preference",
        )


class MinimalProposeWeightStrategy(BaseStrategy):
    strategy_id = "minimal-propose-weight"
    version = "1.0.0"
    description = "Minimal strategy using propose_weight(state)."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def propose_weight(self, state: DayState) -> float:
        mvrv = (
            float(state.features["mvrv_zscore"][0])
            if "mvrv_zscore" in state.features.columns
            else 0.0
        )
        scale = 1.0 + (-0.05 * mvrv)
        proposal = state.uniform_weight * scale
        return max(0.0, min(state.remaining_budget, proposal))


class MinimalTargetProfileStrategy(BaseStrategy):
    strategy_id = "minimal-target-profile"
    version = "1.0.0"
    description = "Minimal strategy using build_target_profile(...)."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("mvrv_zscore", "price_vs_ma")

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
    ) -> dict[str, pl.Series]:
        del ctx
        value_signal = (-features_df["mvrv_zscore"]).clip(-4, 4)
        trend_signal = (-features_df["price_vs_ma"]).clip(-1, 1)
        return {"value": value_signal, "trend": trend_signal}

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx
        preference = (0.7 * signals["value"]) + (0.3 * signals["trend"])
        return TargetProfile(
            values=pl.DataFrame({"date": features_df["date"], "value": preference}),
            mode="preference",
        )


def test_first_strategy_run_profile_example_backtests_and_strict_validates() -> None:
    btc_df = _onboarding_btc_df()
    runner = StrategyRunner.from_dataframe(btc_df)
    strategy = FirstStrategyRunStrategy()

    backtest = runner.backtest(
        strategy,
        BacktestConfig(start_date="2024-01-01", end_date="2024-12-31"),
    )
    validation = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
            min_win_rate=0.0,
            strict=True,
        ),
        btc_df=btc_df,
    )

    assert backtest.spd_table.height > 0
    assert math.isfinite(backtest.score)
    assert bool(validation.forward_leakage_ok) is True
    assert validation.summary()


def test_minimal_propose_example_backtests_from_dataframe_runner() -> None:
    runner = StrategyRunner.from_dataframe(_onboarding_btc_df())

    result = runner.backtest(
        MinimalProposeWeightStrategy(),
        BacktestConfig(start_date="2024-01-01", end_date="2024-12-31"),
    )

    assert result.spd_table.height > 0
    assert math.isfinite(result.score)


def test_minimal_profile_example_backtests_from_dataframe_runner() -> None:
    runner = StrategyRunner.from_dataframe(_onboarding_btc_df())

    result = runner.backtest(
        MinimalTargetProfileStrategy(),
        BacktestConfig(start_date="2024-01-01", end_date="2024-12-31"),
    )

    assert result.spd_table.height > 0
    assert math.isfinite(result.score)


def test_onboarding_docs_use_runnable_polars_and_explicit_python_strict_mode() -> None:
    first_strategy_text = FIRST_STRATEGY_RUN_DOC.read_text(encoding="utf-8")
    minimal_examples_text = MINIMAL_STRATEGY_EXAMPLES_DOC.read_text(encoding="utf-8")

    assert "return ctx.features_df.clone()" in first_strategy_text
    assert "strict=True" in first_strategy_text
    assert ".copy()" not in first_strategy_text

    assert "return ctx.features_df.clone()" in minimal_examples_text
    assert 'float(state.features["mvrv_zscore"][0])' in minimal_examples_text
    assert '"date": features_df["date"]' in minimal_examples_text
    assert ".copy()" not in minimal_examples_text
    assert "ValidationConfig(strict=True, ...)" in minimal_examples_text


def test_cli_docs_qualify_strict_default_as_cli_behavior() -> None:
    tasks_text = TASKS_DOC.read_text(encoding="utf-8")
    validate_command_text = VALIDATE_COMMAND_DOC.read_text(encoding="utf-8")

    assert "Strict validation runs by default in this CLI flow" in tasks_text
    assert "Strict validation is enabled by default for this CLI command." in (
        validate_command_text
    )
