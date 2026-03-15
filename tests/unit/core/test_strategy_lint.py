from __future__ import annotations

import polars as pl

from stacksats.strategy_lint import lint_strategy_class, summarize_lint_findings
from stacksats.strategy_types import BaseStrategy


class _NegativeShiftStrategy(BaseStrategy):
    strategy_id = "lint-negative-shift"

    def transform_features(self, ctx):
        features = ctx.features.data.clone()
        features = features.with_columns(
            pl.col("price_usd").shift(-1).alias("bad")
        )
        return features

    def propose_weight(self, state):
        return state.uniform_weight


class _CenteredRollingStrategy(BaseStrategy):
    strategy_id = "lint-centered"

    def transform_features(self, ctx):
        features = ctx.features.data.clone()
        features = features.with_columns(
            pl.col("price_usd").rolling_mean(window_size=3, center=True).alias("bad")
        )
        return features

    def propose_weight(self, state):
        return state.uniform_weight


class _ExternalIOStrategy(BaseStrategy):
    strategy_id = "lint-io"

    def transform_features(self, ctx):
        pl.read_csv("demo.csv")
        return ctx.features.data.clone()

    def propose_weight(self, state):
        return state.uniform_weight


class _WarningStrategy(BaseStrategy):
    strategy_id = "lint-warning"

    def transform_features(self, ctx):
        features = ctx.features.data.clone()
        _ = features.tail(1)
        _ = features["price_usd"].quantile(0.5)
        return features

    def propose_weight(self, state):
        return state.uniform_weight


def test_lint_strategy_class_finds_hard_errors() -> None:
    findings = lint_strategy_class(_NegativeShiftStrategy)
    errors, _ = summarize_lint_findings(findings)

    assert any("negative-shift" in error for error in errors)


def test_lint_strategy_class_finds_centered_rolling_and_io() -> None:
    findings = lint_strategy_class(_CenteredRollingStrategy) + lint_strategy_class(
        _ExternalIOStrategy
    )
    errors, _ = summarize_lint_findings(findings)

    assert any("centered-rolling" in error for error in errors)
    assert any("external-io" in error for error in errors)


def test_lint_strategy_class_emits_warnings() -> None:
    findings = lint_strategy_class(_WarningStrategy)
    _, warnings = summarize_lint_findings(findings)

    assert any("tail-usage" in warning for warning in warnings)
    assert any("global-ranking" in warning for warning in warnings)


def test_lint_strategy_class_handles_uninspectable_source() -> None:
    DynamicStrategy = type(
        "DynamicStrategy",
        (BaseStrategy,),
        {"propose_weight": lambda self, state: state.uniform_weight},
    )

    findings = lint_strategy_class(DynamicStrategy)
    _, warnings = summarize_lint_findings(findings)

    assert any("source-unavailable" in warning for warning in warnings)
