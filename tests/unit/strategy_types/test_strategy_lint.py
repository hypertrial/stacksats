from __future__ import annotations

import ast

import polars as pl

from stacksats.strategy_lint import (
    _attribute_name,
    _call_argument,
    _is_negative_integer,
    _is_rolling_aggregation,
    _StrategyLintVisitor,
    _is_true_literal,
    lint_strategy_class,
    summarize_lint_findings,
)
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


class _PathIOStrategy(BaseStrategy):
    strategy_id = "lint-path-io"

    def transform_features(self, ctx):
        from pathlib import Path

        path = Path("demo.txt")
        path.read_text()
        return ctx.features.data.clone()

    def propose_weight(self, state):
        return state.uniform_weight


class _IlocWarningStrategy(BaseStrategy):
    strategy_id = "lint-iloc-negative"

    def transform_features(self, ctx):
        frame = ctx.features.data
        _ = frame.iloc[-1]
        return ctx.features.data.clone()

    def propose_weight(self, state):
        return state.uniform_weight


class _LazyEscapeStrategy(BaseStrategy):
    strategy_id = "lint-lazy-escape"

    def build_target_profile_lazy(self, ctx, features_lf):
        del ctx
        return features_lf.collect().lazy()


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
    findings = lint_strategy_class(_WarningStrategy) + lint_strategy_class(_IlocWarningStrategy)
    _, warnings = summarize_lint_findings(findings)

    assert any("tail-usage" in warning for warning in warnings)
    assert any("global-ranking" in warning for warning in warnings)
    assert any("iloc-negative" in warning for warning in warnings)


def test_lint_strategy_class_finds_path_io() -> None:
    findings = lint_strategy_class(_PathIOStrategy)
    errors, _ = summarize_lint_findings(findings)
    assert any("path-io" in error for error in errors)


def test_lint_strategy_class_rejects_lazy_eager_escape() -> None:
    findings = lint_strategy_class(_LazyEscapeStrategy)
    errors, _ = summarize_lint_findings(findings)
    assert any("lazy-eager-escape" in error for error in errors)


def test_lint_strategy_class_handles_uninspectable_source() -> None:
    DynamicStrategy = type(
        "DynamicStrategy",
        (BaseStrategy,),
        {"propose_weight": lambda self, state: state.uniform_weight},
    )

    findings = lint_strategy_class(DynamicStrategy)
    _, warnings = summarize_lint_findings(findings)

    assert any("source-unavailable" in warning for warning in warnings)


def test_strategy_lint_helper_functions_cover_edge_paths() -> None:
    node = ast.parse("obj.attr").body[0].value
    assert _attribute_name(node) == "obj.attr"
    assert _attribute_name(ast.parse("plain").body[0].value) == "plain"
    assert _attribute_name(ast.parse("1").body[0].value) == ""

    call = ast.parse("func(-1, center=True)").body[0].value
    assert _call_argument(call, 0, "periods") is not None
    assert _is_negative_integer(_call_argument(call, 0, "periods")) is True
    assert _is_true_literal(call.keywords[0].value) is True
    assert _is_negative_integer(ast.parse("1").body[0].value) is False
    keyword_only = ast.parse("func(periods=2)").body[0].value
    assert _call_argument(keyword_only, 0, "missing") is None

    rolling_call = ast.parse("series.rolling(3).quantile(0.5)").body[0].value
    plain_call = ast.parse("series.quantile(0.5)").body[0].value
    name_call = ast.parse("func()").body[0].value
    assert _is_rolling_aggregation(rolling_call) is True
    assert _is_rolling_aggregation(plain_call) is False
    assert _is_rolling_aggregation(name_call) is False


def test_lint_strategy_class_does_not_warn_for_nonnegative_iloc() -> None:
    source = "\n".join(
        [
            "class PositiveIlocStrategy:",
            "    def transform_features(self, ctx):",
            f"        _ = ctx.features.data.i{'loc'}[0]",
            "        return ctx.features.data",
        ]
    )
    visitor = _StrategyLintVisitor()
    visitor.visit(ast.parse(source))
    _, warnings = summarize_lint_findings(visitor.findings)
    assert not any("iloc-negative" in warning for warning in warnings)
