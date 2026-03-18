"""Tests for strategy_types lazy profile validation and error paths."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from stacksats.strategy_types import (
    BaseStrategy,
    StrategyLazyContext,
    strategy_context_from_features_df,
)


def _context(*, start: str = "2024-01-01", periods: int = 3):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=offset) for offset in range(periods)]
    frame = pl.DataFrame(
        {"date": dates, "price_usd": [100.0] * len(dates), "mvrv": [1.0] * len(dates)}
    )
    return strategy_context_from_features_df(
        frame,
        frame["date"][0],
        frame["date"][-1],
        frame["date"][-1],
    )


def test_build_target_profile_lazy_base_raises_not_implemented() -> None:
    """BaseStrategy.build_target_profile_lazy raises NotImplementedError."""
    ctx = StrategyLazyContext.from_strategy_context(_context())
    features_lf = pl.DataFrame({"date": [], "price_usd": [], "mvrv": []}).lazy()
    with pytest.raises(NotImplementedError, match="Override build_target_profile_lazy"):
        BaseStrategy().build_target_profile_lazy(ctx, features_lf)


def test_transform_features_lazy_must_return_lazy_frame() -> None:
    """transform_features_lazy returning non-LazyFrame raises TypeError."""

    class BadTransformLazyStrategy(BaseStrategy):
        strategy_id = "bad-transform-lazy"

        def transform_features_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.collect()

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.select("date", pl.lit(1.0).alias("value"))

    with pytest.raises(TypeError, match="transform_features_lazy must return a polars LazyFrame"):
        BadTransformLazyStrategy().compute_weights(_context())


def test_build_signal_exprs_must_return_dict() -> None:
    """build_signal_exprs returning non-dict raises TypeError."""

    class BadSignalExprsStrategy(BaseStrategy):
        strategy_id = "bad-signal-exprs"

        def build_signal_exprs(self, ctx, schema):
            del ctx, schema
            return []

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.select("date", pl.lit(1.0).alias("value"))

    with pytest.raises(TypeError, match="build_signal_exprs must return a dict"):
        BadSignalExprsStrategy().compute_weights(_context())


def test_build_signal_exprs_empty_key_raises_value_error() -> None:
    """build_signal_exprs with empty string key raises ValueError."""

    class EmptyKeyStrategy(BaseStrategy):
        strategy_id = "empty-key"

        def build_signal_exprs(self, ctx, schema):
            del ctx, schema
            return {"": pl.col("price_usd")}

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.select("date", pl.lit(1.0).alias("value"))

    with pytest.raises(ValueError, match="non-empty strings"):
        EmptyKeyStrategy().compute_weights(_context())


def test_build_signal_exprs_non_expr_value_raises_type_error() -> None:
    """build_signal_exprs with non-Expr value raises TypeError."""

    class NonExprStrategy(BaseStrategy):
        strategy_id = "non-expr"

        def build_signal_exprs(self, ctx, schema):
            del ctx, schema
            return {"x": pl.Series("x", [1.0, 2.0, 3.0])}

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.select("date", pl.lit(1.0).alias("value"))

    with pytest.raises(TypeError, match="must be a polars expression"):
        NonExprStrategy().compute_weights(_context())


def test_build_target_profile_lazy_wrong_return_type_raises_type_error() -> None:
    """build_target_profile_lazy returning wrong type raises TypeError."""

    class WrongReturnStrategy(BaseStrategy):
        strategy_id = "wrong-return"

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx, features_lf
            return 42

    with pytest.raises(TypeError, match="LazyFrame or TargetProfile"):
        WrongReturnStrategy().compute_weights(_context())


def test_resolve_lazy_target_profile_target_df_must_be_dataframe() -> None:
    """_resolve_lazy_target_profile when target is not DataFrame raises TypeError.

    build_target_profile_lazy can return TargetProfile. If values is not a DataFrame,
    the TypeError is raised.
    """

    class BadTargetProfileStrategy(BaseStrategy):
        strategy_id = "bad-target-profile"

        def build_target_profile_lazy(self, ctx, features_lf):
            from stacksats.strategy_types import TargetProfile

            del ctx, features_lf
            return TargetProfile(values=pl.Series("x", [1.0, 2.0, 3.0]), mode="absolute")

    with pytest.raises(TypeError, match="target profile must be a Polars DataFrame"):
        BadTargetProfileStrategy().compute_weights(_context())


def test_lazy_profile_empty_target_returns_empty_weights() -> None:
    """build_target_profile_lazy returning empty target yields empty weights."""

    class EmptyTargetStrategy(BaseStrategy):
        strategy_id = "empty-target"

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx, features_lf
            return pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}).lazy()

    weights = EmptyTargetStrategy().compute_weights(_context())
    assert weights.is_empty()
    assert weights.schema == {"date": pl.Datetime("us"), "weight": pl.Float64}


def test_lazy_profile_empty_date_range_returns_empty_weights() -> None:
    """_compute_weights_lazy_profile_from_features_lf with empty date_range returns empty."""

    class SimpleLazyStrategy(BaseStrategy):
        strategy_id = "simple-lazy"

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.select("date", pl.lit(1.0).alias("value"))

    # start_date > end_date yields empty datetime_range
    frame = pl.DataFrame(
        {"date": [datetime(2024, 1, 1)], "price_usd": [100.0], "mvrv": [1.0]}
    )
    ctx = strategy_context_from_features_df(
        frame,
        start_date=datetime(2024, 1, 5),
        end_date=datetime(2024, 1, 1),
        current_date=datetime(2024, 1, 5),
    )
    weights = SimpleLazyStrategy().compute_weights(ctx)
    assert weights.is_empty()
    assert weights.schema == {"date": pl.Datetime("us"), "weight": pl.Float64}
