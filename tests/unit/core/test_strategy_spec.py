from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stacksats.prelude import date_range_list
from stacksats.strategy_types import (
    BaseStrategy,
    StrategyContext,
    StrategyContractWarning,
    StrategyMetadata,
    StrategySpec,
    strategy_context_from_features_df,
)


def _context(*, start: str = "2024-01-01", periods: int = 4) -> StrategyContext:
    start_dt = dt.datetime.strptime(start[:10], "%Y-%m-%d")
    end_dt = start_dt + dt.timedelta(days=periods - 1)
    dates = date_range_list(start_dt, end_dt)
    features_df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": np.linspace(100.0, 103.0, len(dates)),
            "mvrv": np.linspace(1.0, 1.3, len(dates)),
            "price_vs_ma": np.linspace(-0.2, 0.1, len(dates)),
            "mvrv_zscore": np.linspace(-1.0, 1.0, len(dates)),
            "mvrv_gradient": np.linspace(-0.5, 0.5, len(dates)),
        }
    )
    return strategy_context_from_features_df(
        features_df,
        dates[0],
        dates[-1],
        dates[-1],
    )


class _ParamStrategy(BaseStrategy):
    strategy_id = "param-strategy"
    version = "2.0.0"
    description = "params"
    alpha = 0.5
    weights_path = Path("~/weights.csv")
    sequence = (1, np.int64(2))
    when = dt.datetime(2024, 1, 2)
    mapping = {"a": 1, "b": np.float64(2.5)}
    _private_class_value = "skip"

    def __init__(self) -> None:
        self.beta = np.float64(3.5)
        self.gamma = Path("/tmp/example")
        self._runtime_cache = {"skip": True}

    def propose_weight(self, state):
        return state.uniform_weight


def test_metadata_returns_canonical_identity() -> None:
    metadata = _ParamStrategy().metadata()

    assert metadata == StrategyMetadata(
        strategy_id="param-strategy",
        version="2.0.0",
        description="params",
    )


def test_params_include_public_class_and_instance_values() -> None:
    params = _ParamStrategy().params()

    assert params["alpha"] == 0.5
    assert params["beta"] == 3.5
    assert params["gamma"] == "/tmp/example"
    assert params["weights_path"] == str(Path("~/weights.csv"))
    assert params["sequence"] == [1, 2]
    assert params["when"] == "2024-01-02T00:00:00"
    assert params["mapping"] == {"a": 1, "b": 2.5}
    assert "strategy_id" not in params
    assert "_runtime_cache" not in params


def test_params_reject_unsupported_public_values() -> None:
    class _BadParamsStrategy(BaseStrategy):
        strategy_id = "bad-params"
        version = "1.0.0"
        bad = object()

        def propose_weight(self, state):
            return state.uniform_weight

    with pytest.raises(TypeError, match="Unsupported strategy param value"):
        _BadParamsStrategy().params()


def test_params_reject_non_finite_float_values() -> None:
    class _NonFiniteParamsStrategy(BaseStrategy):
        strategy_id = "non-finite-params"
        version = "1.0.0"
        bad = float("nan")

        def propose_weight(self, state):
            return state.uniform_weight

    with pytest.raises(TypeError, match="finite JSON-serializable"):
        _NonFiniteParamsStrategy().params()


def test_spec_returns_stable_public_contract() -> None:
    spec = _ParamStrategy().spec()

    assert isinstance(spec, StrategySpec)
    assert spec.metadata.strategy_id == "param-strategy"
    assert spec.intent_mode == "propose"
    assert spec.params["alpha"] == 0.5
    assert spec.required_feature_columns == ()


def test_intent_mode_warns_and_defaults_to_propose_for_dual_hook_strategy() -> None:
    class _DualHookStrategy(BaseStrategy):
        strategy_id = "dual"
        version = "1.0.0"

        def propose_weight(self, state):
            return state.uniform_weight

        def build_target_profile(self, ctx, features_df, signals):
            del ctx, signals
            return pl.DataFrame({"date": features_df["date"], "value": pl.lit(1.0)})

    with pytest.warns(StrategyContractWarning, match="Current fallback uses propose_weight"):
        assert _DualHookStrategy().intent_mode() == "propose"


def test_intent_mode_uses_explicit_profile_preference() -> None:
    class _ExplicitProfileStrategy(BaseStrategy):
        strategy_id = "explicit-profile"
        version = "1.0.0"
        intent_preference = "profile"

        def propose_weight(self, state):
            return state.uniform_weight

        def build_target_profile(self, ctx, features_df, signals):
            del ctx, signals
            return pl.DataFrame({"date": features_df["date"], "value": pl.lit(1.0)})

    assert _ExplicitProfileStrategy().intent_mode() == "profile"


def test_compute_weights_respects_explicit_profile_intent() -> None:
    class _TrackingDualHookStrategy(BaseStrategy):
        strategy_id = "tracking-dual"
        version = "1.0.0"
        intent_preference = "profile"

        def __init__(self) -> None:
            self.profile_called = False

        def propose_weight(self, state):
            return state.uniform_weight

        def build_target_profile(self, ctx, features_df, signals):
            del ctx, signals
            self.profile_called = True
            n = features_df.height
            return pl.DataFrame({
                "date": features_df["date"],
                "value": np.linspace(0.0, 1.0, n),
            })

    strategy = _TrackingDualHookStrategy()
    strategy.compute_weights(_context())

    assert strategy.profile_called is True


def test_missing_required_feature_columns_fail_early() -> None:
    class _RequiredFeatureStrategy(BaseStrategy):
        strategy_id = "required-cols"
        version = "1.0.0"

        def required_feature_columns(self) -> tuple[str, ...]:
            return ("missing_col",)

        def propose_weight(self, state):
            return state.uniform_weight

    with pytest.raises(ValueError, match="Missing required feature columns"):
        _RequiredFeatureStrategy().compute_weights(_context())
