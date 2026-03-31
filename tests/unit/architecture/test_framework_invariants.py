from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stacksats.export_weights import process_start_date_batch
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS, validate_span_length
from stacksats.model_development import (
    allocate_from_proposals,
    allocate_sequential_stable,
    compute_weights_from_proposals,
)
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import (
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    strategy_context_from_features_df,
)


def _date_range(start: datetime, days: int) -> list[datetime]:
    return [start + timedelta(days=offset) for offset in range(days)]


def _context(days: int = 5, *, current_day_index: int | None = None) -> StrategyContext:
    start = datetime(2024, 1, 1)
    dates = _date_range(start, days)
    features_df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": np.linspace(100.0, 104.0, len(dates)),
            "mvrv_zscore": np.linspace(-1.0, 1.0, len(dates)),
        }
    )
    current_idx = (days - 1) if current_day_index is None else current_day_index
    return strategy_context_from_features_df(
        features_df,
        dates[0],
        dates[-1],
        dates[current_idx],
    )


def test_per_day_feasibility_projection() -> None:
    proposals = np.array([2.0, -1.0, 1.5, 0.5], dtype=float)
    weights = allocate_from_proposals(proposals, n_past=4, n_total=4)
    running_sum = 0.0
    for weight in weights:
        assert weight >= 0.0
        running_sum += float(weight)
        assert running_sum <= 1.0 + 1e-12
    assert np.isclose(weights.sum(), 1.0)


def test_locked_weights_are_immutable_when_present() -> None:
    proposals = np.array([0.8, 0.8, 0.8, 0.8], dtype=float)
    locked = np.array([0.1, 0.2], dtype=float)
    weights = allocate_from_proposals(proposals, n_past=2, n_total=4, locked_weights=locked)
    assert np.isclose(weights[0], 0.1)
    assert np.isclose(weights[1], 0.2)
    assert np.isclose(weights.sum(), 1.0)


def test_partial_locked_prefix_supported_in_stable_kernel() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    locked_prefix = np.array([0.10, 0.20], dtype=float)
    weights = allocate_sequential_stable(raw, n_past=3, locked_weights=locked_prefix)
    assert np.isclose(weights[0], locked_prefix[0])
    assert np.isclose(weights[1], locked_prefix[1])
    assert weights[2] >= 0.0
    assert np.isclose(weights.sum(), 1.0)


def test_span_length_contract_requires_fixed_configured_rows() -> None:
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=ALLOCATION_SPAN_DAYS - 1)
    assert validate_span_length(start, end) == ALLOCATION_SPAN_DAYS
    with pytest.raises(ValueError, match="fixed span"):
        validate_span_length(start, end - timedelta(days=1))


def test_budget_exhaustion_still_sums_to_one() -> None:
    dates = _date_range(datetime(2024, 1, 1), 3)
    proposals = pl.DataFrame({"date": dates, "value": [10.0, 10.0, 10.0]})
    weights = compute_weights_from_proposals(
        proposals=proposals,
        start_date=dates[0],
        end_date=dates[-1],
        n_past=3,
    )
    assert np.isclose(float(weights["weight"].sum()), 1.0)
    assert (weights["weight"] >= 0).all()


class _ProposeStrategy(BaseStrategy):
    def propose_weight(self, state) -> float:
        del state
        return np.inf


def test_nan_inf_rejection_in_propose_hook() -> None:
    with pytest.raises(ValueError, match="finite numeric value"):
        _ProposeStrategy().compute_weights(_context(days=3))


class _BothHooksStrategy(BaseStrategy):
    intent_preference = "propose"

    def propose_weight(self, state) -> float:
        return state.uniform_weight

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx, signals
        values = [1.0] + [0.0] * (features_df.height - 1)
        return TargetProfile(
            values=pl.DataFrame({"date": features_df["date"], "value": values}),
            mode="absolute",
        )


def test_propose_hook_takes_precedence_over_batch_hook() -> None:
    weights = _BothHooksStrategy().compute_weights(_context(days=4))
    assert weights["weight"].to_list() == pytest.approx([0.25, 0.25, 0.25, 0.25])


class _ProfileOnlyStrategy(BaseStrategy):
    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del ctx, signals
        return pl.DataFrame(
            {
                "date": features_df["date"],
                "value": np.linspace(0.0, 1.0, features_df.height),
            }
        )


def test_build_target_profile_path_still_valid() -> None:
    weights = _ProfileOnlyStrategy().compute_weights(_context(days=5))
    assert np.isclose(float(weights["weight"].sum()), 1.0)
    assert (weights["weight"] >= 0.0).all()


class _NoIntentStrategy(BaseStrategy):
    pass


def test_strategy_must_implement_at_least_one_intent_hook() -> None:
    with pytest.raises(TypeError, match="must implement propose_weight"):
        _NoIntentStrategy().compute_weights(_context(days=3))


class _IllegalComputeWeightsStrategy(BaseStrategy):
    def propose_weight(self, state) -> float:
        return state.uniform_weight

    def compute_weights(self, ctx: StrategyContext) -> pl.DataFrame:  # type: ignore[override]
        del ctx
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})


def test_runner_rejects_compute_weights_override() -> None:
    runner = StrategyRunner()
    with pytest.raises(TypeError, match="Custom compute_weights overrides"):
        runner._validate_strategy_contract(_IllegalComputeWeightsStrategy())


def test_export_batch_rejects_compute_weights_override() -> None:
    dates = _date_range(datetime(2024, 1, 1), 3)
    features_df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": np.linspace(100.0, 102.0, len(dates)),
            "mvrv_zscore": np.linspace(-1.0, 1.0, len(dates)),
        }
    )
    btc_df = pl.DataFrame({"date": dates, "price_usd": np.linspace(100.0, 102.0, len(dates))})
    with pytest.raises(TypeError, match="Custom compute_weights overrides"):
        process_start_date_batch(
            start_date=dates[0],
            end_dates=[dates[-1]],
            features_df=features_df,
            btc_df=btc_df,
            current_date=dates[-1],
            btc_price_col="price_usd",
            strategy=_IllegalComputeWeightsStrategy(),
            enforce_span_contract=False,
        )
