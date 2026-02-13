from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.framework_contract import validate_span_length
from stacksats.export_weights import process_start_date_batch
from stacksats.model_development import (
    allocate_sequential_stable,
    allocate_from_proposals,
    compute_weights_from_proposals,
)
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BaseStrategy, StrategyContext, TargetProfile


def _context(days: int = 5, *, current_day_index: int | None = None) -> StrategyContext:
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    features_df = pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(100.0, 104.0, len(idx)),
            "mvrv_zscore": np.linspace(-1.0, 1.0, len(idx)),
        },
        index=idx,
    )
    if current_day_index is None:
        current_day_index = days - 1
    return StrategyContext(
        features_df=features_df,
        start_date=idx.min(),
        end_date=idx.max(),
        current_date=idx[current_day_index],
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


def test_partial_locked_prefix_computes_today_from_intent() -> None:
    proposals = np.array([0.25, 0.25, 0.90, 0.90], dtype=float)
    locked_prefix = np.array([0.20, 0.10], dtype=float)
    weights = allocate_from_proposals(
        proposals,
        n_past=3,
        n_total=4,
        locked_weights=locked_prefix,
    )
    assert np.isclose(weights[0], 0.20)
    assert np.isclose(weights[1], 0.10)
    assert np.isclose(weights[2], 0.70)
    assert np.isclose(weights.sum(), 1.0)


def test_partial_locked_prefix_supported_in_stable_kernel() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    locked_prefix = np.array([0.10, 0.20], dtype=float)
    weights = allocate_sequential_stable(raw, n_past=3, locked_weights=locked_prefix)
    # Stable kernel uses locked_weights only when the full past prefix is provided.
    assert not np.isclose(weights[0], locked_prefix[0])
    assert not np.isclose(weights[1], locked_prefix[1])
    assert weights[2] >= 0.0
    assert np.isclose(weights.sum(), 1.0)


def test_immutability_after_lock_with_changed_intent() -> None:
    locked_prefix = np.array([0.20, 0.10], dtype=float)
    weights_a = allocate_from_proposals(
        np.array([0.2, 0.1, 0.2, 0.8], dtype=float),
        n_past=3,
        n_total=4,
        locked_weights=locked_prefix,
    )
    weights_b = allocate_from_proposals(
        np.array([0.9, 0.9, 0.9, 0.1], dtype=float),
        n_past=3,
        n_total=4,
        locked_weights=locked_prefix,
    )
    np.testing.assert_allclose(weights_a[:2], locked_prefix, atol=1e-12)
    np.testing.assert_allclose(weights_b[:2], locked_prefix, atol=1e-12)


def test_span_length_contract_accepts_365_366_or_367_rows() -> None:
    start = pd.Timestamp("2024-01-01")
    end_366 = pd.Timestamp("2024-12-31")
    end_365 = pd.Timestamp("2023-12-31")
    end_367 = pd.Timestamp("2025-01-01")
    assert validate_span_length(start, end_366) == 366
    assert validate_span_length(pd.Timestamp("2023-01-01"), end_365) == 365
    assert validate_span_length(start, end_367) == 367
    with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
        validate_span_length(start, pd.Timestamp("2024-12-29"))


def test_budget_exhaustion_still_sums_to_one() -> None:
    proposals = pd.Series([10.0, 10.0, 10.0], index=pd.date_range("2024-01-01", periods=3))
    weights = compute_weights_from_proposals(
        proposals=proposals,
        start_date=proposals.index.min(),
        end_date=proposals.index.max(),
        n_past=3,
    )
    assert np.isclose(float(weights.sum()), 1.0)
    assert (weights >= 0).all()


class _ProposeStrategy(BaseStrategy):
    def propose_weight(self, state) -> float:
        del state
        return np.inf


def test_nan_inf_rejection_in_propose_hook() -> None:
    with pytest.raises(ValueError, match="finite numeric value"):
        _ProposeStrategy().compute_weights(_context(days=3))


class _BothHooksStrategy(BaseStrategy):
    def propose_weight(self, state) -> float:
        return state.uniform_weight

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        values = pd.Series([1.0] + [0.0] * (len(features_df.index) - 1), index=features_df.index)
        return TargetProfile(values=values, mode="absolute")


def test_propose_hook_takes_precedence_over_batch_hook() -> None:
    weights = _BothHooksStrategy().compute_weights(_context(days=4))
    assert np.isclose(float(weights.iloc[0]), 0.25)
    assert np.isclose(float(weights.iloc[1]), 0.25)
    assert np.isclose(float(weights.iloc[2]), 0.25)
    assert np.isclose(float(weights.iloc[3]), 0.25)


class _ProfileOnlyStrategy(BaseStrategy):
    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del ctx, signals
        return pd.Series(np.linspace(0.0, 1.0, len(features_df.index)), index=features_df.index)


def test_build_target_profile_path_still_valid() -> None:
    weights = _ProfileOnlyStrategy().compute_weights(_context(days=5))
    assert np.isclose(float(weights.sum()), 1.0)
    assert (weights >= 0.0).all()


class _NoIntentStrategy(BaseStrategy):
    pass


def test_strategy_must_implement_at_least_one_intent_hook() -> None:
    with pytest.raises(TypeError, match="must implement propose_weight"):
        _NoIntentStrategy().compute_weights(_context(days=3))


class _IllegalComputeWeightsStrategy(BaseStrategy):
    def propose_weight(self, state) -> float:
        return state.uniform_weight

    def compute_weights(self, ctx: StrategyContext) -> pd.Series:  # type: ignore[override]
        del ctx
        return pd.Series(dtype=float)


def test_runner_rejects_compute_weights_override() -> None:
    runner = StrategyRunner()
    with pytest.raises(TypeError, match="Custom compute_weights overrides"):
        runner._validate_strategy_contract(_IllegalComputeWeightsStrategy())


def test_export_batch_rejects_compute_weights_override() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    features_df = pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(100.0, 102.0, len(idx)),
            "mvrv_zscore": np.linspace(-1.0, 1.0, len(idx)),
        },
        index=idx,
    )
    btc_df = pd.DataFrame(
        {"PriceUSD_coinmetrics": np.linspace(100.0, 102.0, len(idx))},
        index=idx,
    )
    with pytest.raises(TypeError, match="Custom compute_weights overrides"):
        process_start_date_batch(
            start_date=idx.min(),
            end_dates=[idx.max()],
            features_df=features_df,
            btc_df=btc_df,
            current_date=idx.max(),
            btc_price_col="PriceUSD_coinmetrics",
            strategy=_IllegalComputeWeightsStrategy(),
            enforce_span_contract=False,
        )
