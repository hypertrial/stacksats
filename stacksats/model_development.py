"""Dynamic DCA feature/weight public facade.

This module keeps the historical public API while delegating implementation to
smaller internal modules:
- model_development_features.py
- model_development_allocation.py
- model_development_weights.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from .framework_contract import (
    MIN_DAILY_WEIGHT,
    assert_final_invariants,
    validate_span_length,
)
from .model_development_allocation import (
    _compute_stable_signal,
    allocate_sequential_stable,
    allocate_from_proposals,
    compute_weights_from_proposals as _compute_weights_from_proposals_impl,
    compute_weights_from_target_profile as _compute_weights_from_target_profile_impl,
)
from .model_development_features import (
    _clean_array,
    compute_dynamic_multiplier as _compute_dynamic_multiplier_impl,
    compute_preference_scores as _compute_preference_scores_impl,
    precompute_features as _precompute_features_impl,
)
from .model_development_helpers import (
    classify_mvrv_zone,
    compute_acceleration_modifier,
    compute_adaptive_trend_modifier,
    compute_asymmetric_extreme_boost,
    compute_mvrv_volatility,
    compute_percentile_signal,
    compute_signal_confidence,
    rolling_percentile,
    zscore,
)
from .model_development_weights import (
    compute_weights_fast as _compute_weights_fast_impl,
    compute_window_weights as _compute_window_weights_impl,
)

PRICE_COL = "price_usd"
MVRV_COL = "mvrv"

MIN_W = MIN_DAILY_WEIGHT
MA_WINDOW = 200
MVRV_GRADIENT_WINDOW = 30
MVRV_ROLLING_WINDOW = 365
MVRV_CYCLE_WINDOW = 1461
MVRV_ACCEL_WINDOW = 14
DYNAMIC_STRENGTH = 5.0

MVRV_ZONE_DEEP_VALUE = -2.0
MVRV_ZONE_VALUE = -1.0
MVRV_ZONE_CAUTION = 1.5
MVRV_ZONE_DANGER = 2.5

MVRV_VOLATILITY_WINDOW = 90
MVRV_VOLATILITY_DAMPENING = 0.2

FEATS = [
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_percentile",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
    "signal_confidence",
]


def precompute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute MVRV/MA features used by weight-generation paths."""
    return _precompute_features_impl(
        df,
        price_col=PRICE_COL,
        mvrv_col=MVRV_COL,
        ma_window=MA_WINDOW,
        mvrv_rolling_window=MVRV_ROLLING_WINDOW,
        mvrv_cycle_window=MVRV_CYCLE_WINDOW,
        mvrv_gradient_window=MVRV_GRADIENT_WINDOW,
        mvrv_accel_window=MVRV_ACCEL_WINDOW,
        mvrv_zone_deep_value=MVRV_ZONE_DEEP_VALUE,
        mvrv_zone_value=MVRV_ZONE_VALUE,
        mvrv_zone_caution=MVRV_ZONE_CAUTION,
        mvrv_zone_danger=MVRV_ZONE_DANGER,
        mvrv_volatility_window=MVRV_VOLATILITY_WINDOW,
    )


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_percentile: np.ndarray | None = None,
    mvrv_acceleration: np.ndarray | None = None,
    mvrv_volatility: np.ndarray | None = None,
    signal_confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weight multiplier from MVRV and MA signals."""
    return _compute_dynamic_multiplier_impl(
        price_vs_ma,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_percentile,
        mvrv_acceleration,
        mvrv_volatility,
        signal_confidence,
        mvrv_zone_deep_value=MVRV_ZONE_DEEP_VALUE,
        mvrv_zone_value=MVRV_ZONE_VALUE,
        mvrv_zone_caution=MVRV_ZONE_CAUTION,
        mvrv_zone_danger=MVRV_ZONE_DANGER,
        mvrv_volatility_dampening=MVRV_VOLATILITY_DAMPENING,
        dynamic_strength=DYNAMIC_STRENGTH,
    )


def compute_preference_scores(
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
) -> pl.DataFrame:
    """Compute daily preference scores from model features."""
    return _compute_preference_scores_impl(
        features_df,
        start_date,
        end_date,
        compute_dynamic_multiplier_fn=compute_dynamic_multiplier,
    )


def compute_weights_from_target_profile(
    *,
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    current_date: dt.datetime | str,
    target_profile: pl.DataFrame,
    mode: str = "preference",
    locked_weights: np.ndarray | None = None,
    n_past: int | None = None,
) -> pl.DataFrame:
    """Convert a target profile into final iterative stable allocation weights."""
    del features_df
    return _compute_weights_from_target_profile_impl(
        start_date=start_date,
        end_date=end_date,
        current_date=current_date,
        target_profile=target_profile,
        mode=mode,
        locked_weights=locked_weights,
        n_past=n_past,
    )


def compute_weights_from_proposals(
    *,
    proposals: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> pl.DataFrame:
    """Convert per-day user proposals into final framework weights."""
    return _compute_weights_from_proposals_impl(
        proposals=proposals,
        start_date=start_date,
        end_date=end_date,
        n_past=n_past,
        locked_weights=locked_weights,
    )


def compute_weights_fast(
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pl.DataFrame:
    """Compute weights for a date window using precomputed features."""
    return _compute_weights_fast_impl(
        features_df,
        start_date,
        end_date,
        n_past,
        locked_weights,
        compute_preference_scores_fn=compute_preference_scores,
        allocate_sequential_stable_fn=allocate_sequential_stable,
    )


def compute_window_weights(
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    current_date: dt.datetime | str,
    locked_weights: np.ndarray | None = None,
) -> pl.DataFrame:
    """Compute weights for a date range with lock-on-compute stability."""
    return _compute_window_weights_impl(
        features_df,
        start_date,
        end_date,
        current_date,
        locked_weights,
        validate_span_length_fn=validate_span_length,
        compute_preference_scores_fn=compute_preference_scores,
        compute_weights_from_target_profile_fn=compute_weights_from_target_profile,
        assert_final_invariants_fn=assert_final_invariants,
    )


__all__ = [
    "PRICE_COL",
    "MVRV_COL",
    "MIN_W",
    "MA_WINDOW",
    "MVRV_GRADIENT_WINDOW",
    "MVRV_ROLLING_WINDOW",
    "MVRV_CYCLE_WINDOW",
    "MVRV_ACCEL_WINDOW",
    "DYNAMIC_STRENGTH",
    "MVRV_ZONE_DEEP_VALUE",
    "MVRV_ZONE_VALUE",
    "MVRV_ZONE_CAUTION",
    "MVRV_ZONE_DANGER",
    "MVRV_VOLATILITY_WINDOW",
    "MVRV_VOLATILITY_DAMPENING",
    "FEATS",
    "_compute_stable_signal",
    "_clean_array",
    "allocate_sequential_stable",
    "allocate_from_proposals",
    "classify_mvrv_zone",
    "compute_acceleration_modifier",
    "compute_adaptive_trend_modifier",
    "compute_asymmetric_extreme_boost",
    "compute_mvrv_volatility",
    "compute_percentile_signal",
    "compute_signal_confidence",
    "rolling_percentile",
    "zscore",
    "precompute_features",
    "compute_dynamic_multiplier",
    "compute_preference_scores",
    "compute_weights_from_target_profile",
    "compute_weights_from_proposals",
    "compute_weights_fast",
    "compute_window_weights",
]
