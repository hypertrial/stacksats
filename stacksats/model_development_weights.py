"""Weight API internals for model_development."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
    *,
    compute_preference_scores_fn,
    allocate_sequential_stable_fn,
) -> pd.Series:
    """Compute weights for a date window using precomputed features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    preference = compute_preference_scores_fn(features_df, start_date, end_date)
    raw = (np.ones(n, dtype=float) / n) * np.exp(np.clip(preference.to_numpy(), -50, 50))

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable_fn(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
    *,
    validate_span_length_fn,
    compute_preference_scores_fn,
    compute_weights_from_target_profile_fn,
    assert_final_invariants_fn,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability."""
    validate_span_length_fn(start_date, end_date)
    preference = compute_preference_scores_fn(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
    )
    weights = compute_weights_from_target_profile_fn(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        current_date=current_date,
        target_profile=preference,
        mode="preference",
        locked_weights=locked_weights,
    )
    assert_final_invariants_fn(weights.to_numpy(dtype=float))
    return weights
