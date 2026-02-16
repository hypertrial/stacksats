"""Shared helper functions used by StrategyRunner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .framework_contract import ALLOCATION_SPAN_DAYS
from .strategy_types import TargetProfile


def weights_match(lhs: pd.Series, rhs: pd.Series, *, atol: float = 1e-12) -> bool:
    if not lhs.index.equals(rhs.index):
        return False
    left = pd.to_numeric(lhs, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(rhs, errors="coerce").to_numpy(dtype=float)
    return bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right))) and bool(
        np.allclose(left, right, rtol=0.0, atol=atol)
    )


def profile_values(profile: TargetProfile | pd.Series) -> pd.Series:
    values = profile.values if isinstance(profile, TargetProfile) else profile
    return pd.to_numeric(values, errors="coerce")


def frame_signature(df: pd.DataFrame) -> tuple:
    try:
        row_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
    except TypeError:
        row_hash = hash(df.to_json(date_format="iso", orient="split", default_handler=str))
    return (
        row_hash,
        tuple(str(col) for col in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        tuple(df.shape),
    )


def perturb_future_features(features_df: pd.DataFrame, probe: pd.Timestamp) -> pd.DataFrame:
    perturbed = features_df.copy(deep=True)
    future_mask = perturbed.index > probe
    if not bool(future_mask.any()):
        return perturbed

    future = perturbed.loc[future_mask].copy()
    numeric_cols = list(future.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        numeric = future[numeric_cols].to_numpy(dtype=float)
        ramp = np.linspace(1.0, 2.0, numeric.shape[0], dtype=float).reshape(-1, 1)
        shifted = np.where(np.isfinite(numeric), (-3.0 * numeric) + ramp, 0.0)
        future.loc[:, numeric_cols] = shifted

    non_numeric_cols = [col for col in future.columns if col not in numeric_cols]
    if len(non_numeric_cols) > 0 and len(future.index) > 1:
        future.loc[:, non_numeric_cols] = future[non_numeric_cols].iloc[::-1].to_numpy()

    perturbed.loc[future_mask, :] = future
    return perturbed


def build_fold_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    all_days = pd.date_range(start=start_ts, end=end_ts, freq="D")
    if len(all_days) < (ALLOCATION_SPAN_DAYS * 2):
        return []
    max_folds = min(4, len(all_days) // ALLOCATION_SPAN_DAYS)
    if max_folds < 2:
        return []
    boundaries = np.linspace(0, len(all_days), num=max_folds + 1, dtype=int)
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(max_folds):
        left = int(boundaries[i])
        right = int(boundaries[i + 1]) - 1
        if right <= left:
            continue
        fold_start = all_days[left]
        fold_end = all_days[right]
        if len(pd.date_range(fold_start, fold_end, freq="D")) >= ALLOCATION_SPAN_DAYS:
            folds.append((fold_start, fold_end))
    return folds
