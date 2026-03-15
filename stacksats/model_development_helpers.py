"""Pure helper functions shared by model_development."""

from __future__ import annotations

import numpy as np
import polars as pl

try:
    import pandas as pd
except ImportError:
    pd = None


def _to_pl_series(s) -> pl.Series:
    """Convert to Polars Series for processing."""
    if isinstance(s, pl.Series):
        return s
    if pd is not None and isinstance(s, pd.Series):
        return pl.from_pandas(s)
    raise TypeError("Expected pl.Series or pd.Series")


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


def zscore(series: pl.Series | "pd.Series", window: int) -> pl.Series | "pd.Series":
    """Compute rolling z-score. Returns same type as input."""
    s = _to_pl_series(series)
    mean = s.rolling_mean(window_size=window, min_samples=window // 2)
    std = s.rolling_std(window_size=window, min_samples=window // 2)
    out = (s - mean) / std
    result = out.fill_null(0)
    if pd is not None and isinstance(series, pd.Series):
        out_pd = result.to_pandas()
        out_pd.index = series.index
        return out_pd
    return result


def rolling_percentile(series: pl.Series | "pd.Series", window: int) -> pl.Series | "pd.Series":
    """Compute rolling percentile rank (0 to 1). Returns same type as input."""

    def pct_rank(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.5
        rank = (x[-1] > x[:-1]).sum() / (len(x) - 1)
        return float(rank)

    s = _to_pl_series(series)
    arr = s.to_numpy()
    out = np.full(len(arr), 0.5)
    min_periods = window // 4
    for i in range(min_periods - 1, len(arr)):
        start = max(0, i - window + 1)
        window_slice = arr[start : i + 1]
        out[i] = pct_rank(window_slice)
    result = pl.Series("pct", out)
    if pd is not None and isinstance(series, pd.Series):
        out_pd = result.to_pandas()
        out_pd.index = series.index
        return out_pd
    return result


def classify_mvrv_zone(
    mvrv_zscore: np.ndarray,
    *,
    zone_deep_value: float = -2.0,
    zone_value: float = -1.0,
    zone_caution: float = 1.5,
    zone_danger: float = 2.5,
) -> np.ndarray:
    """Classify MVRV into discrete zones for regime detection."""
    return np.select(
        [
            mvrv_zscore < zone_deep_value,
            mvrv_zscore < zone_value,
            mvrv_zscore < zone_caution,
            mvrv_zscore < zone_danger,
        ],
        [-2, -1, 0, 1],
        default=2,
    )


def compute_mvrv_volatility(mvrv_zscore: pl.Series | "pd.Series", window: int) -> pl.Series | "pd.Series":
    """Compute rolling volatility of MVRV Z-score. Returns same type as input."""
    s = _to_pl_series(mvrv_zscore)
    vol = s.rolling_std(window_size=window, min_samples=window // 4)
    arr = vol.to_numpy()
    rank_window = window * 4
    out = np.full(len(arr), 0.5)
    for i in range(window - 1, len(arr)):
        start = max(0, i - rank_window + 1)
        window_slice = arr[start : i + 1]
        if len(window_slice) > 1:
            out[i] = (window_slice[-1] > window_slice[:-1]).sum() / (len(window_slice) - 1)
    result = pl.Series("vol", np.where(np.isnan(out), 0.5, out))
    if pd is not None and isinstance(mvrv_zscore, pd.Series):
        out_pd = result.to_pandas()
        out_pd.index = mvrv_zscore.index
        return out_pd
    return result


def compute_signal_confidence(
    mvrv_zscore: np.ndarray,
    mvrv_percentile: np.ndarray,
    mvrv_gradient: np.ndarray,
    price_vs_ma: np.ndarray,
) -> np.ndarray:
    """Compute confidence score based on signal agreement."""
    z_signal = -mvrv_zscore / 4
    pct_signal = (0.5 - mvrv_percentile) * 2
    ma_signal = -price_vs_ma

    gradient_alignment = np.where(
        z_signal < 0,
        np.where(mvrv_gradient > 0, 1.0, 0.5),
        np.where(mvrv_gradient < 0, 1.0, 0.5),
    )

    signals = np.stack([z_signal, pct_signal, ma_signal], axis=0)
    signal_std = signals.std(axis=0)
    agreement = 1.0 - np.clip(signal_std / 1.0, 0, 1)
    confidence = agreement * 0.7 + gradient_alignment * 0.3
    return np.clip(confidence, 0, 1)


def compute_asymmetric_extreme_boost(
    mvrv_zscore: np.ndarray,
    *,
    zone_deep_value: float = -2.0,
    zone_value: float = -1.0,
    zone_caution: float = 1.5,
    zone_danger: float = 2.5,
) -> np.ndarray:
    """Compute asymmetric boost for extreme MVRV values."""
    boost = np.zeros_like(mvrv_zscore)

    deep_value_mask = mvrv_zscore < zone_deep_value
    boost = np.where(
        deep_value_mask,
        0.8 * (mvrv_zscore - zone_deep_value) ** 2 + 0.5,
        boost,
    )

    value_mask = (mvrv_zscore >= zone_deep_value) & (mvrv_zscore < zone_value)
    boost = np.where(value_mask, -0.5 * mvrv_zscore, boost)

    caution_mask = (mvrv_zscore >= zone_caution) & (mvrv_zscore < zone_danger)
    boost = np.where(caution_mask, -0.3 * (mvrv_zscore - zone_caution), boost)

    danger_mask = mvrv_zscore >= zone_danger
    boost = np.where(danger_mask, -0.5 * (mvrv_zscore - zone_danger) ** 2 - 0.3, boost)

    return boost


def compute_percentile_signal(mvrv_percentile: np.ndarray) -> np.ndarray:
    """Convert MVRV percentile to buy/sell signal."""
    centered = 0.5 - mvrv_percentile
    signal = np.sign(centered) * (2 * np.abs(centered)) ** 1.5
    return np.clip(signal, -1, 1)


def compute_acceleration_modifier(
    mvrv_acceleration: np.ndarray,
    mvrv_gradient: np.ndarray,
) -> np.ndarray:
    """Compute modifier based on MVRV acceleration."""
    same_direction = (mvrv_acceleration * mvrv_gradient) > 0
    modifier = np.where(
        same_direction,
        1.0 + 0.3 * np.abs(mvrv_acceleration),
        1.0 - 0.2 * np.abs(mvrv_acceleration),
    )
    return np.clip(modifier, 0.5, 1.5)


def compute_adaptive_trend_modifier(
    mvrv_gradient: np.ndarray,
    mvrv_zscore: np.ndarray,
) -> np.ndarray:
    """Compute trend modifier with adaptive thresholds."""
    threshold = np.where(
        mvrv_zscore < -1,
        0.1,
        np.where(mvrv_zscore > 1.5, 0.4, 0.2),
    )
    modifier = np.where(
        mvrv_gradient > threshold,
        1.0 + 0.5 * np.minimum(mvrv_gradient, 1.0),
        np.where(
            mvrv_gradient < -threshold,
            0.3 + 0.2 * (1 + mvrv_gradient),
            1.0,
        ),
    )
    return np.clip(modifier, 0.3, 1.5)
