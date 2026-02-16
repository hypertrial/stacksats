"""Pure helper functions shared by model_development."""

from __future__ import annotations

import numpy as np
import pandas as pd


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank (0 to 1)."""

    def pct_rank(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.5
        rank = (x[-1] > x[:-1]).sum() / (len(x) - 1)
        return float(rank)

    return series.rolling(window, min_periods=window // 4).apply(pct_rank, raw=True)


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


def compute_mvrv_volatility(mvrv_zscore: pd.Series, window: int) -> pd.Series:
    """Compute rolling volatility of MVRV Z-score."""
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5,
        raw=False,
    )
    return vol_pct.fillna(0.5)


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
