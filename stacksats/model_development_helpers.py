"""Pure helper functions shared by model_development."""

from __future__ import annotations

import numpy as np
import polars as pl


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


def zscore(series: pl.Series, window: int) -> pl.Series:
    """Compute rolling z-score."""
    mean = series.rolling_mean(window_size=window, min_samples=window // 2)
    std = series.rolling_std(window_size=window, min_samples=window // 2)
    out = (series - mean) / std
    return out.fill_null(0)


def zscore_expr(expr: pl.Expr, window: int) -> pl.Expr:
    """Compute rolling z-score as a Polars expression."""
    mean = expr.rolling_mean(window_size=window, min_samples=window // 2)
    std = expr.rolling_std(window_size=window, min_samples=window // 2)
    return (expr - mean) / std


def rolling_percentile(series: pl.Series, window: int) -> pl.Series:
    """Compute rolling percentile rank (0 to 1)."""
    arr = series.to_numpy()
    min_periods = max(1, window // 4)
    out = _rolling_last_rank(
        arr,
        window=window,
        min_periods=min_periods,
        default=0.5,
    )
    return pl.Series("pct", out)


def rolling_percentile_expr(expr: pl.Expr, window: int) -> pl.Expr:  # pragma: no cover
    """Legacy placeholder retained for compatibility with older imports."""
    del expr, window
    raise NotImplementedError(
        "rolling_percentile_expr is no longer supported in hot paths; "
        "materialize the series and call rolling_percentile(...) instead."
    )


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


def compute_mvrv_volatility(mvrv_zscore: pl.Series, window: int) -> pl.Series:
    """Compute rolling volatility of MVRV Z-score."""
    vol = mvrv_zscore.rolling_std(window_size=window, min_samples=window // 4)
    arr = vol.to_numpy()
    rank_window = max(1, window * 4)
    out = _rolling_last_rank(
        arr,
        window=rank_window,
        min_periods=max(1, window),
        default=0.5,
    )
    return pl.Series("vol", np.where(np.isnan(out), 0.5, out))


def compute_mvrv_volatility_expr(expr: pl.Expr, window: int) -> pl.Expr:  # pragma: no cover
    """Legacy placeholder retained for compatibility with older imports."""
    del expr, window
    raise NotImplementedError(
        "compute_mvrv_volatility_expr is no longer supported in hot paths; "
        "materialize the series and call compute_mvrv_volatility(...) instead."
    )


def _rolling_last_rank(
    arr: np.ndarray,
    *,
    window: int,
    min_periods: int,
    default: float,
) -> np.ndarray:
    """Return rank of the latest value within each trailing window."""
    values = np.asarray(arr, dtype=float)
    n = values.size
    if n == 0:  # pragma: no cover
        return np.zeros(0, dtype=float)
    if window <= 1:
        return np.full(n, default, dtype=float)  # pragma: no cover

    padded = np.full(n + window - 1, np.nan, dtype=float)
    padded[window - 1 :] = values
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    current = windows[:, -1]
    prior = windows[:, :-1]
    valid_prior = np.isfinite(prior)
    denom = valid_prior.sum(axis=1)
    wins = ((current[:, None] > prior) & valid_prior).sum(axis=1)
    out = np.full(n, float(default), dtype=float)
    eligible = (np.arange(n) >= (max(min_periods, 1) - 1)) & (denom > 0) & np.isfinite(current)
    out[eligible] = wins[eligible] / denom[eligible]
    return out


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


def compute_signal_confidence_expr(
    mvrv_zscore: pl.Expr,
    mvrv_percentile: pl.Expr,
    mvrv_gradient: pl.Expr,
    price_vs_ma: pl.Expr,
) -> pl.Expr:
    """Compute confidence score based on signal agreement as a Polars expression."""
    z_signal = -mvrv_zscore / 4.0
    pct_signal = (0.5 - mvrv_percentile) * 2.0
    ma_signal = -price_vs_ma
    gradient_alignment = (
        pl.when(z_signal < 0.0)
        .then(pl.when(mvrv_gradient > 0.0).then(1.0).otherwise(0.5))
        .otherwise(pl.when(mvrv_gradient < 0.0).then(1.0).otherwise(0.5))
    )
    signal_mean = (z_signal + pct_signal + ma_signal) / 3.0
    variance = (
        ((z_signal - signal_mean) ** 2)
        + ((pct_signal - signal_mean) ** 2)
        + ((ma_signal - signal_mean) ** 2)
    ) / 3.0
    agreement = 1.0 - variance.sqrt().clip(0.0, 1.0)
    return (agreement * 0.7 + gradient_alignment * 0.3).clip(0.0, 1.0)


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
