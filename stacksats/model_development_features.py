"""Feature engineering and signal-scoring internals for model_development."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def precompute_features(
    df: pd.DataFrame,
    *,
    price_col: str,
    mvrv_col: str,
    ma_window: int,
    mvrv_rolling_window: int,
    mvrv_cycle_window: int,
    mvrv_gradient_window: int,
    mvrv_accel_window: int,
    mvrv_zone_deep_value: float,
    mvrv_zone_value: float,
    mvrv_zone_caution: float,
    mvrv_zone_danger: float,
    mvrv_volatility_window: int,
) -> pd.DataFrame:
    """Compute lagged BTC/MVRV features used by model weight scoring."""
    if price_col not in df.columns:
        raise KeyError(f"'{price_col}' not found. Available: {list(df.columns)}")

    price = df[price_col].loc["2010-07-18":].copy()

    ma = price.rolling(ma_window, min_periods=ma_window // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    if mvrv_col in df.columns:
        mvrv = df[mvrv_col].loc[price.index]
        mvrv_z = zscore(mvrv, mvrv_rolling_window).clip(-4, 4)
        mvrv_pct = rolling_percentile(mvrv, mvrv_cycle_window).fillna(0.5)

        gradient_raw = mvrv_z.diff(mvrv_gradient_window)
        gradient_smooth = gradient_raw.ewm(span=mvrv_gradient_window, adjust=False).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)

        accel_raw = mvrv_gradient.diff(mvrv_accel_window)
        mvrv_acceleration = accel_raw.ewm(span=mvrv_accel_window, adjust=False).mean()
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3).fillna(0)

        mvrv_zone = pd.Series(
            classify_mvrv_zone(
                mvrv_z.values,
                zone_deep_value=mvrv_zone_deep_value,
                zone_value=mvrv_zone_value,
                zone_caution=mvrv_zone_caution,
                zone_danger=mvrv_zone_danger,
            ),
            index=mvrv_z.index,
        )

        mvrv_volatility = compute_mvrv_volatility(mvrv_z, mvrv_volatility_window)
        signal_confidence = pd.Series(0.5, index=price.index)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_pct = pd.Series(0.5, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        signal_confidence = pd.Series(0.5, index=price.index)

    features = pd.DataFrame(
        {
            price_col: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma,
            "mvrv_zscore": mvrv_z,
            "mvrv_gradient": mvrv_gradient,
            "mvrv_percentile": mvrv_pct,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_zone": mvrv_zone,
            "mvrv_volatility": mvrv_volatility,
            "signal_confidence": signal_confidence,
        },
        index=price.index,
    )

    signal_cols = [
        "price_vs_ma",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_percentile",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    features["mvrv_percentile"] = features["mvrv_percentile"].fillna(0.5)
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features = features.fillna(0)

    features["signal_confidence"] = compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_percentile"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    return features


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_percentile: np.ndarray | None = None,
    mvrv_acceleration: np.ndarray | None = None,
    mvrv_volatility: np.ndarray | None = None,
    signal_confidence: np.ndarray | None = None,
    *,
    mvrv_zone_deep_value: float,
    mvrv_zone_value: float,
    mvrv_zone_caution: float,
    mvrv_zone_danger: float,
    mvrv_volatility_dampening: float,
    dynamic_strength: float,
) -> np.ndarray:
    """Compute model multipliers from MVRV and MA signals."""
    if mvrv_percentile is None:
        mvrv_percentile = np.full_like(mvrv_zscore, 0.5)
    if mvrv_acceleration is None:
        mvrv_acceleration = np.zeros_like(mvrv_zscore)
    if mvrv_volatility is None:
        mvrv_volatility = np.full_like(mvrv_zscore, 0.5)
    if signal_confidence is None:
        signal_confidence = np.full_like(mvrv_zscore, 0.5)

    value_signal = -mvrv_zscore
    extreme_boost = compute_asymmetric_extreme_boost(
        mvrv_zscore,
        zone_deep_value=mvrv_zone_deep_value,
        zone_value=mvrv_zone_value,
        zone_caution=mvrv_zone_caution,
        zone_danger=mvrv_zone_danger,
    )
    value_signal = value_signal + extreme_boost

    ma_signal = -price_vs_ma
    trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    ma_signal = ma_signal * trend_modifier

    pct_signal = compute_percentile_signal(mvrv_percentile)
    accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)

    combined = value_signal * 0.70 + ma_signal * 0.20 + pct_signal * 0.10

    accel_modifier_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
    accel_modifier_subtle = np.clip(accel_modifier_subtle, 0.85, 1.15)
    combined = combined * accel_modifier_subtle

    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,
        1.0,
    )
    combined = combined * confidence_boost

    volatility_dampening = np.where(
        mvrv_volatility > 0.8,
        1.0 - mvrv_volatility_dampening * (mvrv_volatility - 0.8) / 0.2,
        1.0,
    )
    combined = combined * volatility_dampening

    adjustment = np.clip(combined * dynamic_strength, -5, 100)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


def _clean_array(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, 0)


def compute_preference_scores(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    compute_dynamic_multiplier_fn,
) -> pd.Series:
    """Compute per-day preference logits from engineered features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    mvrv_zscore = _clean_array(df["mvrv_zscore"].values)
    mvrv_gradient = _clean_array(df["mvrv_gradient"].values)

    if "mvrv_percentile" in df.columns:
        mvrv_percentile = _clean_array(df["mvrv_percentile"].values)
        mvrv_percentile = np.where(mvrv_percentile == 0, 0.5, mvrv_percentile)
    else:
        mvrv_percentile = None

    if "mvrv_acceleration" in df.columns:
        mvrv_acceleration = _clean_array(df["mvrv_acceleration"].values)
    else:
        mvrv_acceleration = None

    if "mvrv_volatility" in df.columns:
        mvrv_volatility = _clean_array(df["mvrv_volatility"].values)
        mvrv_volatility = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility)
    else:
        mvrv_volatility = None

    if "signal_confidence" in df.columns:
        signal_confidence = _clean_array(df["signal_confidence"].values)
        signal_confidence = np.where(signal_confidence == 0, 0.5, signal_confidence)
    else:
        signal_confidence = None

    multiplier = compute_dynamic_multiplier_fn(
        price_vs_ma,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_percentile,
        mvrv_acceleration,
        mvrv_volatility,
        signal_confidence,
    )
    safe_multiplier = np.clip(multiplier, 1e-12, None)
    preference = np.log(safe_multiplier)
    return pd.Series(preference, index=df.index, dtype=float)
