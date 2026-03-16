"""Feature engineering and signal-scoring internals for model_development."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from .model_development_helpers import (
    compute_acceleration_modifier,
    compute_adaptive_trend_modifier,
    compute_asymmetric_extreme_boost,
    compute_mvrv_volatility,
    compute_percentile_signal,
    compute_signal_confidence,
    rolling_percentile,
    zscore,
)

DATE_COL = "date"


def precompute_features(
    df: pl.DataFrame,
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
) -> pl.DataFrame:
    """Compute lagged BTC/MVRV features used by model weight scoring."""
    if price_col not in df.columns:
        raise KeyError(f"'{price_col}' not found. Available: {list(df.columns)}")

    date_col = DATE_COL if DATE_COL in df.columns else None
    if date_col is None:
        raise KeyError("DataFrame must have 'date' column.")

    # Filter from 2010-07-18 for consistency with legacy
    cutoff = dt.datetime(2010, 7, 18)
    frame = df.filter(pl.col(date_col) >= cutoff).sort(date_col)
    if frame.is_empty():
        return pl.DataFrame(schema={date_col: pl.Datetime("us")})

    frame = frame.with_columns(pl.col(price_col).cast(pl.Float64, strict=False).alias(price_col))
    n = frame.height

    price_frame = frame.with_row_index("__row").with_columns(
        pl.when(pl.col("__row") >= (ma_window // 2))
        .then(pl.col(price_col).rolling_mean(window_size=ma_window, min_samples=1))
        .otherwise(float("nan"))
        .alias("price_ma")
    )
    price_frame = price_frame.with_columns(
        pl.when(
            pl.col("price_ma").is_not_null()
            & pl.col("price_ma").is_finite()
            & (pl.col("price_ma") != 0.0)
        )
        .then(((pl.col(price_col) / pl.col("price_ma")) - 1.0).clip(-1.0, 1.0))
        .otherwise(None)
        .fill_null(0.0)
        .fill_nan(0.0)
        .alias("price_vs_ma")
    )
    price_ma = price_frame["price_ma"]
    price_vs_ma = price_frame["price_vs_ma"]

    if mvrv_col in frame.columns:
        mvrv_s = frame[mvrv_col].cast(pl.Float64, strict=False).rename("mvrv")
        mvrv_z = (
            zscore(mvrv_s, mvrv_rolling_window)
            .clip(-4, 4)
            .replace([np.inf, -np.inf], 0.0)
            .fill_null(0.0)
            .rename("mvrv_zscore")
        )
        mvrv_z_arr = mvrv_z.to_numpy()

        mvrv_pct_s = rolling_percentile(mvrv_s, mvrv_cycle_window)
        mvrv_pct_s = (
            mvrv_pct_s
            .replace([np.inf, -np.inf], 0.5)
            .fill_null(0.5)
            .rename("mvrv_percentile")
        )
        mvrv_pct = mvrv_pct_s.to_numpy()
        mvrv_pct = np.where(np.isfinite(mvrv_pct), mvrv_pct, 0.5)

        gradient_raw = np.zeros(n)
        gradient_raw[mvrv_gradient_window:] = (
            mvrv_z_arr[mvrv_gradient_window:] - mvrv_z_arr[:-mvrv_gradient_window]
        )
        # EWM span
        alpha = 2 / (mvrv_gradient_window + 1)
        gradient_smooth = np.zeros(n)
        for i in range(1, n):
            gradient_smooth[i] = alpha * gradient_raw[i] + (1 - alpha) * gradient_smooth[i - 1]
        mvrv_gradient = np.tanh(gradient_smooth * 2)
        mvrv_gradient = np.where(np.isfinite(mvrv_gradient), mvrv_gradient, 0)

        accel_raw = np.zeros(n)
        accel_raw[mvrv_accel_window:] = (
            mvrv_gradient[mvrv_accel_window:] - mvrv_gradient[:-mvrv_accel_window]
        )
        accel_alpha = 2 / (mvrv_accel_window + 1)
        mvrv_acceleration = np.zeros(n)
        for i in range(1, n):
            mvrv_acceleration[i] = (
                accel_alpha * accel_raw[i] + (1 - accel_alpha) * mvrv_acceleration[i - 1]
            )
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3)
        mvrv_acceleration = np.where(np.isfinite(mvrv_acceleration), mvrv_acceleration, 0)

        mvrv_volatility = compute_mvrv_volatility(
            pl.Series("z", mvrv_z_arr), mvrv_volatility_window
        ).replace([np.inf, -np.inf], 0.5).fill_null(0.5).rename("mvrv_volatility")
        mvrv_volatility_arr = mvrv_volatility.to_numpy()
        mvrv_volatility_arr = np.where(np.isfinite(mvrv_volatility_arr), mvrv_volatility_arr, 0.5)

        signal_confidence = np.full(n, 0.5)
    else:
        mvrv_z = pl.Series("mvrv_zscore", np.zeros(n, dtype=float))
        mvrv_z_arr = mvrv_z.to_numpy()
        mvrv_pct_s = pl.Series("mvrv_percentile", np.full(n, 0.5, dtype=float))
        mvrv_pct = np.full(n, 0.5)
        mvrv_gradient = np.zeros(n)
        mvrv_acceleration = np.zeros(n)
        mvrv_volatility = pl.Series("mvrv_volatility", np.full(n, 0.5, dtype=float))
        mvrv_volatility_arr = mvrv_volatility.to_numpy()
        signal_confidence = np.full(n, 0.5)

    features = frame.select(date_col, price_col).with_columns(
        price_ma.alias("price_ma"),
        price_vs_ma.alias("price_vs_ma"),
        mvrv_z,
        pl.Series("mvrv_gradient", mvrv_gradient),
        mvrv_pct_s,
        pl.Series("mvrv_acceleration", mvrv_acceleration),
        mvrv_volatility.alias("mvrv_volatility"),
        pl.Series("signal_confidence", signal_confidence),
    )
    features = features.with_columns(
        pl.when(pl.col("mvrv_zscore") < mvrv_zone_deep_value)
        .then(pl.lit(-2))
        .when(pl.col("mvrv_zscore") < mvrv_zone_value)
        .then(pl.lit(-1))
        .when(pl.col("mvrv_zscore") < mvrv_zone_caution)
        .then(pl.lit(0))
        .when(pl.col("mvrv_zscore") < mvrv_zone_danger)
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("mvrv_zone"),
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
    features = features.with_columns([pl.col(col).shift(1).alias(col) for col in signal_cols])
    features = features.with_columns(
        pl.col("price_vs_ma").fill_null(0.0),
        pl.col("mvrv_zscore").fill_null(0.0),
        pl.col("mvrv_gradient").fill_null(0.0),
        pl.col("mvrv_percentile").fill_null(0.5),
        pl.col("mvrv_acceleration").fill_null(0.0),
        pl.col("mvrv_zone").fill_null(0),
        pl.col("mvrv_volatility").fill_null(0.5),
    )

    sc = compute_signal_confidence(
        features["mvrv_zscore"].to_numpy(),
        features["mvrv_percentile"].to_numpy(),
        features["mvrv_gradient"].to_numpy(),
        features["price_vs_ma"].to_numpy(),
    )
    features = features.with_columns(pl.Series("signal_confidence", sc))

    return features.select(
        date_col,
        price_col,
        "price_ma",
        "price_vs_ma",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_percentile",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
        "signal_confidence",
    )


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
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    *,
    compute_dynamic_multiplier_fn,
) -> pl.DataFrame:
    """Compute per-day preference logits from engineered features."""
    start_ts = (
        dt.datetime.strptime(start_date[:10], "%Y-%m-%d")
        if isinstance(start_date, str) else start_date
    )
    end_ts = (
        dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
        if isinstance(end_date, str) else end_date
    )
    date_col = DATE_COL if DATE_COL in features_df.columns else "date"
    df = features_df.filter(
        (pl.col(date_col) >= start_ts) & (pl.col(date_col) <= end_ts)
    )
    if df.is_empty():
        return pl.DataFrame({date_col: [], "preference": []})

    price_vs_ma = _clean_array(df["price_vs_ma"].to_numpy())
    mvrv_zscore = _clean_array(df["mvrv_zscore"].to_numpy())
    mvrv_gradient = _clean_array(df["mvrv_gradient"].to_numpy())

    mvrv_percentile = (
        _clean_array(df["mvrv_percentile"].to_numpy())
        if "mvrv_percentile" in df.columns
        else np.full(len(price_vs_ma), 0.5)
    )
    mvrv_percentile = np.where(mvrv_percentile == 0, 0.5, mvrv_percentile)

    mvrv_acceleration = (
        _clean_array(df["mvrv_acceleration"].to_numpy())
        if "mvrv_acceleration" in df.columns
        else None
    )
    mvrv_volatility = (
        _clean_array(df["mvrv_volatility"].to_numpy())
        if "mvrv_volatility" in df.columns
        else None
    )
    mvrv_volatility = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility) if mvrv_volatility is not None else None

    signal_confidence = (
        _clean_array(df["signal_confidence"].to_numpy())
        if "signal_confidence" in df.columns
        else None
    )
    signal_confidence = np.where(signal_confidence == 0, 0.5, signal_confidence) if signal_confidence is not None else None

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
    return pl.DataFrame({
        date_col: df[date_col].to_list(),
        "preference": preference.astype(float),
    })
