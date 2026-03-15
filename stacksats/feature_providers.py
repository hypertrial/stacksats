"""Framework-owned feature providers for causal strategy materialization."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from .feature_materialization import build_observed_frame
from .model_development import precompute_features

DATE_COL = "date"


class FeatureProvider(Protocol):
    """Protocol implemented by framework-owned feature providers."""

    provider_id: str

    def required_source_columns(self) -> tuple[str, ...]:
        """Return the source columns required by this provider."""

    def materialize(
        self,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        as_of_date: dt.datetime,
    ) -> pl.DataFrame:
        """Return a feature frame with date column."""


def _ensure_columns(
    btc_df: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    provider_id: str,
) -> None:
    missing = [c for c in columns if c not in btc_df.columns]
    if missing:
        raise ValueError(
            f"Feature provider '{provider_id}' is missing source columns: {missing}."
        )


def _rolling_zscore_pl(s: pl.Series, window: int) -> pl.Series:
    min_periods = max(30, window // 3)
    mean = s.rolling_mean(window_size=window, min_samples=min_periods)
    std = s.rolling_std(window_size=window, min_samples=min_periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (s - mean) / std
    return z.fill_null(0).replace([np.inf, -np.inf], 0.0)


def _btc_frame_cache_key(
    btc_df: pl.DataFrame,
    *,
    price_column: str = "price_usd",
) -> tuple[int, int, int, float, float]:
    if btc_df.is_empty():
        return (0, 0, 0, 0.0, 0.0)
    dates = btc_df[DATE_COL]
    if dates.len() == 0:
        return (0, 0, 0, 0.0, 0.0)
    prices = btc_df[price_column].to_numpy()
    first_price = float(prices[0]) if np.isfinite(prices[0]) else 0.0
    last_price = float(prices[-1]) if np.isfinite(prices[-1]) else 0.0
    d0 = dates[0]
    d1 = dates[-1]
    v0 = int(d0.timestamp() * 1e6) if hasattr(d0, "timestamp") else 0
    v1 = int(d1.timestamp() * 1e6) if hasattr(d1, "timestamp") else 0
    return (btc_df.height, v0, v1, first_price, last_price)


@dataclass(slots=True)
class CoreModelFeatureProvider:
    provider_id: str = "core_model_features_v1"
    _cache_key: tuple[int, int, int, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cache_features: pl.DataFrame | None = field(default=None, init=False, repr=False)

    def required_source_columns(self) -> tuple[str, ...]:
        return ("price_usd",)

    def materialize(
        self,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        as_of_date: dt.datetime,
    ) -> pl.DataFrame:
        del end_date
        _ensure_columns(
            btc_df,
            self.required_source_columns(),
            provider_id=self.provider_id,
        )
        cache_key = _btc_frame_cache_key(btc_df)
        if self._cache_key == cache_key and self._cache_features is not None:
            features = self._cache_features
        else:
            features = precompute_features(btc_df)
            self._cache_key = cache_key
            self._cache_features = features
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )


def _overlay_from_btc(btc_df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Extract overlay source columns from btc_df if present."""
    out_cols = [pl.col(DATE_COL)]
    for col in columns:
        if col in btc_df.columns:
            out_cols.append(pl.col(col).cast(pl.Float64, strict=False))
    return btc_df.select(out_cols) if len(out_cols) > 1 else btc_df.select(DATE_COL)


@dataclass(slots=True)
class BRKOverlayFeatureProvider:
    provider_id: str = "brk_overlay_v1"
    _cache_key: tuple[int, int, int, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cache_features: pl.DataFrame | None = field(default=None, init=False, repr=False)

    def required_source_columns(self) -> tuple[str, ...]:
        return ("price_usd", "mvrv")

    def materialize(
        self,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        as_of_date: dt.datetime,
    ) -> pl.DataFrame:
        del end_date
        _ensure_columns(
            btc_df,
            self.required_source_columns(),
            provider_id=self.provider_id,
        )
        cache_key = _btc_frame_cache_key(btc_df)
        if self._cache_key == cache_key and self._cache_features is not None:
            return build_observed_frame(
                self._cache_features,
                start_date=start_date,
                current_date=as_of_date,
            )

        price_arr = btc_df["price_usd"].cast(pl.Float64, strict=False).to_numpy()
        price_safe = np.where(price_arr > 0, price_arr, 1.0)
        price_log = np.log(price_safe)
        diff_30 = np.zeros_like(price_log)
        diff_30[30:] = price_log[30:] - price_log[:-30]
        diff_90 = np.zeros_like(price_log)
        diff_90[90:] = price_log[90:] - price_log[:-90]
        mom_30 = _rolling_zscore_pl(pl.Series("d30", diff_30), 365)
        mom_90 = _rolling_zscore_pl(pl.Series("d90", diff_90), 365)

        n = btc_df.height
        features = pl.DataFrame({
            DATE_COL: btc_df[DATE_COL].to_list(),
            "brk_flow": [0.0] * n,
            "brk_supply_pressure": [0.0] * n,
            "brk_activity_div": [0.0] * n,
            "brk_roi_context": [0.0] * n,
            "brk_liquidity_impulse": [0.0] * n,
            "brk_miner_pressure": [0.0] * n,
            "brk_hash_momentum": [0.0] * n,
            "brk_sentiment": [0.0] * n,
        })

        if "adjusted_sopr" in btc_df.columns and "adjusted_sopr_7d_ema" in btc_df.columns:
            sopr = btc_df["adjusted_sopr"].cast(pl.Float64, strict=False).fill_null(0)
            sopr_ema = btc_df["adjusted_sopr_7d_ema"].cast(pl.Float64, strict=False).fill_null(0)
            diff = sopr - sopr_ema
            features = features.with_columns(
                _rolling_zscore_pl(diff, 240).alias("brk_flow"),
                _rolling_zscore_pl((-0.65 * sopr) + (-0.35 * sopr_ema), 365).alias("brk_roi_context"),
            )
        else:
            features = features.with_columns(
                ((-0.65 * mom_30) + (-0.35 * mom_90)).alias("brk_roi_context"),
            )

        if "realized_cap_growth_rate" in btc_df.columns and "market_cap_growth_rate" in btc_df.columns:
            mkt = btc_df["market_cap_growth_rate"].cast(pl.Float64, strict=False).fill_null(0)
            real = btc_df["realized_cap_growth_rate"].cast(pl.Float64, strict=False).fill_null(0)
            features = features.with_columns(
                _rolling_zscore_pl(mkt - real, 365).alias("brk_supply_pressure"),
            )

        flow_fast = features["brk_flow"].rolling_mean(window_size=7, min_samples=3)
        flow_slow = features["brk_flow"].rolling_mean(window_size=30, min_samples=10)
        features = features.with_columns(
            _rolling_zscore_pl(flow_fast, 120).alias("brk_netflow_fast"),
            _rolling_zscore_pl(flow_slow, 240).alias("brk_netflow_slow"),
            _rolling_zscore_pl(flow_fast - flow_slow, 180).alias("brk_netflow_slope"),
            _rolling_zscore_pl(flow_fast, 180).alias("brk_netflow"),
        )
        activity_level = _rolling_zscore_pl(features["brk_activity_div"] + mom_30, 365)
        features = features.with_columns(
            activity_level.alias("brk_activity_level"),
            features["brk_activity_div"].alias("brk_activity_div_fast"),
            (activity_level - mom_90).alias("brk_activity_div_slow"),
            _rolling_zscore_pl(features["brk_liquidity_impulse"], 180).alias("brk_liquidity_level"),
            _rolling_zscore_pl(features["brk_supply_pressure"], 240).alias("brk_exchange_share_level"),
        )
        exchange_level = features["brk_exchange_share_level"]
        features = features.with_columns(
            _rolling_zscore_pl(exchange_level.diff(30), 240).alias("brk_exchange_share_delta"),
            _rolling_zscore_pl(exchange_level, 365).alias("brk_exchange_share"),
            _rolling_zscore_pl(mom_30, 365).alias("brk_roi30"),
            _rolling_zscore_pl(mom_90, 365).alias("brk_roi1y"),
        )

        features = features.shift(1)
        features = features.fill_null(0)
        float_cols = [
            c for c in features.columns
            if c != DATE_COL and features.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]
        if float_cols:
            features = features.with_columns([pl.col(c).clip(-6.0, 6.0) for c in float_cols])
        self._cache_key = cache_key
        self._cache_features = features
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )
