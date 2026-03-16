"""Framework-owned feature providers for causal strategy materialization."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from .feature_materialization import build_observed_frame, hash_dataframe
from .model_development import precompute_features

DATE_COL = "date"
BRK_OPTIONAL_SOURCE_COLUMNS = (
    "adjusted_sopr",
    "adjusted_sopr_7d_ema",
    "realized_cap_growth_rate",
    "market_cap_growth_rate",
)


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

    def materialize_lazy(
        self,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        as_of_date: dt.datetime,
    ) -> pl.LazyFrame:
        """Return a lazy feature frame with date column."""


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


def _rolling_zscore_expr(expr: pl.Expr, window: int) -> pl.Expr:
    min_periods = max(30, window // 3)
    mean = expr.rolling_mean(window_size=window, min_samples=min_periods)
    std = expr.rolling_std(window_size=window, min_samples=min_periods)
    raw = (expr - mean) / std
    return raw.fill_null(0.0).replace([np.inf, -np.inf], 0.0)


def _btc_frame_cache_key(
    btc_df: pl.DataFrame,
    *,
    source_columns: tuple[str, ...],
) -> tuple[int, str]:
    if btc_df.is_empty():
        return (0, "")
    selected_columns = [DATE_COL, *[c for c in source_columns if c in btc_df.columns]]
    return (btc_df.height, hash_dataframe(btc_df.select(selected_columns)))


@dataclass(slots=True)
class CoreModelFeatureProvider:
    provider_id: str = "core_model_features_v1"
    _cache_key: tuple[int, str] | None = field(
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
        if btc_df.is_empty():
            return pl.DataFrame(schema={DATE_COL: pl.Datetime("us")})
        _ensure_columns(
            btc_df,
            self.required_source_columns(),
            provider_id=self.provider_id,
        )
        cache_key = _btc_frame_cache_key(
            btc_df,
            source_columns=self.required_source_columns(),
        )
        if self._cache_key == cache_key and self._cache_features is not None:
            return build_observed_frame(
                self._cache_features,
                start_date=start_date,
                current_date=as_of_date,
            )
        features = precompute_features(btc_df)
        self._cache_key = cache_key
        self._cache_features = features
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )

    def materialize_lazy(
        self,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        as_of_date: dt.datetime,
    ) -> pl.LazyFrame:
        if btc_df.is_empty():
            return pl.DataFrame(schema={DATE_COL: pl.Datetime("us")}).lazy()
        return self.materialize(
            btc_df,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        ).lazy()


@dataclass(slots=True)
class BRKOverlayFeatureProvider:
    provider_id: str = "brk_overlay_v1"
    _cache_key: tuple[int, str] | None = field(
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
        if btc_df.is_empty():
            return pl.DataFrame(schema={DATE_COL: pl.Datetime("us")})
        _ensure_columns(
            btc_df,
            self.required_source_columns(),
            provider_id=self.provider_id,
        )
        cache_key = _btc_frame_cache_key(
            btc_df,
            source_columns=self.required_source_columns() + BRK_OPTIONAL_SOURCE_COLUMNS,
        )
        if self._cache_key == cache_key and self._cache_features is not None:
            return build_observed_frame(
                self._cache_features,
                start_date=start_date,
                current_date=as_of_date,
            )
        features = self.materialize_lazy(
            btc_df,
            start_date=btc_df[DATE_COL][0],
            end_date=btc_df[DATE_COL][-1],
            as_of_date=btc_df[DATE_COL][-1],
        ).collect()
        self._cache_key = cache_key
        self._cache_features = features
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )

    def materialize_lazy(
        self,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        as_of_date: dt.datetime,
    ) -> pl.LazyFrame:
        del start_date, end_date, as_of_date
        if btc_df.is_empty():
            return pl.DataFrame(schema={DATE_COL: pl.Datetime("us")}).lazy()
        base = btc_df.lazy().select(
            pl.col(DATE_COL),
            pl.col("price_usd").cast(pl.Float64, strict=False).alias("price_usd"),
            *[
                pl.col(col).cast(pl.Float64, strict=False).alias(col)
                for col in BRK_OPTIONAL_SOURCE_COLUMNS
                if col in btc_df.columns
            ],
        ).with_columns(
            pl.when(pl.col("price_usd") > 0.0)
            .then(pl.col("price_usd"))
            .otherwise(1.0)
            .log()
            .alias("_price_log")
        )
        base = base.with_columns(
            (pl.col("_price_log") - pl.col("_price_log").shift(30)).fill_null(0.0).alias("_diff_30"),
            (pl.col("_price_log") - pl.col("_price_log").shift(90)).fill_null(0.0).alias("_diff_90"),
        )
        base = base.with_columns(
            _rolling_zscore_expr(pl.col("_diff_30"), 365).alias("_mom_30"),
            _rolling_zscore_expr(pl.col("_diff_90"), 365).alias("_mom_90"),
        ).with_columns(
            pl.lit(0.0).alias("brk_flow"),
            pl.lit(0.0).alias("brk_supply_pressure"),
            pl.lit(0.0).alias("brk_activity_div"),
            pl.lit(0.0).alias("brk_roi_context"),
            pl.lit(0.0).alias("brk_liquidity_impulse"),
            pl.lit(0.0).alias("brk_miner_pressure"),
            pl.lit(0.0).alias("brk_hash_momentum"),
            pl.lit(0.0).alias("brk_sentiment"),
        )

        if "adjusted_sopr" in btc_df.columns and "adjusted_sopr_7d_ema" in btc_df.columns:
            sopr = pl.col("adjusted_sopr").fill_null(0.0)
            sopr_ema = pl.col("adjusted_sopr_7d_ema").fill_null(0.0)
            base = base.with_columns(
                _rolling_zscore_expr(sopr - sopr_ema, 240).alias("brk_flow"),
                _rolling_zscore_expr((-0.65 * sopr) + (-0.35 * sopr_ema), 365).alias("brk_roi_context"),
            )
        else:
            base = base.with_columns(
                ((-0.65 * pl.col("_mom_30")) + (-0.35 * pl.col("_mom_90"))).alias("brk_roi_context"),
            )

        if "realized_cap_growth_rate" in btc_df.columns and "market_cap_growth_rate" in btc_df.columns:
            base = base.with_columns(
                _rolling_zscore_expr(
                    pl.col("market_cap_growth_rate").fill_null(0.0)
                    - pl.col("realized_cap_growth_rate").fill_null(0.0),
                    365,
                ).alias("brk_supply_pressure"),
            )

        flow_fast_expr = pl.col("brk_flow").rolling_mean(window_size=7, min_samples=3)
        flow_slow_expr = pl.col("brk_flow").rolling_mean(window_size=30, min_samples=10)
        exchange_level_expr = _rolling_zscore_expr(pl.col("brk_supply_pressure"), 240)
        activity_level_expr = _rolling_zscore_expr(pl.col("brk_activity_div") + pl.col("_mom_30"), 365)

        features = base.with_columns(
            _rolling_zscore_expr(flow_fast_expr, 120).alias("brk_netflow_fast"),
            _rolling_zscore_expr(flow_slow_expr, 240).alias("brk_netflow_slow"),
            _rolling_zscore_expr(flow_fast_expr - flow_slow_expr, 180).alias("brk_netflow_slope"),
            _rolling_zscore_expr(flow_fast_expr, 180).alias("brk_netflow"),
            activity_level_expr.alias("brk_activity_level"),
            pl.col("brk_activity_div").alias("brk_activity_div_fast"),
            (activity_level_expr - pl.col("_mom_90")).alias("brk_activity_div_slow"),
            _rolling_zscore_expr(pl.col("brk_liquidity_impulse"), 180).alias("brk_liquidity_level"),
            exchange_level_expr.alias("brk_exchange_share_level"),
            _rolling_zscore_expr(exchange_level_expr.diff(30), 240).alias("brk_exchange_share_delta"),
            _rolling_zscore_expr(exchange_level_expr, 365).alias("brk_exchange_share"),
            _rolling_zscore_expr(pl.col("_mom_30"), 365).alias("brk_roi30"),
            _rolling_zscore_expr(pl.col("_mom_90"), 365).alias("brk_roi1y"),
        )

        lagged_cols = [
            "brk_flow",
            "brk_supply_pressure",
            "brk_activity_div",
            "brk_roi_context",
            "brk_liquidity_impulse",
            "brk_miner_pressure",
            "brk_hash_momentum",
            "brk_sentiment",
            "brk_netflow_fast",
            "brk_netflow_slow",
            "brk_netflow_slope",
            "brk_netflow",
            "brk_activity_level",
            "brk_activity_div_fast",
            "brk_activity_div_slow",
            "brk_liquidity_level",
            "brk_exchange_share_level",
            "brk_exchange_share_delta",
            "brk_exchange_share",
            "brk_roi30",
            "brk_roi1y",
        ]
        features = features.with_columns(
            [pl.col(c).shift(1).fill_null(0.0).alias(c) for c in lagged_cols]
        ).with_columns(
            [pl.col(c).clip(-6.0, 6.0).alias(c) for c in lagged_cols]
        ).fill_null(0.0)
        return features.select([DATE_COL, *lagged_cols])
