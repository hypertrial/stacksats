"""Registry for framework-owned feature providers."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import polars as pl

from .feature_materialization import hash_dataframe, normalize_timestamp
from .feature_providers import (
    BRKOverlayFeatureProvider,
    CoreModelFeatureProvider,
    FeatureProvider,
)

DATE_COL = "date"


def _lazy_daily_date_expr(dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.Utf8:
        base = pl.col(DATE_COL).str.to_datetime(strict=False)
    elif dtype == pl.Date:
        base = pl.col(DATE_COL).cast(pl.Datetime)
    else:
        base = pl.col(DATE_COL).cast(pl.Datetime, strict=False)
    return base.dt.replace_time_zone(None).dt.truncate("1d")


def _lazy_observed_frame(
    frame_lazy: pl.LazyFrame,
    *,
    schema: pl.Schema,
    start_date: dt.datetime,
    current_date: dt.datetime,
) -> pl.LazyFrame:
    if DATE_COL not in schema:
        raise ValueError(f"features_df must have '{DATE_COL}' column.")
    return (
        frame_lazy
        .with_columns(_lazy_daily_date_expr(schema[DATE_COL]).alias(DATE_COL))
        .filter((pl.col(DATE_COL) >= start_date) & (pl.col(DATE_COL) <= current_date))
        .unique(subset=[DATE_COL], keep="last")
        .sort(DATE_COL)
    )


class FeatureRegistry:
    """Registry for deterministic feature provider lookup and materialization."""

    def __init__(self) -> None:
        self._providers: dict[str, FeatureProvider] = {}

    def register(self, provider: FeatureProvider) -> None:
        provider_id = str(provider.provider_id)
        if provider_id in self._providers:
            raise ValueError(f"Feature provider '{provider_id}' is already registered.")
        self._providers[provider_id] = provider

    def get(self, provider_id: str) -> FeatureProvider:
        if provider_id not in self._providers:
            raise KeyError(f"Unknown feature provider '{provider_id}'.")
        return self._providers[provider_id]

    def materialize_for_strategy_lazy(
        self,
        strategy,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
    ) -> pl.LazyFrame:
        observed_end = min(normalize_timestamp(current_date), normalize_timestamp(end_date))
        start_ts = normalize_timestamp(start_date)
        dates = pl.datetime_range(start_ts, observed_end, interval="1d", eager=True)
        provider_ids = tuple(strategy.required_feature_sets())
        if not provider_ids:
            return pl.DataFrame({DATE_COL: dates}).lazy()

        merged_lazy = pl.DataFrame({DATE_COL: dates}).lazy()
        merged_columns = {DATE_COL}
        for provider_id in provider_ids:
            provider = self.get(provider_id)
            if hasattr(provider, "materialize_lazy"):
                frame_lazy = provider.materialize_lazy(
                    btc_df,
                    start_date=start_ts,
                    end_date=normalize_timestamp(end_date),
                    as_of_date=observed_end,
                )
            else:
                frame_lazy = provider.materialize(
                    btc_df,
                    start_date=start_ts,
                    end_date=normalize_timestamp(end_date),
                    as_of_date=observed_end,
                ).lazy()
            observed_schema = frame_lazy.collect_schema()
            observed = _lazy_observed_frame(
                frame_lazy,
                schema=observed_schema,
                start_date=start_ts,
                current_date=observed_end,
            )
            duplicate_columns = sorted(
                (set(observed_schema.names()) & merged_columns) - {DATE_COL}
            )
            if duplicate_columns:
                raise ValueError(
                    f"Feature providers produced duplicate columns for strategy "
                    f"{strategy.metadata().strategy_id}: {duplicate_columns}."
                )
            merged_columns.update(name for name in observed_schema.names() if name != DATE_COL)
            merged_lazy = merged_lazy.join(observed, on=DATE_COL, how="left")
        return merged_lazy.sort(DATE_COL)

    def materialize_for_strategy(
        self,
        strategy,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
    ) -> pl.DataFrame:
        return self.materialize_for_strategy_lazy(
            strategy,
            btc_df,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        ).collect()

    def provider_fingerprint(self, strategy) -> str:
        payload = {"provider_ids": list(strategy.required_feature_sets())}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def materialization_fingerprint(
        self,
        strategy,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
    ) -> tuple[pl.DataFrame, str]:
        frame = self.materialize_for_strategy(
            strategy,
            btc_df,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        )
        payload = {
            "providers": list(strategy.required_feature_sets()),
            "provider_fingerprint": self.provider_fingerprint(strategy),
            "frame_hash": hash_dataframe(frame),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return frame, digest


DEFAULT_FEATURE_REGISTRY = FeatureRegistry()
DEFAULT_FEATURE_REGISTRY.register(CoreModelFeatureProvider())
DEFAULT_FEATURE_REGISTRY.register(BRKOverlayFeatureProvider())
