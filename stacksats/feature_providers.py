"""Framework-owned feature providers for causal strategy materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .feature_materialization import build_observed_frame
from .model_development import precompute_features


class FeatureProvider(Protocol):
    """Protocol implemented by framework-owned feature providers."""

    provider_id: str

    def required_source_columns(self) -> tuple[str, ...]:
        """Return the source columns required by this provider."""

    def materialize(
        self,
        btc_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a feature frame indexed by calendar date."""


def _ensure_columns(
    btc_df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    provider_id: str,
) -> None:
    missing = [column for column in columns if column not in btc_df.columns]
    if missing:
        raise ValueError(
            f"Feature provider '{provider_id}' is missing source columns: {missing}."
        )


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(30, window // 3)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _to_numeric(raw: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = raw.copy(deep=True)
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _signed_log1p(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def _btc_frame_cache_key(
    btc_df: pd.DataFrame,
    *,
    price_column: str = "price_usd",
) -> tuple[int, int, int, float, float]:
    idx = pd.DatetimeIndex(btc_df.index).normalize()
    if len(idx) == 0:
        return (0, 0, 0, 0.0, 0.0)
    prices = pd.to_numeric(btc_df.get(price_column), errors="coerce")
    first_price = float(prices.iloc[0]) if len(prices) > 0 else 0.0
    last_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
    if not np.isfinite(first_price):
        first_price = 0.0
    if not np.isfinite(last_price):
        last_price = 0.0
    return (len(idx), int(idx[0].value), int(idx[-1].value), first_price, last_price)


@dataclass(slots=True)
class CoreModelFeatureProvider:
    provider_id: str = "core_model_features_v1"
    _cache_key: tuple[int, int, int, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cache_features: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def required_source_columns(self) -> tuple[str, ...]:
        return ("price_usd",)

    def materialize(
        self,
        btc_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
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


def _overlay_series_from_btc(btc_df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Extract overlay source columns from btc_df if present; empty DataFrame otherwise."""
    out = pd.DataFrame(index=btc_df.index)
    for col in columns:
        if col in btc_df.columns:
            out[col] = pd.to_numeric(btc_df[col], errors="coerce")
    return out


@dataclass(slots=True)
class BRKOverlayFeatureProvider:
    provider_id: str = "brk_overlay_v1"
    _cache_key: tuple[int, int, int, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cache_features: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def required_source_columns(self) -> tuple[str, ...]:
        return ("price_usd", "mvrv")

    def materialize(
        self,
        btc_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        del end_date
        _ensure_columns(
            btc_df,
            self.required_source_columns(),
            provider_id=self.provider_id,
        )
        cache_key = _btc_frame_cache_key(btc_df)
        if self._cache_key == cache_key and self._cache_features is not None:
            features = self._cache_features
            return build_observed_frame(
                features,
                start_date=start_date,
                current_date=as_of_date,
            )

        raw = btc_df.copy(deep=True)
        raw.index = pd.DatetimeIndex(raw.index).normalize()
        raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
        raw = _to_numeric(raw, ("price_usd", "mvrv"))
        features = pd.DataFrame(index=raw.index)
        features["brk_flow"] = 0.0
        features["brk_supply_pressure"] = 0.0
        features["brk_activity_div"] = 0.0
        features["brk_roi_context"] = 0.0
        features["brk_liquidity_impulse"] = 0.0
        features["brk_miner_pressure"] = 0.0
        features["brk_hash_momentum"] = 0.0
        features["brk_sentiment"] = 0.0

        price = pd.to_numeric(raw["price_usd"], errors="coerce")
        price_log = np.log(price.where(price > 0.0))
        mom_30 = _rolling_zscore(price_log.diff(30), 365)
        mom_90 = _rolling_zscore(price_log.diff(90), 365)

        flow_df = _overlay_series_from_btc(raw, ("adjusted_sopr", "adjusted_sopr_7d_ema"))
        if not flow_df.empty and {"adjusted_sopr", "adjusted_sopr_7d_ema"}.issubset(flow_df.columns):
            flow = flow_df.reindex(features.index).ffill()
            sopr = pd.to_numeric(flow["adjusted_sopr"], errors="coerce")
            sopr_ema = pd.to_numeric(flow["adjusted_sopr_7d_ema"], errors="coerce")
            features["brk_flow"] = _rolling_zscore((sopr - sopr_ema), 240)
            features["brk_roi_context"] = _rolling_zscore((-0.65 * sopr) + (-0.35 * sopr_ema), 365)
        else:
            features["brk_roi_context"] = (-0.65 * mom_30) + (-0.35 * mom_90)

        supply_df = _overlay_series_from_btc(raw, ("realized_cap_growth_rate", "market_cap_growth_rate"))
        if not supply_df.empty and {"realized_cap_growth_rate", "market_cap_growth_rate"}.issubset(supply_df.columns):
            supply = supply_df.reindex(features.index).ffill()
            realized = pd.to_numeric(supply["realized_cap_growth_rate"], errors="coerce")
            market = pd.to_numeric(supply["market_cap_growth_rate"], errors="coerce")
            features["brk_supply_pressure"] = _rolling_zscore(market - realized, 365)

        tx_df = _overlay_series_from_btc(raw, ("tx_count_pct10", "annualized_volume_usd"))
        if not tx_df.empty and {"tx_count_pct10", "annualized_volume_usd"}.issubset(tx_df.columns):
            tx = tx_df.reindex(features.index).ffill()
            tx_count = _signed_log1p(pd.to_numeric(tx["tx_count_pct10"], errors="coerce"))
            tx_vol = _signed_log1p(pd.to_numeric(tx["annualized_volume_usd"], errors="coerce"))
            activity = (_rolling_zscore(tx_count, 365) + _rolling_zscore(tx_vol, 365)) / 2.0
            features["brk_activity_div"] = activity - mom_30
            features["brk_liquidity_impulse"] = _rolling_zscore(tx_vol.diff(7), 180)

        hash_df = _overlay_series_from_btc(raw, ("hash_rate_1y_sma", "subsidy_usd_average"))
        if not hash_df.empty and {"hash_rate_1y_sma", "subsidy_usd_average"}.issubset(hash_df.columns):
            hr = hash_df.reindex(features.index).ffill()
            hash_rate = _signed_log1p(pd.to_numeric(hr["hash_rate_1y_sma"], errors="coerce"))
            subsidy = _signed_log1p(pd.to_numeric(hr["subsidy_usd_average"], errors="coerce"))
            features["brk_hash_momentum"] = _rolling_zscore(hash_rate.diff(30), 365)
            features["brk_miner_pressure"] = _rolling_zscore(subsidy - hash_rate, 365)

        sentiment_df = _overlay_series_from_btc(raw, ("net_sentiment", "greed_index", "pain_index"))
        if not sentiment_df.empty:
            sentiment = sentiment_df.reindex(features.index).ffill()
            components: list[pd.Series] = []
            for column in ("net_sentiment", "greed_index", "pain_index"):
                if column in sentiment.columns:
                    components.append(_rolling_zscore(pd.to_numeric(sentiment[column], errors="coerce"), 365))
            if components:
                features["brk_sentiment"] = sum(components) / float(len(components))

        # Compatibility aliases for existing built-in strategy formulas.
        flow_fast = features["brk_flow"].rolling(7, min_periods=3).mean()
        flow_slow = features["brk_flow"].rolling(30, min_periods=10).mean()
        features["brk_netflow_fast"] = _rolling_zscore(flow_fast, 120)
        features["brk_netflow_slow"] = _rolling_zscore(flow_slow, 240)
        features["brk_netflow_slope"] = _rolling_zscore(flow_fast - flow_slow, 180)
        features["brk_netflow"] = _rolling_zscore(flow_fast, 180)

        activity_level = _rolling_zscore(features["brk_activity_div"] + mom_30, 365)
        features["brk_activity_level"] = activity_level
        features["brk_activity_div_fast"] = features["brk_activity_div"]
        features["brk_activity_div_slow"] = activity_level - mom_90

        features["brk_liquidity_level"] = _rolling_zscore(features["brk_liquidity_impulse"], 180)

        exchange_level = _rolling_zscore(features["brk_supply_pressure"], 240)
        features["brk_exchange_share_level"] = exchange_level
        features["brk_exchange_share_delta"] = _rolling_zscore(exchange_level.diff(30), 240)
        features["brk_exchange_share"] = _rolling_zscore(exchange_level, 365)

        features["brk_roi30"] = _rolling_zscore(mom_30, 365)
        features["brk_roi1y"] = _rolling_zscore(mom_90, 365)

        features = features.shift(1)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features = features.clip(-6.0, 6.0)
        self._cache_key = cache_key
        self._cache_features = features
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )
