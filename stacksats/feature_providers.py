"""Framework-owned feature providers for causal strategy materialization."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class CoreModelFeatureProvider:
    provider_id: str = "core_model_features_v1"

    def required_source_columns(self) -> tuple[str, ...]:
        return ("PriceUSD_coinmetrics",)

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
        features = precompute_features(btc_df)
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )


@dataclass(frozen=True, slots=True)
class CoinMetricsOverlayFeatureProvider:
    provider_id: str = "coinmetrics_overlay_v1"

    def required_source_columns(self) -> tuple[str, ...]:
        return ("PriceUSD_coinmetrics",)

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
        raw = btc_df.copy(deep=True)
        raw.index = pd.DatetimeIndex(raw.index).normalize()
        raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
        raw = _to_numeric(
            raw,
            (
                "PriceUSD_coinmetrics",
                "PriceUSD",
                "CapMrktCurUSD",
                "FlowInExUSD",
                "FlowOutExUSD",
                "AdrActCnt",
                "TxCnt",
                "TxTfrCnt",
                "FeeTotNtv",
                "volume_reported_spot_usd_1d",
                "SplyExNtv",
                "SplyCur",
                "IssTotUSD",
                "HashRate",
                "ROI30d",
                "ROI1yr",
            ),
        )
        if "PriceUSD" not in raw.columns:
            raw["PriceUSD"] = raw["PriceUSD_coinmetrics"]

        features = pd.DataFrame(index=raw.index)

        if {"FlowInExUSD", "FlowOutExUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            flow_ratio = (raw["FlowInExUSD"] - raw["FlowOutExUSD"]) / cap
            flow_fast = flow_ratio.rolling(7, min_periods=3).mean()
            flow_slow = flow_ratio.rolling(30, min_periods=10).mean()
            features["cm_netflow_fast"] = _rolling_zscore(flow_fast, 120)
            features["cm_netflow_slow"] = _rolling_zscore(flow_slow, 240)
            features["cm_netflow_slope"] = _rolling_zscore(flow_fast - flow_slow, 180)
            features["cm_netflow"] = _rolling_zscore(flow_fast, 180)
        else:
            features["cm_netflow_fast"] = 0.0
            features["cm_netflow_slow"] = 0.0
            features["cm_netflow_slope"] = 0.0
            features["cm_netflow"] = 0.0

        activity_parts: list[pd.Series] = []
        for column in ("AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv"):
            if column in raw.columns:
                activity_parts.append(_rolling_zscore(np.log1p(raw[column]), 365))
        price_log = np.log(raw["PriceUSD"].replace(0.0, np.nan))
        mom_30 = _rolling_zscore(price_log.diff(30), 365)
        mom_90 = _rolling_zscore(price_log.diff(90), 365)
        if activity_parts:
            activity = sum(activity_parts) / float(len(activity_parts))
            features["cm_activity_level"] = activity
            features["cm_activity_div_fast"] = activity - mom_30
            features["cm_activity_div_slow"] = activity - mom_90
            features["cm_activity_div"] = activity - mom_30
        else:
            features["cm_activity_level"] = 0.0
            features["cm_activity_div_fast"] = 0.0
            features["cm_activity_div_slow"] = 0.0
            features["cm_activity_div"] = 0.0

        if "volume_reported_spot_usd_1d" in raw.columns:
            vol_log = np.log1p(raw["volume_reported_spot_usd_1d"])
            features["cm_liquidity_level"] = _rolling_zscore(vol_log, 180)
            features["cm_liquidity_impulse"] = _rolling_zscore(vol_log.diff(7), 120)
        else:
            features["cm_liquidity_level"] = 0.0
            features["cm_liquidity_impulse"] = 0.0

        if {"SplyExNtv", "SplyCur"}.issubset(raw.columns):
            exchange_share = raw["SplyExNtv"] / raw["SplyCur"].replace(0.0, np.nan)
            features["cm_exchange_share_level"] = _rolling_zscore(exchange_share, 240)
            features["cm_exchange_share_delta"] = _rolling_zscore(
                exchange_share.diff(30),
                240,
            )
            features["cm_exchange_share"] = _rolling_zscore(exchange_share, 365)
        else:
            features["cm_exchange_share_level"] = 0.0
            features["cm_exchange_share_delta"] = 0.0
            features["cm_exchange_share"] = 0.0

        if {"IssTotUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            features["cm_miner_pressure"] = _rolling_zscore(raw["IssTotUSD"] / cap, 365)
        else:
            features["cm_miner_pressure"] = 0.0

        if "HashRate" in raw.columns:
            hash_log = np.log(raw["HashRate"].replace(0.0, np.nan))
            hash_fast = _rolling_zscore(hash_log.diff(30), 365)
            hash_slow = _rolling_zscore(hash_log.diff(90), 365)
            features["cm_hash_momentum"] = (0.6 * hash_fast) + (0.4 * hash_slow)
        else:
            features["cm_hash_momentum"] = 0.0

        if "ROI30d" in raw.columns:
            features["cm_roi30"] = _rolling_zscore(raw["ROI30d"], 365)
        else:
            features["cm_roi30"] = mom_30
        if "ROI1yr" in raw.columns:
            features["cm_roi1y"] = _rolling_zscore(raw["ROI1yr"], 365)
        else:
            features["cm_roi1y"] = mom_90
        features["cm_roi_context"] = (-0.65 * features["cm_roi30"]) + (
            -0.35 * features["cm_roi1y"]
        )

        features = features.shift(1)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features = features.clip(-6.0, 6.0)
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )

