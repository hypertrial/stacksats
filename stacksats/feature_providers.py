"""Framework-owned feature providers for causal strategy materialization."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
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
    price_column: str = "PriceUSD_coinmetrics",
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


@dataclass(slots=True)
class CoinMetricsOverlayFeatureProvider:
    provider_id: str = "coinmetrics_overlay_v1"
    _cache_key: tuple[int, int, int, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cache_features: pd.DataFrame | None = field(default=None, init=False, repr=False)

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
                non_negative = pd.to_numeric(raw[column], errors="coerce").clip(lower=0.0)
                activity_parts.append(_rolling_zscore(np.log1p(non_negative), 365))
        price_series = pd.to_numeric(raw["PriceUSD"], errors="coerce")
        price_log = np.log(price_series.where(price_series > 0.0))
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
            vol_series = pd.to_numeric(
                raw["volume_reported_spot_usd_1d"],
                errors="coerce",
            ).clip(lower=0.0)
            vol_log = np.log1p(vol_series)
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
            hash_rate = pd.to_numeric(raw["HashRate"], errors="coerce")
            hash_log = np.log(hash_rate.where(hash_rate > 0.0))
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
        self._cache_key = cache_key
        self._cache_features = features
        return build_observed_frame(
            features,
            start_date=start_date,
            current_date=as_of_date,
        )


DUCKDB_ANALYTICS_METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "metrics_market": (
        "days_since_price_ath",
        "price_1y_sma_ratio_2y_m3sd",
        "price_1y_sma_ratio_2y_m2_5sd",
        "price_350d_sma_ratio_2y_m3sd",
        "price_2y_ema_ratio_m1_5sd",
    ),
    "metrics_cointime": (
        "reserve_risk",
        "activity_to_vaultedness_ratio",
        "liveliness",
        "vaultedness",
        "cointime_price_ratio",
        "true_market_mean_ratio_pct95",
    ),
    "metrics_supply": (
        "btc_velocity",
        "realized_cap_growth_rate",
        "market_cap_growth_rate",
        "cap_growth_rate_diff",
        "usd_velocity",
    ),
    "metrics_transactions": (
        "received_sum_usd",
        "sent_sum_usd",
        "annualized_volume_usd",
        "tx_count_pct25",
        "tx_count_pct10",
        "tx_v1",
    ),
    "metrics_price": (
        "price_close",
        "price_low",
        "price_high",
        "price_sats_close",
        "price_sats_low",
    ),
    "metrics_pools": (
        "braiinspool_subsidy_usd",
        "braiinspool_coinbase_usd",
        "bytepool_1y_dominance",
        "okexpool_1m_dominance",
        "onethash_1m_dominance",
    ),
    "metrics_blocks": (
        "subsidy_usd_sum",
        "subsidy_usd_average",
        "coinbase_usd_min",
        "hash_rate_1y_sma",
        "block_fullness_median",
    ),
    "metrics_scripts": (
        "p2pk33_count_cumulative",
        "p2pkh_count_pct25",
        "p2sh_count_pct10",
        "p2tr_count_pct90",
        "taproot_adoption_sum",
    ),
    # Distribution is intentionally filtered to a compact, stable subset.
    "metrics_distribution": (
        "net_realized_pnl_rel_to_realized_cap_cumulative",
        "utxos_under_8y_old_net_realized_pnl_rel_to_realized_cap_cumulative",
        "utxos_under_7y_old_net_realized_pnl_rel_to_realized_cap_cumulative",
        "utxos_over_10m_sats_net_realized_pnl_rel_to_realized_cap_cumulative",
        "adjusted_sopr",
        "adjusted_sopr_7d_ema",
        "greed_index",
        "pain_index",
        "net_sentiment",
    ),
}


def _sanitize_metric_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return clean or "metric"


def _duckdb_metric_column(table_name: str, metric: str) -> str:
    table_suffix = table_name.replace("metrics_", "")
    return f"ddb_{table_suffix}_{_sanitize_metric_name(metric)}"


@dataclass(slots=True)
class DuckDBAnalyticsFeatureProvider:
    provider_id: str = "duckdb_analytics_factors_v1"
    default_duckdb_path: str = "./bitcoin_analytics.duckdb"
    max_abs_feature: float = 8.0
    _cache_key: tuple[str, int, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cache_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _aligned_cache_key: tuple[tuple[str, int, int], str, str, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _aligned_cache_frame: pd.DataFrame | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def required_source_columns(self) -> tuple[str, ...]:
        return ("PriceUSD_coinmetrics",)

    def _resolve_duckdb_path(self) -> Path:
        env_path = os.getenv("STACKSATS_ANALYTICS_DUCKDB")
        path = Path(env_path or self.default_duckdb_path).expanduser()
        if not path.exists():
            raise ValueError(
                "Feature provider 'duckdb_analytics_factors_v1' could not find DuckDB "
                f"file at '{path}'. Set STACKSATS_ANALYTICS_DUCKDB or place "
                f"'{self.default_duckdb_path}' in the working directory."
            )
        return path

    @staticmethod
    def _source_key(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))

    @staticmethod
    def _fetch_table_metrics(
        connection,
        *,
        table_name: str,
        metrics: tuple[str, ...],
    ) -> pd.DataFrame:
        escaped = ", ".join(
            "'" + metric.replace("'", "''") + "'" for metric in metrics
        )
        query = (
            "SELECT date_day, metric, value "
            f"FROM {table_name} "
            f"WHERE metric IN ({escaped}) "
            "ORDER BY date_day"
        )
        long_df = connection.execute(query).fetchdf()
        if long_df.empty:
            raise ValueError(
                f"Feature provider 'duckdb_analytics_factors_v1' found no rows for table "
                f"'{table_name}'."
            )
        long_df["date_day"] = pd.to_datetime(long_df["date_day"]).dt.normalize()
        wide_df = (
            long_df.pivot_table(
                index="date_day",
                columns="metric",
                values="value",
                aggfunc="last",
            )
            .sort_index()
            .copy()
        )
        missing_metrics = [metric for metric in metrics if metric not in wide_df.columns]
        if missing_metrics:
            raise ValueError(
                "Feature provider 'duckdb_analytics_factors_v1' is missing expected "
                f"metrics in '{table_name}': {missing_metrics}."
            )
        renamed = {
            metric: _duckdb_metric_column(table_name, metric)
            for metric in metrics
        }
        return wide_df[list(metrics)].rename(columns=renamed)

    @staticmethod
    def _validate_tables(connection) -> None:
        available_tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main'"
            ).fetchall()
        }
        missing_tables = sorted(
            set(DUCKDB_ANALYTICS_METRIC_FAMILIES).difference(available_tables)
        )
        if missing_tables:
            raise ValueError(
                "Feature provider 'duckdb_analytics_factors_v1' is missing required "
                f"DuckDB tables: {missing_tables}."
            )

    def _load_duckdb_frame(self, path: Path) -> pd.DataFrame:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - dependency should be installed
            raise RuntimeError(
                "duckdb is required for provider 'duckdb_analytics_factors_v1'. "
                "Install stacksats with duckdb dependency support."
            ) from exc

        con = duckdb.connect(str(path), read_only=True)
        try:
            self._validate_tables(con)
            frames: list[pd.DataFrame] = []
            for table_name, metrics in DUCKDB_ANALYTICS_METRIC_FAMILIES.items():
                frames.append(
                    self._fetch_table_metrics(
                        con,
                        table_name=table_name,
                        metrics=metrics,
                    )
                )
        finally:
            con.close()
        merged = pd.concat(frames, axis=1).sort_index()
        merged.index = pd.DatetimeIndex(merged.index).normalize()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        return merged

    def _engineer_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        engineered_columns: dict[str, pd.Series] = {}
        for column in raw_df.columns:
            transformed = _signed_log1p(raw_df[column])
            engineered_columns[f"{column}_level"] = _rolling_zscore(transformed, 365)
            engineered_columns[f"{column}_mom7"] = _rolling_zscore(transformed.diff(7), 240)
            engineered_columns[f"{column}_mom30"] = _rolling_zscore(transformed.diff(30), 365)

        engineered = pd.DataFrame(engineered_columns, index=raw_df.index)

        reserve_col = _duckdb_metric_column("metrics_cointime", "reserve_risk")
        drawdown_col = _duckdb_metric_column("metrics_market", "days_since_price_ath")
        sopr_col = _duckdb_metric_column("metrics_distribution", "adjusted_sopr")
        sentiment_col = _duckdb_metric_column("metrics_distribution", "net_sentiment")
        fee_col = _duckdb_metric_column("metrics_transactions", "tx_count_pct10")
        hash_col = _duckdb_metric_column("metrics_blocks", "hash_rate_1y_sma")

        def _col(name: str) -> pd.Series:
            if name in engineered.columns:
                return pd.to_numeric(engineered[name], errors="coerce").fillna(0.0)
            return pd.Series(0.0, index=engineered.index, dtype=float)

        engineered_columns["ddb_cross_reserve_drawdown"] = _rolling_zscore(
            _col(f"{reserve_col}_level") * (-_col(f"{drawdown_col}_level")),
            365,
        )
        engineered_columns["ddb_cross_sopr_sentiment"] = _rolling_zscore(
            _col(f"{sopr_col}_level") * _col(f"{sentiment_col}_mom7"),
            365,
        )
        engineered_columns["ddb_cross_fee_hash"] = _rolling_zscore(
            _col(f"{fee_col}_mom7") - _col(f"{hash_col}_mom30"),
            365,
        )
        engineered = pd.DataFrame(engineered_columns, index=raw_df.index)

        engineered = engineered.shift(1)
        engineered = engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return engineered.clip(-self.max_abs_feature, self.max_abs_feature)

    def _load_cached_frame(self, path: Path) -> tuple[pd.DataFrame, tuple[str, int, int]]:
        source_key = self._source_key(path)
        if self._cache_key == source_key and self._cache_frame is not None:
            return self._cache_frame, source_key

        raw_df = self._load_duckdb_frame(path)
        self._cache_frame = self._engineer_features(raw_df)
        self._cache_key = source_key
        self._aligned_cache_key = None
        self._aligned_cache_frame = None
        return self._cache_frame, source_key

    def _align_to_btc_index(
        self,
        engineered: pd.DataFrame,
        source_key: tuple[str, int, int],
        btc_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        if len(btc_index) == 0:
            return pd.DataFrame(index=btc_index)
        aligned_key = (
            source_key,
            btc_index.min().strftime("%Y-%m-%d"),
            btc_index.max().strftime("%Y-%m-%d"),
            len(btc_index),
        )
        if self._aligned_cache_key == aligned_key and self._aligned_cache_frame is not None:
            return self._aligned_cache_frame

        aligned = engineered.reindex(btc_index).ffill()
        if len(engineered.index) > 0:
            last_available = engineered.index.max()
            aligned.loc[aligned.index > last_available] = 0.0
        aligned = aligned.fillna(0.0)

        self._aligned_cache_key = aligned_key
        self._aligned_cache_frame = aligned
        return aligned

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
        db_path = self._resolve_duckdb_path()
        engineered, source_key = self._load_cached_frame(db_path)

        btc_index = pd.DatetimeIndex(btc_df.index).normalize().sort_values()
        aligned = self._align_to_btc_index(engineered, source_key, btc_index)

        return build_observed_frame(
            aligned,
            start_date=start_date,
            current_date=as_of_date,
        )
