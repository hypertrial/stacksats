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


def _resolve_duckdb_path(default_duckdb_path: str) -> Path:
    env_path = os.getenv("STACKSATS_ANALYTICS_DUCKDB")
    path = Path(env_path or default_duckdb_path).expanduser()
    if not path.exists():
        raise ValueError(
            f"DuckDB file not found at '{path}'. "
            "Set STACKSATS_ANALYTICS_DUCKDB or place a file at "
            f"'{default_duckdb_path}'."
        )
    return path


def _load_long_metrics(
    connection,
    *,
    table_name: str,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    escaped = ", ".join("'" + metric.replace("'", "''") + "'" for metric in metrics)
    query = (
        "SELECT date_day, metric, value "
        f"FROM {table_name} "
        f"WHERE metric IN ({escaped}) "
        "ORDER BY date_day"
    )
    long_df = connection.execute(query).fetchdf()
    if long_df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date_day"))
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
    return wide_df


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


@dataclass(slots=True)
class BRKOverlayFeatureProvider:
    provider_id: str = "brk_overlay_v1"
    default_duckdb_path: str = "./bitcoin_analytics.duckdb"
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

        db_path = _resolve_duckdb_path(self.default_duckdb_path)
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "duckdb is not installed. Provider 'brk_overlay_v1' requires the BRK "
                "DuckDB backend. Install it with: pip install stacksats[brk]"
            ) from exc

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            flow_df = _load_long_metrics(
                con,
                table_name="metrics_distribution",
                metrics=("adjusted_sopr", "adjusted_sopr_7d_ema"),
            )
            supply_df = _load_long_metrics(
                con,
                table_name="metrics_supply",
                metrics=("realized_cap_growth_rate", "market_cap_growth_rate"),
            )
            tx_df = _load_long_metrics(
                con,
                table_name="metrics_transactions",
                metrics=("tx_count_pct10", "annualized_volume_usd"),
            )
            hash_df = _load_long_metrics(
                con,
                table_name="metrics_blocks",
                metrics=("hash_rate_1y_sma", "subsidy_usd_average"),
            )
            sentiment_df = _load_long_metrics(
                con,
                table_name="metrics_distribution",
                metrics=("net_sentiment", "greed_index", "pain_index"),
            )
        finally:
            con.close()

        price = pd.to_numeric(raw["price_usd"], errors="coerce")
        price_log = np.log(price.where(price > 0.0))
        mom_30 = _rolling_zscore(price_log.diff(30), 365)
        mom_90 = _rolling_zscore(price_log.diff(90), 365)

        if not flow_df.empty and {"adjusted_sopr", "adjusted_sopr_7d_ema"}.issubset(flow_df.columns):
            flow = flow_df.reindex(features.index).ffill()
            sopr = pd.to_numeric(flow["adjusted_sopr"], errors="coerce")
            sopr_ema = pd.to_numeric(flow["adjusted_sopr_7d_ema"], errors="coerce")
            features["brk_flow"] = _rolling_zscore((sopr - sopr_ema), 240)
            features["brk_roi_context"] = _rolling_zscore((-0.65 * sopr) + (-0.35 * sopr_ema), 365)
        else:
            features["brk_roi_context"] = (-0.65 * mom_30) + (-0.35 * mom_90)

        if not supply_df.empty and {"realized_cap_growth_rate", "market_cap_growth_rate"}.issubset(supply_df.columns):
            supply = supply_df.reindex(features.index).ffill()
            realized = pd.to_numeric(supply["realized_cap_growth_rate"], errors="coerce")
            market = pd.to_numeric(supply["market_cap_growth_rate"], errors="coerce")
            features["brk_supply_pressure"] = _rolling_zscore(market - realized, 365)

        if not tx_df.empty and {"tx_count_pct10", "annualized_volume_usd"}.issubset(tx_df.columns):
            tx = tx_df.reindex(features.index).ffill()
            tx_count = _signed_log1p(pd.to_numeric(tx["tx_count_pct10"], errors="coerce"))
            tx_vol = _signed_log1p(pd.to_numeric(tx["annualized_volume_usd"], errors="coerce"))
            activity = (_rolling_zscore(tx_count, 365) + _rolling_zscore(tx_vol, 365)) / 2.0
            features["brk_activity_div"] = activity - mom_30
            features["brk_liquidity_impulse"] = _rolling_zscore(tx_vol.diff(7), 180)

        if not hash_df.empty and {"hash_rate_1y_sma", "subsidy_usd_average"}.issubset(hash_df.columns):
            hr = hash_df.reindex(features.index).ffill()
            hash_rate = _signed_log1p(pd.to_numeric(hr["hash_rate_1y_sma"], errors="coerce"))
            subsidy = _signed_log1p(pd.to_numeric(hr["subsidy_usd_average"], errors="coerce"))
            features["brk_hash_momentum"] = _rolling_zscore(hash_rate.diff(30), 365)
            features["brk_miner_pressure"] = _rolling_zscore(subsidy - hash_rate, 365)

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
        return ("price_usd",)

    def _resolve_duckdb_path(self) -> Path:
        return _resolve_duckdb_path(self.default_duckdb_path)

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
        except ImportError as exc:
            raise RuntimeError(
                "duckdb is not installed. Provider 'duckdb_analytics_factors_v1' requires "
                "the BRK DuckDB backend. Install it with: pip install stacksats[brk]"
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
