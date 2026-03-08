"""BTC data provider backed by local BRK-generated DuckDB metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

DUCKDB_DEFAULT_PATH = "./bitcoin_analytics.duckdb"
DUCKDB_ENV_VAR = "STACKSATS_ANALYTICS_DUCKDB"


class DataLoadError(RuntimeError):
    """Raised when BTC source data cannot be loaded safely."""


def _resolve_duckdb_path(path_override: str | None) -> Path:
    path = Path(os.getenv(DUCKDB_ENV_VAR) or path_override or DUCKDB_DEFAULT_PATH).expanduser()
    if not path.exists():
        raise DataLoadError(
            f"DuckDB file not found at '{path}'. Set {DUCKDB_ENV_VAR} or provide duckdb_path."
        )
    return path


def _require_daily_index(df: pd.DataFrame, *, backtest_start_ts: pd.Timestamp, target_end: pd.Timestamp) -> None:
    expected = pd.date_range(start=backtest_start_ts, end=target_end, freq="D")
    if not df.index.equals(expected):
        missing = expected.difference(df.index)
        first_missing = missing[0].strftime("%Y-%m-%d") if len(missing) else "unknown"
        raise DataLoadError(
            "BRK DuckDB data has missing dates in requested window. "
            f"First missing date: {first_missing}."
        )


def _load_btc_from_duckdb(path: Path) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise DataLoadError("duckdb package is required for BTCDataProvider.") from exc

    con = duckdb.connect(str(path), read_only=True)
    try:
        price_df = con.execute(
            """
            SELECT date_day, value AS price_usd
            FROM metrics_price
            WHERE metric = 'price_close'
            ORDER BY date_day
            """
        ).fetchdf()
        mvrv_df = con.execute(
            """
            SELECT date_day, value AS mvrv
            FROM metrics_distribution
            WHERE metric = 'mvrv'
            ORDER BY date_day
            """
        ).fetchdf()
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(
            "Could not query required BRK tables/metrics "
            "(metrics_price.price_close, metrics_distribution.mvrv)."
        ) from exc
    finally:
        con.close()

    if price_df.empty:
        raise DataLoadError("No rows found for metrics_price.price_close.")

    price_df["date_day"] = pd.to_datetime(price_df["date_day"]).dt.normalize()
    price_df["price_usd"] = pd.to_numeric(price_df["price_usd"], errors="coerce")
    price_df = price_df.dropna(subset=["date_day"]).sort_values("date_day")
    price_df = price_df.drop_duplicates(subset=["date_day"], keep="last")

    mvrv_df["date_day"] = pd.to_datetime(mvrv_df["date_day"]).dt.normalize()
    mvrv_df["mvrv"] = pd.to_numeric(mvrv_df["mvrv"], errors="coerce")
    mvrv_df = mvrv_df.dropna(subset=["date_day"]).sort_values("date_day")
    mvrv_df = mvrv_df.drop_duplicates(subset=["date_day"], keep="last")

    merged = price_df.merge(mvrv_df, on="date_day", how="left")
    merged = merged.set_index("date_day")
    merged.index = pd.DatetimeIndex(merged.index).normalize()
    merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
    if merged.empty:
        raise DataLoadError("Merged BRK BTC frame is empty.")
    return merged


@dataclass
class BTCDataProvider:
    """BTC-only provider using BRK-generated local-node DuckDB metrics."""

    duckdb_path: str | None = None
    clock: Callable[[], pd.Timestamp] = pd.Timestamp.now
    max_staleness_days: int = 3

    def load(
        self,
        *,
        backtest_start: str = "2018-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        now = self.clock().normalize()
        backtest_start_ts = pd.to_datetime(backtest_start).normalize()
        if end_date is not None:
            try:
                target_end = pd.to_datetime(end_date).normalize()
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid end_date value: {end_date!r}") from exc
        else:
            target_end = now
        target_end = min(target_end, now)
        if target_end < backtest_start_ts:
            raise ValueError(
                "end_date must be on or after backtest_start. "
                f"Received backtest_start={backtest_start_ts.date()} and end_date={target_end.date()}."
            )

        frame = _load_btc_from_duckdb(_resolve_duckdb_path(self.duckdb_path))
        if "price_usd" not in frame.columns:
            raise DataLoadError("Required price_usd series missing from BRK data.")

        latest_price_idx = frame.index[pd.to_numeric(frame["price_usd"], errors="coerce").notna()]
        if len(latest_price_idx) == 0:
            raise DataLoadError("BRK data contains no valid price_usd values.")
        latest_price_date = latest_price_idx.max().normalize()
        if latest_price_date < target_end:
            raise DataLoadError(
                "BRK data does not cover requested end_date. "
                f"Latest available={latest_price_date.date()}, requested={target_end.date()}."
            )
        if latest_price_date < (now - pd.Timedelta(days=int(self.max_staleness_days))):
            raise DataLoadError(
                "BRK data is stale for runtime usage. "
                f"Latest available={latest_price_date.date()}, now={now.date()}."
            )

        window = frame.loc[backtest_start_ts:target_end].copy()
        if window.empty:
            raise DataLoadError("No BRK rows available in requested backtest window.")
        if window["price_usd"].isna().any():
            first_missing = window.index[window["price_usd"].isna()][0].strftime("%Y-%m-%d")
            raise DataLoadError(
                "BRK data has missing price_usd values in requested window. "
                f"First missing date: {first_missing}."
            )
        _require_daily_index(
            window,
            backtest_start_ts=backtest_start_ts,
            target_end=target_end,
        )
        return window
