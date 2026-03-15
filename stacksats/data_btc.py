"""BTC data provider backed by local BRK-generated parquet metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

PARQUET_DEFAULT_PATH = "./bitcoin_analytics.parquet"
PARQUET_ENV_VAR = "STACKSATS_ANALYTICS_PARQUET"


class DataLoadError(RuntimeError):
    """Raised when BTC source data cannot be loaded safely."""


def _resolve_parquet_path(path_override: str | None) -> Path:
    path = Path(os.getenv(PARQUET_ENV_VAR) or path_override or PARQUET_DEFAULT_PATH).expanduser()
    if not path.exists():
        raise DataLoadError(
            f"Parquet file not found at '{path}'. Set {PARQUET_ENV_VAR} or provide parquet_path."
        )
    return path


def _require_daily_index(df: pd.DataFrame, *, backtest_start_ts: pd.Timestamp, target_end: pd.Timestamp) -> None:
    expected = pd.date_range(start=backtest_start_ts, end=target_end, freq="D")
    if not df.index.equals(expected):
        missing = expected.difference(df.index)
        first_missing = missing[0].strftime("%Y-%m-%d") if len(missing) else "unknown"
        raise DataLoadError(
            "BRK parquet data has missing dates in requested window. "
            f"First missing date: {first_missing}."
        )


def _load_btc_from_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise DataLoadError("Parquet file is empty.")
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = frame.index.normalize()
    elif "date" in frame.columns:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        frame = frame.set_index("date")
        frame.index = pd.DatetimeIndex(frame.index).normalize()
    else:
        raise DataLoadError("Parquet must have a DatetimeIndex or a 'date' column.")
    if "price_usd" not in frame.columns:
        raise DataLoadError("Parquet must contain a 'price_usd' column.")
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame["price_usd"] = pd.to_numeric(frame["price_usd"], errors="coerce")
    if "mvrv" in frame.columns:
        frame["mvrv"] = pd.to_numeric(frame["mvrv"], errors="coerce")
    return frame


@dataclass
class BTCDataProvider:
    """BTC-only provider using BRK-generated local parquet metrics."""

    parquet_path: str | None = None
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

        frame = _load_btc_from_parquet(_resolve_parquet_path(self.parquet_path))
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
