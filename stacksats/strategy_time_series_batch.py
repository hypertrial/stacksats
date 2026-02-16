"""Batch container for strategy time-series outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .strategy_time_series_metadata import StrategySeriesMetadata, _utc_now

if TYPE_CHECKING:
    from .strategy_time_series import StrategyTimeSeries


@dataclass(frozen=True, slots=True)
class StrategyTimeSeriesBatch:
    """Collection of single-window strategy time-series outputs."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    windows: tuple["StrategyTimeSeries", ...]
    schema_version: str = "1.0.0"
    generated_at: pd.Timestamp = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if len(self.windows) == 0:
            raise ValueError("StrategyTimeSeriesBatch.windows must not be empty.")
        self.validate()

    @classmethod
    def from_flat_dataframe(
        cls,
        data: pd.DataFrame,
        *,
        strategy_id: str,
        strategy_version: str,
        run_id: str,
        config_hash: str,
        schema_version: str = "1.0.0",
    ) -> "StrategyTimeSeriesBatch":
        """Build a batch object from a flattened export dataframe."""
        from .strategy_time_series import StrategyTimeSeries

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        required = {"start_date", "end_date", "date", "weight", "price_usd"}
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(
                "Flat dataframe missing required columns: " + ", ".join(sorted(missing))
            )

        normalized = data.copy(deep=True)
        normalized["start_date"] = pd.to_datetime(normalized["start_date"], errors="coerce")
        normalized["end_date"] = pd.to_datetime(normalized["end_date"], errors="coerce")
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        if (
            normalized["start_date"].isna().any()
            or normalized["end_date"].isna().any()
            or normalized["date"].isna().any()
        ):
            raise ValueError("start_date, end_date, and date must be valid datetimes.")

        normalized = normalized.sort_values(["start_date", "end_date", "date"]).reset_index(
            drop=True
        )

        windows: list[StrategyTimeSeries] = []
        grouped = normalized.groupby(["start_date", "end_date"], sort=True, dropna=False)
        for (window_start, window_end), window_frame in grouped:
            payload_columns = [col for col in window_frame.columns if col not in {"start_date", "end_date"}]
            payload = window_frame[payload_columns].reset_index(drop=True)
            if "day_index" not in payload.columns:
                payload.insert(0, "day_index", np.arange(len(payload), dtype=int))
            metadata = StrategySeriesMetadata(
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                run_id=run_id,
                config_hash=config_hash,
                schema_version=schema_version,
                window_start=pd.Timestamp(window_start),
                window_end=pd.Timestamp(window_end),
            )
            windows.append(StrategyTimeSeries(metadata=metadata, data=payload))

        return cls(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_id=run_id,
            config_hash=config_hash,
            windows=tuple(windows),
            schema_version=schema_version,
        )

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def row_count(self) -> int:
        return int(sum(len(window.data) for window in self.windows))

    def validate(self) -> None:
        """Validate cross-window metadata and uniqueness invariants."""
        seen_keys: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
        for window in self.windows:
            window.validate()
            md = window.metadata
            if md.strategy_id != self.strategy_id:
                raise ValueError("Window metadata strategy_id does not match batch strategy_id.")
            if md.strategy_version != self.strategy_version:
                raise ValueError(
                    "Window metadata strategy_version does not match batch strategy_version."
                )
            if md.run_id != self.run_id:
                raise ValueError("Window metadata run_id does not match batch run_id.")
            if md.config_hash != self.config_hash:
                raise ValueError("Window metadata config_hash does not match batch config_hash.")
            if md.schema_version != self.schema_version:
                raise ValueError("Window metadata schema_version does not match batch schema_version.")
            if md.window_start is None or md.window_end is None:
                raise ValueError("Each window must define metadata.window_start and metadata.window_end.")
            key = (pd.Timestamp(md.window_start), pd.Timestamp(md.window_end))
            if key in seen_keys:
                raise ValueError(
                    "Duplicate window key detected in batch: "
                    f"{key[0].strftime('%Y-%m-%d')} -> {key[1].strftime('%Y-%m-%d')}"
                )
            seen_keys.add(key)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the batch into one canonical dataframe."""
        frames: list[pd.DataFrame] = []
        for window in self.windows:
            md = window.metadata
            payload = window.to_dataframe()
            payload.insert(0, "end_date", pd.Timestamp(md.window_end))
            payload.insert(0, "start_date", pd.Timestamp(md.window_start))
            frames.append(payload)
        return pd.concat(frames, ignore_index=True)

    def schema_markdown(self) -> str:
        """Render the shared window schema as markdown."""
        return self.windows[0].schema_markdown()

    def iter_windows(self) -> Iterable["StrategyTimeSeries"]:
        """Yield windows in batch order."""
        return iter(self.windows)

    def for_window(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> "StrategyTimeSeries":
        """Return the window object for a specific date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        for window in self.windows:
            md = window.metadata
            if pd.Timestamp(md.window_start) == start and pd.Timestamp(md.window_end) == end:
                return window
        raise KeyError(f"Window not found: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")
