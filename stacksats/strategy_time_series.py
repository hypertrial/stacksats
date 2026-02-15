"""Typed time-series output objects for strategy export runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Iterable

import numpy as np
import pandas as pd


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Handwritten schema specification for a single column."""

    name: str
    dtype: str
    required: bool
    description: str
    unit: str | None = None
    constraints: tuple[str, ...] = ()
    source: str = "framework"
    formula: str | None = None


@dataclass(frozen=True, slots=True)
class StrategySeriesMetadata:
    """Provenance and window metadata for a single strategy time series."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    schema_version: str = "1.0.0"
    generated_at: pd.Timestamp = field(default_factory=_utc_now)
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None


@dataclass(frozen=True, slots=True)
class StrategyTimeSeries:
    """Single-window normalized strategy output time series."""

    metadata: StrategySeriesMetadata
    data: pd.DataFrame

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "weight", "price_usd")

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("StrategyTimeSeries.data must be a pandas DataFrame.")

        normalized = self.data.copy(deep=True)
        if "date" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
            normalized = normalized.sort_values("date").reset_index(drop=True)
        if "weight" in normalized.columns:
            normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce")
        if "price_usd" in normalized.columns:
            normalized["price_usd"] = pd.to_numeric(normalized["price_usd"], errors="coerce")
        if "day_index" in normalized.columns:
            normalized["day_index"] = pd.to_numeric(normalized["day_index"], errors="coerce")

        object.__setattr__(self, "data", normalized)
        self.validate()

    @staticmethod
    def _schema_specs() -> tuple[ColumnSpec, ...]:
        return (
            ColumnSpec(
                name="day_index",
                dtype="int64",
                required=False,
                description="Zero-based day index within the allocation window.",
                constraints=(">=0", "strictly increasing by 1"),
            ),
            ColumnSpec(
                name="date",
                dtype="datetime64[ns]",
                required=True,
                description="Calendar day for this allocation row.",
                constraints=("unique", "sorted ascending", "daily grain"),
            ),
            ColumnSpec(
                name="weight",
                dtype="float64",
                required=True,
                description=(
                    "Final feasible daily allocation after clipping, lock preservation, "
                    "and remaining-budget constraints."
                ),
                constraints=("finite", ">=0", "sum ~= 1.0"),
            ),
            ColumnSpec(
                name="price_usd",
                dtype="float64",
                required=True,
                description="BTC price in USD for the given date when available.",
                unit="USD",
                constraints=("finite when present", "nullable for future dates"),
            ),
            ColumnSpec(
                name="locked",
                dtype="bool",
                required=False,
                description="True when a row belongs to an immutable locked history prefix.",
                constraints=("boolean values only",),
            ),
        )

    def schema(self) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return {spec.name: spec for spec in self._schema_specs()}

    def schema_markdown(self) -> str:
        """Render schema specs as a markdown table."""
        specs = self._schema_specs()
        header = (
            "| name | dtype | required | description | unit | constraints | source | formula |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        rows = [
            "| {name} | {dtype} | {required} | {description} | {unit} | {constraints} | {source} | {formula} |".format(
                name=spec.name,
                dtype=spec.dtype,
                required=str(spec.required),
                description=spec.description.replace("|", "\\|"),
                unit=(spec.unit or ""),
                constraints=", ".join(spec.constraints),
                source=spec.source,
                formula=(spec.formula or ""),
            )
            for spec in specs
        ]
        return "\n".join([header, *rows])

    def validate_schema_coverage(self) -> None:
        """Ensure each column has an explicit handwritten schema entry."""
        covered = set(self.schema().keys())
        unknown = [col for col in self.data.columns if col not in covered]
        if unknown:
            raise ValueError(
                "Schema coverage missing for columns: " + ", ".join(str(col) for col in unknown)
            )

    def validate(self) -> None:
        """Validate data and metadata invariants."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            raise ValueError(
                "StrategyTimeSeries missing required columns: "
                + ", ".join(str(col) for col in missing)
            )

        self.validate_schema_coverage()

        dates = pd.to_datetime(self.data["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Column 'date' must contain valid datetimes.")
        if dates.duplicated().any():
            raise ValueError("Column 'date' must not contain duplicates.")
        if not dates.is_monotonic_increasing:
            raise ValueError("Column 'date' must be sorted ascending.")

        if self.metadata.window_start is not None and len(dates) > 0:
            start = pd.Timestamp(self.metadata.window_start)
            if pd.Timestamp(dates.iloc[0]) != start:
                raise ValueError(
                    "Series start date does not match metadata.window_start: "
                    f"{dates.iloc[0]!s} != {start!s}"
                )
        if self.metadata.window_end is not None and len(dates) > 0:
            end = pd.Timestamp(self.metadata.window_end)
            if pd.Timestamp(dates.iloc[-1]) != end:
                raise ValueError(
                    "Series end date does not match metadata.window_end: "
                    f"{dates.iloc[-1]!s} != {end!s}"
                )

        weights = pd.to_numeric(self.data["weight"], errors="coerce")
        if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError("Column 'weight' must contain finite numeric values.")
        if bool((weights < 0).any()):
            raise ValueError("Column 'weight' must not contain negative values.")
        if len(weights) > 0:
            weight_sum = float(weights.sum())
            if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
                raise ValueError(
                    "Column 'weight' must sum to 1.0 "
                    f"(got {weight_sum:.10f})."
                )

        raw_price = self.data["price_usd"]
        prices = pd.to_numeric(raw_price, errors="coerce")
        invalid_non_null = raw_price.notna() & prices.isna()
        if invalid_non_null.any():
            raise ValueError("Column 'price_usd' must be numeric when present.")
        finite_mask = prices.notna()
        if finite_mask.any() and not np.isfinite(prices.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError("Column 'price_usd' must be finite when present.")

        if "locked" in self.data.columns:
            locked = self.data["locked"]
            valid_locked = locked.isin([True, False])
            if not bool(valid_locked.all()):
                raise ValueError("Column 'locked' must contain only boolean values.")

        if "day_index" in self.data.columns:
            day_index = pd.to_numeric(self.data["day_index"], errors="coerce")
            if day_index.isna().any():
                raise ValueError("Column 'day_index' must contain integer values.")
            if bool((day_index < 0).any()):
                raise ValueError("Column 'day_index' must be >= 0.")
            if len(day_index) > 0:
                expected = np.arange(len(day_index), dtype=float)
                if not np.array_equal(day_index.to_numpy(dtype=float), expected):
                    raise ValueError("Column 'day_index' must be contiguous starting at 0.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized dataframe payload."""
        return self.data.copy(deep=True)


@dataclass(frozen=True, slots=True)
class StrategyTimeSeriesBatch:
    """Collection of single-window strategy time-series outputs."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    windows: tuple[StrategyTimeSeries, ...]
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
            payload_columns = [col for col in ("day_index", "date", "weight", "price_usd", "locked") if col in window_frame.columns]
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

    def iter_windows(self) -> Iterable[StrategyTimeSeries]:
        """Yield windows in batch order."""
        return iter(self.windows)

    def for_window(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> StrategyTimeSeries:
        """Return the window object for a specific date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        for window in self.windows:
            md = window.metadata
            if pd.Timestamp(md.window_start) == start and pd.Timestamp(md.window_end) == end:
                return window
        raise KeyError(f"Window not found: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")
