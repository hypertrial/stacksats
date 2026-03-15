"""Typed time-series output objects for strategy export runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
import pandas as pd

from .strategy_time_series_analysis import StrategyTimeSeriesAnalysisMixin
from .strategy_time_series_batch import TimeSeriesBatch
from .strategy_time_series_diagnostics import StrategyTimeSeriesDiagnosticsMixin
from .strategy_time_series_metadata import StrategySeriesMetadata
from .strategy_time_series_schema import (
    BRK_BTC_CSV_COLUMNS as _DEFAULT_BRK_BTC_CSV_COLUMNS,
    BRK_SOURCE_COLUMNS as _DEFAULT_BRK_SOURCE_COLUMNS,
    BRK_LINEAGE as _DEFAULT_BRK_LINEAGE,
    BRKLineageSpec,
    ColumnSpec,
    brk_lineage_markdown as render_brk_lineage_markdown,
    merge_schema_specs,
    render_schema_markdown,
    schema_dict as build_schema_dict,
    schema_specs as build_core_schema_specs,
    validate_brk_lineage_coverage as validate_lineage_coverage,
    validate_schema_specs,
)


@dataclass(frozen=True, slots=True, init=False)
class TimeSeries(StrategyTimeSeriesDiagnosticsMixin, StrategyTimeSeriesAnalysisMixin):
    """Single-window normalized strategy output time series."""

    metadata: StrategySeriesMetadata
    extra_schema: tuple[ColumnSpec, ...]
    _data: pd.DataFrame = field(repr=False)

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "weight", "price_usd")
    BRK_SOURCE_COLUMNS: ClassVar[tuple[str, ...]] = _DEFAULT_BRK_SOURCE_COLUMNS
    # Backward-compatible alias kept for existing callers.
    BRK_BTC_CSV_COLUMNS: ClassVar[tuple[str, ...]] = _DEFAULT_BRK_BTC_CSV_COLUMNS
    BRK_LINEAGE: ClassVar[tuple[BRKLineageSpec, ...]] = _DEFAULT_BRK_LINEAGE

    def __init__(
        self,
        *,
        metadata: StrategySeriesMetadata,
        data: pd.DataFrame,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("TimeSeries.data must be a pandas DataFrame.")

        normalized_extra_schema = validate_schema_specs(extra_schema, forbid_core_name_collisions=True)
        normalized_data = self._normalize_core_columns(data)

        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "extra_schema", normalized_extra_schema)
        object.__setattr__(self, "_data", normalized_data)
        self.validate()
        object.__setattr__(self, "_data", self._coerce_validated_columns(self._data))

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        *,
        metadata: StrategySeriesMetadata,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "TimeSeries":
        """Build a validated TimeSeries from a dataframe payload."""
        return cls(metadata=metadata, data=data, extra_schema=extra_schema)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        metadata: StrategySeriesMetadata,
        *,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "TimeSeries":
        """Load a TimeSeries from CSV."""
        csv_path = Path(path)
        return cls(
            metadata=metadata,
            data=pd.read_csv(csv_path),
            extra_schema=extra_schema,
        )

    @property
    def data(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized payload."""
        return self._data.copy(deep=True)

    @property
    def columns(self) -> tuple[str, ...]:
        """Return normalized column names."""
        return tuple(str(col) for col in self._data.columns)

    @property
    def row_count(self) -> int:
        """Return normalized row count."""
        return int(self._data.shape[0])

    def date_index(self) -> pd.DatetimeIndex:
        """Return the normalized date column as a DatetimeIndex."""
        return pd.DatetimeIndex(pd.to_datetime(self._data["date"], errors="coerce"))

    def window_key(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Return the metadata window key."""
        return (self.metadata.window_start, self.metadata.window_end)

    @classmethod
    def _core_schema_specs(cls) -> tuple[ColumnSpec, ...]:
        return build_core_schema_specs()

    @classmethod
    def _merged_schema_specs(
        cls,
        extra_schema: Iterable[ColumnSpec] = (),
    ) -> tuple[ColumnSpec, ...]:
        return merge_schema_specs(cls._core_schema_specs(), extra_schema)

    @staticmethod
    def _normalize_core_columns(data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy(deep=True)
        if "date" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
            normalized["date"] = normalized["date"].dt.normalize()
            normalized = normalized.sort_values("date").reset_index(drop=True)
        return normalized

    @staticmethod
    def _coerce_validated_columns(data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy(deep=True)
        if "weight" in normalized.columns:
            normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce")
        if "price_usd" in normalized.columns:
            normalized["price_usd"] = pd.to_numeric(normalized["price_usd"], errors="coerce")
        if "day_index" in normalized.columns:
            normalized["day_index"] = pd.to_numeric(normalized["day_index"], errors="coerce")
        return normalized

    def schema(self) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return self.schema_dict(self.extra_schema)

    @classmethod
    def schema_dict(
        cls,
        extra_schema: Iterable[ColumnSpec] = (),
    ) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return build_schema_dict(cls._merged_schema_specs(extra_schema))

    @classmethod
    def validate_brk_lineage_coverage(cls) -> None:
        """Ensure lineage mappings target documented core TimeSeries columns."""
        validate_lineage_coverage(
            lineage=cls.BRK_LINEAGE,
            schema_specs_iter=cls._core_schema_specs(),
            source_columns=cls.BRK_SOURCE_COLUMNS,
        )

    @classmethod
    def brk_lineage_markdown(cls) -> str:
        """Render BRK source-to-schema lineage as a markdown table."""
        cls.validate_brk_lineage_coverage()
        return render_brk_lineage_markdown(cls.BRK_LINEAGE)

    @classmethod
    def schema_markdown_table(cls, extra_schema: Iterable[ColumnSpec] = ()) -> str:
        """Render TimeSeries schema specs as a markdown table."""
        cls.validate_brk_lineage_coverage()
        return cls._render_schema_markdown(cls._merged_schema_specs(extra_schema))

    @staticmethod
    def _render_schema_markdown(specs: Iterable[ColumnSpec]) -> str:
        """Render schema specs as a markdown table."""
        return render_schema_markdown(specs)

    def schema_markdown(self) -> str:
        """Render schema specs as a markdown table."""
        return self._render_schema_markdown(self.schema().values())

    def validate_schema_coverage(self) -> None:
        """Ensure each column has an explicit handwritten schema entry."""
        covered = set(self.schema().keys())
        unknown = [col for col in self._data.columns if col not in covered]
        if unknown:
            raise ValueError(
                "Schema coverage missing for columns: " + ", ".join(str(col) for col in unknown)
            )

    def _validate_required_columns(self) -> None:
        schema_required = [spec.name for spec in self._merged_schema_specs(self.extra_schema) if spec.required]
        missing = [col for col in schema_required if col not in self._data.columns]
        if missing:
            raise ValueError(
                "TimeSeries missing required columns: "
                + ", ".join(str(col) for col in missing)
            )

    def _validate_date_contract(self, dates: pd.Series) -> None:
        if dates.isna().any():
            raise ValueError("Column 'date' must contain valid datetimes.")
        if dates.duplicated().any():
            raise ValueError("Column 'date' must not contain duplicates.")
        if not dates.is_monotonic_increasing:
            raise ValueError("Column 'date' must be sorted ascending.")

        start = self.metadata.window_start
        end = self.metadata.window_end
        if start is not None and len(dates) > 0 and pd.Timestamp(dates.iloc[0]) != pd.Timestamp(start):
            raise ValueError(
                "Series start date does not match metadata.window_start: "
                f"{dates.iloc[0]!s} != {pd.Timestamp(start)!s}"
            )
        if end is not None and len(dates) > 0 and pd.Timestamp(dates.iloc[-1]) != pd.Timestamp(end):
            raise ValueError(
                "Series end date does not match metadata.window_end: "
                f"{dates.iloc[-1]!s} != {pd.Timestamp(end)!s}"
            )
        if start is None or end is None:
            return

        expected = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        actual = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce"))
        if actual.equals(expected):
            return

        missing = expected.difference(actual)
        extra = actual.difference(expected)
        if len(missing) > 0 and len(extra) == 0:
            raise ValueError(
                "Column 'date' must exactly match the daily range from metadata.window_start "
                "to metadata.window_end; missing or skipped dates detected: "
                + ", ".join(ts.strftime("%Y-%m-%d") for ts in missing[:5])
            )
        if len(extra) > 0 and len(missing) == 0:
            raise ValueError(
                "Column 'date' must exactly match the daily range from metadata.window_start "
                "to metadata.window_end; unexpected dates detected: "
                + ", ".join(ts.strftime("%Y-%m-%d") for ts in extra[:5])
            )
        raise ValueError(
            "Column 'date' must exactly match the daily range from metadata.window_start "
            "to metadata.window_end."
        )

    def validate(self) -> None:
        """Validate data and metadata invariants."""
        self._validate_required_columns()
        self.validate_schema_coverage()

        dates = pd.to_datetime(self._data["date"], errors="coerce")
        self._validate_date_contract(dates)

        weights = pd.to_numeric(self._data["weight"], errors="coerce")
        if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError("Column 'weight' must contain finite numeric values.")
        if bool((weights < 0).any()):
            raise ValueError("Column 'weight' must not contain negative values.")
        if len(weights) > 0:
            weight_sum = float(weights.sum())
            if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
                raise ValueError("Column 'weight' must sum to 1.0 " f"(got {weight_sum:.10f}).")

        raw_price = self._data["price_usd"]
        prices = pd.to_numeric(raw_price, errors="coerce")
        invalid_non_null = raw_price.notna() & prices.isna()
        if invalid_non_null.any():
            raise ValueError("Column 'price_usd' must be numeric when present.")
        finite_mask = prices.notna()
        if finite_mask.any() and not np.isfinite(prices.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError("Column 'price_usd' must be finite when present.")

        if "locked" in self._data.columns:
            locked = self._data["locked"]
            valid_locked = locked.isin([True, False])
            if not bool(valid_locked.all()):
                raise ValueError("Column 'locked' must contain only boolean values.")

        if "day_index" in self._data.columns:
            day_index = pd.to_numeric(self._data["day_index"], errors="coerce")
            if day_index.isna().any():
                raise ValueError("Column 'day_index' must contain integer values.")
            if bool((day_index < 0).any()):
                raise ValueError("Column 'day_index' must be >= 0.")
            if len(day_index) > 0:
                expected = np.arange(len(day_index), dtype=float)
                if not np.array_equal(day_index.to_numpy(dtype=float), expected):
                    raise ValueError("Column 'day_index' must be contiguous starting at 0.")

        for spec in self._merged_schema_specs(self.extra_schema):
            if spec.name not in self._data.columns:
                continue
            if spec.dtype in {"int64", "float64"}:
                self._validate_optional_numeric_column(spec.name)
            elif spec.dtype.startswith("datetime64"):
                self._validate_optional_datetime_column(spec.name)

    def _validate_optional_numeric_column(self, column: str) -> None:
        """Validate optional numeric columns."""
        raw = self._data[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be numeric when present.")
        finite_mask = values.notna()
        if finite_mask.any() and not np.isfinite(values.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' must be finite when present.")

    def _validate_optional_datetime_column(self, column: str) -> None:
        """Validate optional datetime columns."""
        raw = self._data[column]
        values = pd.to_datetime(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be datetime when present.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized dataframe payload."""
        return self._data.copy(deep=True)

    def to_csv(self, path: str | Path, *, index: bool = False) -> None:
        """Write the normalized payload to CSV."""
        self.to_dataframe().to_csv(Path(path), index=index)

    @staticmethod
    def _native_float(value: float | np.floating | int | None) -> float | None:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out

    @staticmethod
    def _native_timestamp(value: pd.Timestamp | None) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

    @staticmethod
    def _series_numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
        non_null = values.dropna()
        if non_null.empty:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }
        return {
            "count": int(non_null.shape[0]),
            "mean": TimeSeries._native_float(non_null.mean()),
            "std": TimeSeries._native_float(non_null.std(ddof=0)),
            "min": TimeSeries._native_float(non_null.min()),
            "p25": TimeSeries._native_float(non_null.quantile(0.25)),
            "median": TimeSeries._native_float(non_null.median()),
            "p75": TimeSeries._native_float(non_null.quantile(0.75)),
            "max": TimeSeries._native_float(non_null.max()),
        }


# Deprecated aliases — remove in 0.9.0
StrategyTimeSeries = TimeSeries
StrategyTimeSeriesBatch = TimeSeriesBatch

__all__ = [
    "ColumnSpec",
    "BRKLineageSpec",
    "StrategySeriesMetadata",
    "TimeSeries",
    "TimeSeriesBatch",
    # Deprecated aliases
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
]
