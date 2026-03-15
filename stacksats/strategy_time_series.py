"""Typed time-series output objects for strategy export runs."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
import polars as pl

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

from .strategy_time_series_analysis import StrategyTimeSeriesAnalysisMixin
from .strategy_time_series_batch import WeightTimeSeriesBatch
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


def _to_naive_dt(value: dt.datetime | object) -> dt.datetime:
    """Normalize to naive datetime (midnight UTC) for comparison."""
    from .framework_contract import _to_naive_utc

    return _to_naive_utc(value)


@dataclass(frozen=True, slots=True, init=False)
class WeightTimeSeries(StrategyTimeSeriesDiagnosticsMixin, StrategyTimeSeriesAnalysisMixin):
    """Single-window validated strategy output (weights, prices, metadata)."""

    metadata: StrategySeriesMetadata
    extra_schema: tuple[ColumnSpec, ...]
    _data: pl.DataFrame = field(repr=False)

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "weight", "price_usd")
    BRK_SOURCE_COLUMNS: ClassVar[tuple[str, ...]] = _DEFAULT_BRK_SOURCE_COLUMNS
    BRK_BTC_CSV_COLUMNS: ClassVar[tuple[str, ...]] = _DEFAULT_BRK_BTC_CSV_COLUMNS
    BRK_LINEAGE: ClassVar[tuple[BRKLineageSpec, ...]] = _DEFAULT_BRK_LINEAGE

    def __init__(
        self,
        *,
        metadata: StrategySeriesMetadata,
        data: pl.DataFrame,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> None:
        if isinstance(data, pl.DataFrame):
            pass
        elif pd is not None and isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        else:
            raise TypeError("WeightTimeSeries.data must be a Polars or pandas DataFrame.")

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
        data: pl.DataFrame | "pd.DataFrame",
        *,
        metadata: StrategySeriesMetadata,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "WeightTimeSeries":
        """Build a validated WeightTimeSeries from a dataframe payload."""
        return cls(metadata=metadata, data=data, extra_schema=extra_schema)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        metadata: StrategySeriesMetadata,
        *,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "WeightTimeSeries":
        """Load a WeightTimeSeries from CSV."""
        csv_path = Path(path)
        return cls(
            metadata=metadata,
            data=pl.read_csv(csv_path),
            extra_schema=extra_schema,
        )

    @property
    def data(self) -> pl.DataFrame:
        """Return a defensive copy of the normalized payload."""
        return self._data.clone()

    @property
    def columns(self) -> tuple[str, ...]:
        """Return normalized column names."""
        return tuple(self._data.columns)

    @property
    def row_count(self) -> int:
        """Return normalized row count."""
        return self._data.height

    def date_index(self) -> list[dt.datetime]:
        """Return the normalized date column as a list of datetime."""
        return self._data["date"].to_list()

    def window_key(self) -> tuple[dt.datetime | None, dt.datetime | None]:
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
    def _normalize_core_columns(data: pl.DataFrame) -> pl.DataFrame:
        if "date" not in data.columns:
            return data
        if data["date"].dtype == pl.Utf8:
            date_expr = pl.col("date").str.to_datetime().dt.replace_time_zone(None).dt.truncate("1d")
        else:
            date_expr = pl.col("date").cast(pl.Datetime).dt.replace_time_zone(None).dt.truncate("1d")
        return data.with_columns(date_expr.alias("date")).sort("date")

    @staticmethod
    def _coerce_validated_columns(data: pl.DataFrame) -> pl.DataFrame:
        out = data
        if "weight" in out.columns:
            out = out.with_columns(pl.col("weight").cast(pl.Float64, strict=False))
        if "price_usd" in out.columns:
            out = out.with_columns(pl.col("price_usd").cast(pl.Float64, strict=False))
        if "day_index" in out.columns:
            out = out.with_columns(pl.col("day_index").cast(pl.Float64, strict=False))
        return out

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
                "WeightTimeSeries missing required columns: "
                + ", ".join(str(col) for col in missing)
            )

    def _validate_date_contract(self, dates: pl.Series) -> None:
        if dates.null_count() > 0:
            raise ValueError("Column 'date' must contain valid datetimes.")
        if dates.is_duplicated().sum() > 0:
            raise ValueError("Column 'date' must not contain duplicates.")
        if not dates.is_sorted():
            raise ValueError("Column 'date' must be sorted ascending.")

        start = self.metadata.window_start
        end = self.metadata.window_end
        first = dates[0] if dates.len() > 0 else None
        last = dates[-1] if dates.len() > 0 else None
        if start is not None and self._data.height > 0 and first is not None and _to_naive_dt(first) != _to_naive_dt(start):
            raise ValueError(
                "Series start date does not match metadata.window_start: "
                f"{first!s} != {start!s}"
            )
        if end is not None and self._data.height > 0 and last is not None and _to_naive_dt(last) != _to_naive_dt(end):
            raise ValueError(
                "Series end date does not match metadata.window_end: "
                f"{last!s} != {end!s}"
            )
        if start is None or end is None:
            return

        from .prelude import date_range_list

        expected = date_range_list(start, end)
        actual = dates.to_list()
        expected_set = set(_to_naive_dt(d) for d in expected)
        actual_set = set(_to_naive_dt(d) for d in actual)
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if len(missing) > 0 and len(extra) == 0:
            raise ValueError(
                "Column 'date' must exactly match the daily range from metadata.window_start "
                "to metadata.window_end; missing or skipped dates detected: "
                + ", ".join(d.strftime("%Y-%m-%d") for d in sorted(missing)[:5])
            )
        if len(extra) > 0 and len(missing) == 0:
            raise ValueError(
                "Column 'date' must exactly match the daily range from metadata.window_start "
                "to metadata.window_end; unexpected dates detected: "
                + ", ".join(d.strftime("%Y-%m-%d") for d in sorted(extra)[:5])
            )
        if missing or extra:
            raise ValueError(
                "Column 'date' must exactly match the daily range from metadata.window_start "
                "to metadata.window_end."
            )

    def validate(self) -> None:
        """Validate data and metadata invariants."""
        self._validate_required_columns()
        self.validate_schema_coverage()

        dates = self._data["date"]
        if dates.dtype == pl.Utf8:
            dates = self._data.with_columns(pl.col("date").str.to_datetime())["date"]
        self._validate_date_contract(dates)

        weights = self._data["weight"].cast(pl.Float64, strict=False)
        if weights.null_count() > 0 or (weights.is_nan() | weights.is_infinite()).any():
            raise ValueError("Column 'weight' must contain finite numeric values.")
        if (weights < 0).any():
            raise ValueError("Column 'weight' must not contain negative values.")
        if weights.len() > 0:
            weight_sum = float(weights.sum())
            if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
                raise ValueError("Column 'weight' must sum to 1.0 " f"(got {weight_sum:.10f}).")

        raw_price = self._data["price_usd"]
        prices = raw_price.cast(pl.Float64, strict=False)
        invalid_non_null = raw_price.is_not_null() & prices.is_null()
        if invalid_non_null.any():
            raise ValueError("Column 'price_usd' must be numeric when present.")
        if prices.is_finite().any() and (~prices.is_finite()).any():
            raise ValueError("Column 'price_usd' must be finite when present.")

        if "locked" in self._data.columns:
            locked = self._data["locked"]
            valid_locked = locked.is_in([True, False])
            if not valid_locked.all():
                raise ValueError("Column 'locked' must contain only boolean values.")

        if "day_index" in self._data.columns:
            day_index = self._data["day_index"].cast(pl.Float64, strict=False)
            if day_index.null_count() > 0:
                raise ValueError("Column 'day_index' must contain integer values.")
            if (day_index < 0).any():
                raise ValueError("Column 'day_index' must be >= 0.")
            if day_index.len() > 0:
                expected = np.arange(day_index.len(), dtype=float)
                if not np.array_equal(day_index.to_numpy(), expected):
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
        values = raw.cast(pl.Float64, strict=False)
        invalid_non_null = raw.is_not_null() & values.is_null()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be numeric when present.")
        if values.is_finite().any() and (~values.is_finite()).any():
            raise ValueError(f"Column '{column}' must be finite when present.")

    def _validate_optional_datetime_column(self, column: str) -> None:
        """Validate optional datetime columns."""
        raw = self._data[column]
        if raw.dtype == pl.Utf8:
            values = self._data.with_columns(pl.col(column).str.to_datetime(strict=False))[column]
        else:
            values = self._data.with_columns(pl.col(column).cast(pl.Datetime, strict=False))[column]
        invalid_non_null = raw.is_not_null() & values.is_null()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be datetime when present.")

    def to_dataframe(self) -> pl.DataFrame:
        """Return a copy of the normalized dataframe payload."""
        return self._data.clone()

    def to_csv(self, path: str | Path, *, index: bool = False) -> None:
        """Write the normalized payload to CSV."""
        self.to_dataframe().write_csv(Path(path), include_header=True)

    @staticmethod
    def _native_float(value: float | np.floating | int | None) -> float | None:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out

    @staticmethod
    def _native_timestamp(value: dt.datetime | object | None) -> str | None:
        if value is None:
            return None
        if pd is not None and hasattr(pd, "NaT") and value is pd.NaT:
            return None
        if hasattr(value, "is_null") and value.is_null():
            return None
        try:
            d = _to_naive_dt(value)
            return d.isoformat()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _series_numeric_summary(values: pl.Series | "pd.Series") -> dict[str, float | int | None]:
        if pd is not None and isinstance(values, pd.Series):
            values = pl.from_pandas(values)
        non_null = values.drop_nulls()
        if non_null.len() == 0:
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
        arr = non_null.to_numpy()
        return {
            "count": int(non_null.len()),
            "mean": WeightTimeSeries._native_float(float(np.mean(arr))),
            "std": WeightTimeSeries._native_float(float(np.std(arr))),
            "min": WeightTimeSeries._native_float(float(np.min(arr))),
            "p25": WeightTimeSeries._native_float(float(np.percentile(arr, 25))),
            "median": WeightTimeSeries._native_float(float(np.median(arr))),
            "p75": WeightTimeSeries._native_float(float(np.percentile(arr, 75))),
            "max": WeightTimeSeries._native_float(float(np.max(arr))),
        }


# Deprecated aliases — remove in 0.9.0
TimeSeries = WeightTimeSeries
TimeSeriesBatch = WeightTimeSeriesBatch
StrategyTimeSeries = WeightTimeSeries
StrategyTimeSeriesBatch = WeightTimeSeriesBatch

__all__ = [
    "ColumnSpec",
    "BRKLineageSpec",
    "StrategySeriesMetadata",
    "WeightTimeSeries",
    "WeightTimeSeriesBatch",
    "TimeSeries",
    "TimeSeriesBatch",
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
]
