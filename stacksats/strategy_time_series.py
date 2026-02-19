"""Typed time-series output objects for strategy export runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

import numpy as np
import pandas as pd

from .strategy_time_series_analysis import StrategyTimeSeriesAnalysisMixin
from .strategy_time_series_batch import StrategyTimeSeriesBatch
from .strategy_time_series_diagnostics import StrategyTimeSeriesDiagnosticsMixin
from .strategy_time_series_metadata import StrategySeriesMetadata
from .strategy_time_series_schema import (
    COINMETRICS_BTC_CSV_COLUMNS as _DEFAULT_COINMETRICS_BTC_CSV_COLUMNS,
    COINMETRICS_LINEAGE as _DEFAULT_COINMETRICS_LINEAGE,
    CoinMetricsLineageSpec,
    ColumnSpec,
    coinmetrics_lineage_markdown as render_coinmetrics_lineage_markdown,
    render_schema_markdown,
    schema_dict as build_schema_dict,
    schema_specs as build_schema_specs,
    validate_coinmetrics_lineage_coverage as validate_lineage_coverage,
)


@dataclass(frozen=True, slots=True)
class StrategyTimeSeries(StrategyTimeSeriesDiagnosticsMixin, StrategyTimeSeriesAnalysisMixin):
    """Single-window normalized strategy output time series."""

    metadata: StrategySeriesMetadata
    data: pd.DataFrame

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "weight", "price_usd")
    COINMETRICS_BTC_CSV_COLUMNS: ClassVar[tuple[str, ...]] = _DEFAULT_COINMETRICS_BTC_CSV_COLUMNS
    COINMETRICS_LINEAGE: ClassVar[tuple[CoinMetricsLineageSpec, ...]] = _DEFAULT_COINMETRICS_LINEAGE

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
        return build_schema_specs()

    def schema(self) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return self.schema_dict()

    @classmethod
    def schema_dict(cls) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return build_schema_dict(cls._schema_specs())

    @classmethod
    def validate_coinmetrics_lineage_coverage(cls) -> None:
        """Ensure lineage mappings target documented StrategyTimeSeries columns."""
        validate_lineage_coverage(
            lineage=cls.COINMETRICS_LINEAGE,
            schema_specs_iter=cls._schema_specs(),
            source_columns=cls.COINMETRICS_BTC_CSV_COLUMNS,
        )

    @classmethod
    def coinmetrics_lineage_markdown(cls) -> str:
        """Render CoinMetrics source-to-schema lineage as a markdown table."""
        cls.validate_coinmetrics_lineage_coverage()
        return render_coinmetrics_lineage_markdown(cls.COINMETRICS_LINEAGE)

    @classmethod
    def schema_markdown_table(cls) -> str:
        """Render StrategyTimeSeries schema specs as a markdown table."""
        cls.validate_coinmetrics_lineage_coverage()
        return cls._render_schema_markdown(cls._schema_specs())

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
                raise ValueError("Column 'weight' must sum to 1.0 " f"(got {weight_sum:.10f}).")

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

        for spec in self._schema_specs():
            if spec.source != "coinmetrics":
                continue
            if spec.name not in self.data.columns:
                continue
            if spec.dtype in {"int64", "float64"}:
                self._validate_optional_numeric_column(spec.name)
            elif spec.dtype.startswith("datetime64"):
                self._validate_optional_datetime_column(spec.name)

    def _validate_optional_numeric_column(self, column: str) -> None:
        """Validate optional CoinMetrics passthrough numeric columns."""
        raw = self.data[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be numeric when present.")
        finite_mask = values.notna()
        if finite_mask.any() and not np.isfinite(values.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' must be finite when present.")

    def _validate_optional_datetime_column(self, column: str) -> None:
        """Validate optional CoinMetrics passthrough datetime columns."""
        raw = self.data[column]
        values = pd.to_datetime(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be datetime when present.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized dataframe payload."""
        return self.data.copy(deep=True)

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
            "mean": StrategyTimeSeries._native_float(non_null.mean()),
            "std": StrategyTimeSeries._native_float(non_null.std(ddof=0)),
            "min": StrategyTimeSeries._native_float(non_null.min()),
            "p25": StrategyTimeSeries._native_float(non_null.quantile(0.25)),
            "median": StrategyTimeSeries._native_float(non_null.median()),
            "p75": StrategyTimeSeries._native_float(non_null.quantile(0.75)),
            "max": StrategyTimeSeries._native_float(non_null.max()),
        }


__all__ = [
    "ColumnSpec",
    "CoinMetricsLineageSpec",
    "StrategySeriesMetadata",
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
]
