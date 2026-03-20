"""Public EDA helpers for canonical merged_metrics parquet analysis."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Sequence

import polars as pl

from .data_setup import MANAGED_BRK_DIR, packaged_text

CANONICAL_COLUMNS = ("day_utc", "metric", "value")
PACKAGED_CATALOG_NAME = "brk_merged_metrics_catalog.json"
CATALOG_SEARCH_COLUMNS = (
    "metric",
    "display_label",
    "family",
    "access_category",
    "notes",
)


def _latest_canonical_parquet(brk_dir: Path = MANAGED_BRK_DIR) -> Path:
    directory = brk_dir.expanduser().resolve()
    matches = sorted(
        directory.glob("merged_metrics*.parquet"),
        key=lambda item: item.stat().st_mtime,
    )
    if matches:
        return matches[-1]
    raise FileNotFoundError(
        f"No canonical merged_metrics parquet found under {directory}."
    )


def _normalize_bound(
    value: str | dt.date | dt.datetime | None,
) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value).strip()[:10])


def _canonical_lazy_frame(parquet_path: Path) -> pl.LazyFrame:
    schema = pl.read_parquet_schema(parquet_path)
    names = tuple(schema)
    columns = set(names)
    if {"date", "price_usd"}.issubset(columns):
        raise ValueError(
            "stacksats.eda supports canonical merged_metrics parquet only; "
            "received a runtime parquet with columns like 'date' and 'price_usd'. "
            "Use the canonical long-format merged_metrics*.parquet from "
            "`stacksats data fetch`."
        )
    if columns != set(CANONICAL_COLUMNS):
        raise ValueError(
            "Unsupported parquet schema for stacksats.eda. Expected canonical "
            "merged_metrics columns ('day_utc', 'metric', 'value')."
        )

    frame = pl.scan_parquet(parquet_path)
    date_dtype = schema["day_utc"]
    if date_dtype == pl.Utf8:
        day_expr = pl.col("day_utc").str.to_date(strict=False)
    else:
        day_expr = pl.col("day_utc").cast(pl.Date, strict=False)

    return frame.select(
        day_expr.alias("day_utc"),
        pl.col("metric").cast(pl.Utf8, strict=False).alias("metric"),
        pl.col("value").cast(pl.Float64, strict=False).alias("value"),
    )


def _resolve_canonical_parquet(parquet_path: str | Path | None) -> Path:
    if parquet_path is not None:
        resolved = Path(parquet_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Canonical parquet not found: {resolved}")
        return resolved
    try:
        return _latest_canonical_parquet(MANAGED_BRK_DIR)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No canonical merged_metrics parquet could be resolved under "
            f"{MANAGED_BRK_DIR.expanduser().resolve()}. Run `stacksats data fetch` "
            "or pass an explicit parquet_path."
        ) from exc


def _validate_known_values(
    *,
    provided: Sequence[str] | None,
    known: Sequence[str],
    label: str,
) -> None:
    if not provided:
        return
    missing = sorted({item for item in provided if item not in set(known)})
    if missing:
        raise ValueError(f"Unknown {label}: {', '.join(missing)}")


def _normalize_catalog_frame(frame: pl.DataFrame) -> pl.DataFrame:
    string_columns = [
        "access_category",
        "cohort_scheme",
        "display_label",
        "entity_scope",
        "example_metric_group",
        "family",
        "metric",
        "notes",
        "semantic_class",
        "statistic",
        "transform",
        "unit",
        "window",
    ]
    existing_string_columns = [column for column in string_columns if column in frame.columns]
    return frame.with_columns(
        *[
            pl.col(column).cast(pl.Utf8, strict=False).alias(column)
            for column in existing_string_columns
        ],
        pl.col("coverage_rows").cast(pl.Int64, strict=False).alias("coverage_rows"),
        pl.col("first_day").str.to_date(strict=False).alias("first_day"),
        pl.col("last_day").str.to_date(strict=False).alias("last_day"),
    ).sort("metric")


@dataclass(frozen=True, slots=True)
class MetricCatalog:
    """Catalog wrapper for merged-metrics metadata."""

    _frame: pl.DataFrame

    def frame(self) -> pl.DataFrame:
        """Return a defensive copy of the catalog frame."""
        return self._frame.clone()

    def summary(self) -> dict[str, object]:
        """Return high-level catalog summary values."""
        frame = self._frame
        return {
            "metric_count": frame.height,
            "family_count": len(self.families()),
            "category_count": len(self.categories()),
            "first_day": frame["first_day"].drop_nulls().min(),
            "last_day": frame["last_day"].drop_nulls().max(),
        }

    def families(self) -> list[str]:
        """Return sorted unique family names."""
        values = self._frame["family"].drop_nulls().unique().sort()
        return values.to_list()

    def categories(self) -> list[str]:
        """Return sorted unique access categories."""
        values = self._frame["access_category"].drop_nulls().unique().sort()
        return values.to_list()

    def metrics(self) -> list[str]:
        """Return sorted unique metric names."""
        values = self._frame["metric"].drop_nulls().unique().sort()
        return values.to_list()

    def search(self, text: str) -> pl.DataFrame:
        """Search common catalog text fields with case-insensitive substring matching."""
        query = text.strip().lower()
        if not query:
            return self.frame()

        frame = self._frame.with_columns(
            pl.concat_str(
                [
                    pl.col(column).cast(pl.Utf8, strict=False).fill_null("")
                    for column in CATALOG_SEARCH_COLUMNS
                ],
                separator=" ",
            )
            .str.to_lowercase()
            .alias("_search_blob")
        )
        return (
            frame.filter(pl.col("_search_blob").str.contains(re.escape(query), literal=False))
            .drop("_search_blob")
            .sort("metric")
        )

    def filter(
        self,
        *,
        metrics: Sequence[str] | None = None,
        prefixes: Sequence[str] | None = None,
        regex: str | None = None,
        families: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Filter catalog rows using union semantics across selectors."""
        _validate_known_values(provided=metrics, known=self.metrics(), label="metrics")
        _validate_known_values(provided=families, known=self.families(), label="families")
        _validate_known_values(
            provided=categories,
            known=self.categories(),
            label="categories",
        )
        if not any((metrics, prefixes, regex, families, categories)):
            return self.frame()

        selected_metrics: set[str] = set()
        frame = self._frame
        if metrics:
            selected_metrics.update(metrics)
        if prefixes:
            for prefix in prefixes:
                selected_metrics.update(
                    frame.filter(pl.col("metric").str.starts_with(prefix))["metric"].to_list()
                )
        if regex:
            pattern = re.compile(regex)
            selected_metrics.update(
                metric
                for metric in frame["metric"].to_list()
                if pattern.search(metric)
            )
        if families:
            selected_metrics.update(
                frame.filter(pl.col("family").is_in(list(families)))["metric"].to_list()
            )
        if categories:
            selected_metrics.update(
                frame.filter(pl.col("access_category").is_in(list(categories)))["metric"].to_list()
            )
        if not selected_metrics:
            return frame.head(0)
        return frame.filter(pl.col("metric").is_in(sorted(selected_metrics))).sort("metric")

    def coverage(self, metrics: Sequence[str] | None = None) -> pl.DataFrame:
        """Return coverage metadata for the selected metrics."""
        _validate_known_values(provided=metrics, known=self.metrics(), label="metrics")
        frame = self._frame
        if metrics:
            frame = frame.filter(pl.col("metric").is_in(list(metrics)))
        return frame.select(
            "metric",
            "display_label",
            "family",
            "access_category",
            "coverage_rows",
            "first_day",
            "last_day",
        ).sort("metric")


@dataclass(frozen=True, slots=True)
class MergedMetricsDataset:
    """Lazy-first wrapper over canonical merged_metrics parquet data."""

    parquet_path: Path
    _lazyframe: pl.LazyFrame
    catalog: MetricCatalog

    def lazy(self) -> pl.LazyFrame:
        """Return the underlying lazy Polars frame."""
        return self._lazyframe

    def collect(self) -> pl.DataFrame:
        """Collect the current lazy dataset into an eager frame."""
        return self._lazyframe.collect()

    def summary(self) -> dict[str, object]:
        """Return high-level dataset summary values."""
        row = (
            self._lazyframe.select(
                pl.len().alias("row_count"),
                pl.col("day_utc").n_unique().alias("distinct_days"),
                pl.col("metric").n_unique().alias("distinct_metrics"),
                pl.col("day_utc").min().alias("first_day"),
                pl.col("day_utc").max().alias("last_day"),
            )
            .collect()
            .row(0, named=True)
        )
        return {
            "row_count": int(row["row_count"]),
            "distinct_days": int(row["distinct_days"]),
            "distinct_metrics": int(row["distinct_metrics"]),
            "first_day": None if row["first_day"] is None else str(row["first_day"]),
            "last_day": None if row["last_day"] is None else str(row["last_day"]),
            "parquet_path": str(self.parquet_path),
        }

    def filter_dates(
        self,
        start: str | dt.date | dt.datetime | None = None,
        end: str | dt.date | dt.datetime | None = None,
    ) -> MergedMetricsDataset:
        """Return a dataset filtered to the inclusive date window."""
        start_day = _normalize_bound(start)
        end_day = _normalize_bound(end)
        if start_day is None and end_day is None:
            return self
        if start_day is not None and end_day is not None and start_day > end_day:
            raise ValueError("start must be on or before end.")

        frame = self._lazyframe
        if start_day is not None:
            frame = frame.filter(pl.col("day_utc") >= pl.lit(start_day))
        if end_day is not None:
            frame = frame.filter(pl.col("day_utc") <= pl.lit(end_day))
        return MergedMetricsDataset(
            parquet_path=self.parquet_path,
            _lazyframe=frame,
            catalog=self.catalog,
        )

    def filter_metrics(
        self,
        *,
        metrics: Sequence[str] | None = None,
        prefixes: Sequence[str] | None = None,
        regex: str | None = None,
        families: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
    ) -> MergedMetricsDataset:
        """Return a dataset filtered by unioned metric selectors."""
        if not any((metrics, prefixes, regex, families, categories)):
            return self

        filtered_catalog = self.catalog.filter(
            metrics=metrics,
            prefixes=prefixes,
            regex=regex,
            families=families,
            categories=categories,
        )
        if filtered_catalog.is_empty():
            frame = self._lazyframe.filter(pl.lit(False))
        else:
            frame = self._lazyframe.filter(pl.col("metric").is_in(filtered_catalog["metric"].to_list()))
        return MergedMetricsDataset(
            parquet_path=self.parquet_path,
            _lazyframe=frame,
            catalog=self.catalog,
        )

    def metric_series(self, metric: str) -> pl.DataFrame:
        """Return one metric as a sorted long-format eager frame."""
        _validate_known_values(provided=[metric], known=self.catalog.metrics(), label="metric")
        return (
            self._lazyframe.filter(pl.col("metric") == metric)
            .sort("day_utc")
            .collect()
        )

    def metric_coverage(self, metrics: Sequence[str] | None = None) -> pl.DataFrame:
        """Return observed coverage within the current dataset window."""
        _validate_known_values(provided=metrics, known=self.catalog.metrics(), label="metrics")
        frame = self._lazyframe
        if metrics:
            frame = frame.filter(pl.col("metric").is_in(list(metrics)))
        return (
            frame.group_by("metric")
            .agg(
                pl.len().alias("coverage_rows"),
                pl.col("day_utc").min().alias("first_day"),
                pl.col("day_utc").max().alias("last_day"),
            )
            .sort("metric")
            .collect()
        )

    def pivot_wide(
        self,
        metrics: Sequence[str] | None = None,
        *,
        fill_null: float | None = None,
    ) -> pl.DataFrame:
        """Pivot the current dataset to a day-indexed wide frame."""
        _validate_known_values(provided=metrics, known=self.catalog.metrics(), label="metrics")
        frame = self._lazyframe
        if metrics:
            frame = frame.filter(pl.col("metric").is_in(list(metrics)))
        wide = (
            frame.select("day_utc", "metric", "value")
            .collect()
            .pivot(values="value", index="day_utc", on="metric")
            .sort("day_utc")
        )
        value_columns = sorted(column for column in wide.columns if column != "day_utc")
        if value_columns:
            wide = wide.select("day_utc", *value_columns)
        if fill_null is not None:
            if value_columns:
                wide = wide.with_columns(
                    [pl.col(column).fill_null(fill_null) for column in value_columns]
                )
        return wide


@lru_cache(maxsize=1)
def load_metric_catalog() -> MetricCatalog:
    """Load the packaged merged-metrics catalog."""

    payload = json.loads(packaged_text(PACKAGED_CATALOG_NAME))
    metrics = payload.get("metrics", [])
    frame = _normalize_catalog_frame(pl.DataFrame(metrics))
    return MetricCatalog(_frame=frame)


def open_merged_metrics(
    parquet_path: str | Path | None = None,
) -> MergedMetricsDataset:
    """Open the canonical merged_metrics parquet as a lazy-first dataset."""

    resolved = _resolve_canonical_parquet(parquet_path)
    return MergedMetricsDataset(
        parquet_path=resolved,
        _lazyframe=_canonical_lazy_frame(resolved),
        catalog=load_metric_catalog(),
    )
