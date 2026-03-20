"""Public EDA helpers for canonical merged_metrics parquet analysis."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from functools import lru_cache
import json
from pathlib import Path
import random
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
METRIC_DESCRIPTION_FIELDS = (
    "metric",
    "display_label",
    "family",
    "access_category",
    "coverage_rows",
    "first_day",
    "last_day",
    "semantic_class",
    "transform",
    "unit",
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
    known_set = set(known)
    missing = sorted({item for item in provided if item not in known_set})
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


def _search_blob_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.col(column).cast(pl.Utf8, strict=False).fill_null("")
            for column in CATALOG_SEARCH_COLUMNS
        ],
        separator=" ",
    ).str.to_lowercase()


def _known_metric_row(catalog: "MetricCatalog", metric: str) -> dict[str, object]:
    _validate_known_values(provided=[metric], known=catalog.metrics(), label="metric")
    return catalog.frame().filter(pl.col("metric") == metric).row(0, named=True)


def _available_metric_rows(frame: pl.LazyFrame) -> pl.DataFrame:
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


def _ranked_suggestion_matches(frame: pl.DataFrame, query: str) -> pl.DataFrame:
    normalized = query.strip().lower()
    if not normalized:
        return frame.head(0)
    escaped = re.escape(normalized)
    lower_metric = pl.col("metric").str.to_lowercase()
    lower_label = pl.col("display_label").str.to_lowercase()
    search_blob = _search_blob_expr()
    ranked = frame.with_columns(
        pl.when(lower_metric == normalized)
        .then(pl.lit(0))
        .when(lower_label == normalized)
        .then(pl.lit(1))
        .when(lower_metric.str.starts_with(normalized))
        .then(pl.lit(2))
        .when(lower_label.str.starts_with(normalized))
        .then(pl.lit(3))
        .when(search_blob.str.contains(escaped, literal=False))
        .then(pl.lit(4))
        .otherwise(pl.lit(99))
        .alias("_rank")
    ).filter(pl.col("_rank") < 99)
    return ranked.sort(
        by=["_rank", "coverage_rows", "metric"],
        descending=[False, True, False],
    )


def _with_row_index(frame: pl.LazyFrame, name: str) -> pl.LazyFrame:
    if hasattr(frame, "with_row_index"):
        return frame.with_row_index(name)
    return frame.with_row_count(name)


def _sample_row_indexes(total_rows: int, sample_size: int, seed: int | None) -> list[int]:
    if sample_size <= 0 or total_rows <= 0:
        return []
    if sample_size >= total_rows:
        return list(range(total_rows))
    return sorted(random.Random(seed).sample(range(total_rows), sample_size))


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

        frame = self._frame.with_columns(_search_blob_expr().alias("_search_blob"))
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

    def describe_metric(self, metric: str) -> dict[str, object]:
        """Return one metric's metadata as a structured dictionary."""
        row = _known_metric_row(self, metric)
        return {field: row[field] for field in METRIC_DESCRIPTION_FIELDS}

    def suggest_metrics(self, query: str, *, limit: int = 10) -> pl.DataFrame:
        """Return ranked metric suggestions for a likely intended query."""
        if limit <= 0:
            return self._frame.head(0)
        return _ranked_suggestion_matches(self._frame, query).drop("_rank").head(limit)


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

    def _with_lazyframe(self, frame: pl.LazyFrame) -> MergedMetricsDataset:
        return MergedMetricsDataset(
            parquet_path=self.parquet_path,
            _lazyframe=frame,
            catalog=self.catalog,
        )

    def _filter_to_metrics(self, metrics: Sequence[str]) -> MergedMetricsDataset:
        if not metrics:
            return self._with_lazyframe(self._lazyframe.filter(pl.lit(False)))
        return self._with_lazyframe(self._lazyframe.filter(pl.col("metric").is_in(list(metrics))))

    def head(self, n: int = 10) -> pl.DataFrame:
        """Collect the first ``n`` rows from the current filtered dataset."""
        return self._lazyframe.limit(max(int(n), 0)).collect()

    def sample(self, n: int = 10, *, seed: int | None = 0) -> pl.DataFrame:
        """Collect a small sample from the current filtered dataset.

        This samples by row index so only the selected rows are materialized.
        """
        size = max(int(n), 0)
        if size == 0:
            return self.head(0)
        total_rows = int(self._lazyframe.select(pl.len()).collect().item())
        row_indexes = _sample_row_indexes(total_rows, size, seed)
        if not row_indexes:
            return self.head(0)
        if len(row_indexes) == total_rows:
            return self.collect()
        return (
            _with_row_index(self._lazyframe, "__sample_row")
            .filter(pl.col("__sample_row").is_in(row_indexes))
            .sort("__sample_row")
            .drop("__sample_row")
            .collect()
        )

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
        return self._with_lazyframe(frame)

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
        return self._filter_to_metrics(filtered_catalog["metric"].to_list())

    def filter_search(self, query: str) -> MergedMetricsDataset:
        """Filter the dataset using catalog text search."""
        if not query.strip():
            return self
        matched = self.catalog.search(query)
        return self._filter_to_metrics(matched["metric"].to_list())

    def available_metrics(self) -> list[str]:
        """Return sorted metric names present in the current filtered dataset."""
        return _available_metric_rows(self._lazyframe)["metric"].to_list()

    def metric_counts(self) -> pl.DataFrame:
        """Return per-metric row counts for the current filtered dataset."""
        return _available_metric_rows(self._lazyframe).sort(
            by=["coverage_rows", "metric"],
            descending=[True, False],
        )

    def metric_series(self, metric: str, *, error_if_empty: bool = False) -> pl.DataFrame:
        """Return one metric as a sorted long-format eager frame."""
        _known_metric_row(self.catalog, metric)
        frame = (
            self._lazyframe.filter(pl.col("metric") == metric)
            .sort("day_utc")
            .collect()
        )
        if error_if_empty and frame.is_empty():
            raise ValueError(
                "Current filtered dataset has no rows for metric "
                f"{metric!r}. Adjust your filters or set error_if_empty=False."
            )
        return frame

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
