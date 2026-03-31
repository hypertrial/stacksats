"""Compatibility façade for split EDA helper modules."""

from . import (
    CANONICAL_COLUMNS,
    CATALOG_SEARCH_COLUMNS,
    METRIC_DESCRIPTION_FIELDS,
    PACKAGED_CATALOG_NAME,
    MergedMetricsDataset,
    MetricCatalog,
    _canonical_lazy_frame,
    _latest_canonical_parquet,
    _normalize_bound,
    _resolve_canonical_parquet,
    load_metric_catalog,
    open_merged_metrics,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "CATALOG_SEARCH_COLUMNS",
    "METRIC_DESCRIPTION_FIELDS",
    "PACKAGED_CATALOG_NAME",
    "MergedMetricsDataset",
    "MetricCatalog",
    "_canonical_lazy_frame",
    "_latest_canonical_parquet",
    "_normalize_bound",
    "_resolve_canonical_parquet",
    "load_metric_catalog",
    "open_merged_metrics",
]
