---
title: EDA Quickstart
description: Explore the canonical Bitcoin Research Kit (BRK) merged_metrics parquet with the stable top-level EDA helpers.
---

# EDA Quickstart

Use the top-level `open_merged_metrics()` and `load_metric_catalog()` helpers when you want notebook/script-style exploration of the canonical long-format Bitcoin Research Kit (BRK) `merged_metrics*.parquet` dataset.

StackSats supports BRK at the project and data-workflow level, but these helpers are part of the StackSats Python API. Use [BRK Data Source](../data-source.md) for the canonical upstream links and support boundary.

## Prerequisite

Fetch the canonical source parquet first:

```bash
stacksats data fetch
```

These helpers do not auto-download data and do not read the reduced runtime `bitcoin_analytics.parquet`.

## Open the canonical parquet

```python
from stacksats import open_merged_metrics

dataset = open_merged_metrics()
print(dataset.summary())
print(dataset.head(5))
print(dataset.sample(5, seed=0))
```

By default, `open_merged_metrics()` resolves the newest fetched canonical parquet under `~/.stacksats/data/brk/`. You can also pass an explicit path.

Use `head(...)` and `sample(...)` before larger collects or pivots so you can inspect the current slice safely.

## Discover metrics with the catalog

```python
from stacksats import load_metric_catalog

catalog = load_metric_catalog()
print(catalog.summary())
print(catalog.search("sopr").select("metric", "family", "access_category").head(5))
print(catalog.suggest_metrics("mvr"))
print(catalog.describe_metric("adjusted_sopr"))
```

Use the catalog to discover metric names, families, access categories, and coverage metadata before narrowing the dataset.

Global catalog metadata tells you which metrics exist in the canonical dataset overall. It does not guarantee those metrics are present in your current filtered slice.

## Filter by date, metric, family, or category

```python
filtered = (
    dataset
    .filter_dates(start="2020-01-01", end="2024-12-31")
    .filter_metrics(
        metrics=["market_cap"],
        prefixes=["adjusted_"],
        families=["mvrv"],
        categories=["Supply, issuance, and scarcity"],
    )
)

print(filtered.summary())
print(filtered.metric_coverage())
print(filtered.available_metrics())
print(filtered.metric_counts())
```

Selector filters use union semantics across the provided selectors.

## Bridge search directly into data filtering

```python
searched = (
    dataset
    .filter_search("sopr")
    .filter_dates(start="2020-01-01", end="2024-12-31")
)

print(searched.available_metrics())
print(searched.metric_series("adjusted_sopr", error_if_empty=True).head())
```

`filter_search(...)` uses the same broad text matching as the catalog search. If a metric exists globally but is absent from your current filtered slice, `metric_series(..., error_if_empty=True)` raises a clear error instead of returning an empty frame silently.

## Pivot wide for downstream analysis

```python
wide = dataset.pivot_wide(
    metrics=["market_cap", "mvrv", "adjusted_sopr"],
    fill_null=0.0,
)
print(wide.head())
```

Long-format methods preserve canonical columns (`day_utc`, `metric`, `value`). `pivot_wide(...)` keeps `day_utc` as the date column.

## Related pages

- [Full Data Setup](full-data-setup.md)
- [BRK Data Source](../data-source.md)
- [Merged Metrics Data Guide](../reference/merged-metrics-data-guide.md)
- [API Reference](../reference/api/eda.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+EDA+Quickstart)
