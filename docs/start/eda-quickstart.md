---
title: EDA Quickstart
description: Explore the canonical merged_metrics parquet with the public stacksats.eda API.
---

# EDA Quickstart

Use `stacksats.eda` when you want notebook/script-style exploration of the canonical long-format `merged_metrics*.parquet` dataset.

## Prerequisite

Fetch the canonical source parquet first:

```bash
stacksats data fetch
```

`stacksats.eda` does not auto-download data and does not read the reduced runtime `bitcoin_analytics.parquet`.

## Open the canonical parquet

```python
from stacksats import open_merged_metrics

dataset = open_merged_metrics()
print(dataset.summary())
```

By default, `open_merged_metrics()` resolves the newest fetched canonical parquet under `~/.stacksats/data/brk/`. You can also pass an explicit path.

## Search the metric catalog

```python
from stacksats import load_metric_catalog

catalog = load_metric_catalog()
print(catalog.summary())
print(catalog.search("sopr").select("metric", "family", "access_category").head(5))
```

Use the catalog to discover metric names, families, access categories, and coverage metadata before narrowing the dataset.

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
```

Selector filters use union semantics across the provided selectors.

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
