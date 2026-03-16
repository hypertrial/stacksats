---
title: Merged Metrics Parquet Schema
description: Canonical schema for the Google Drive distributed merged_metrics parquet dataset.
---

# Merged Metrics Parquet Schema

This page defines the canonical StackSats dataset contract for the Google Drive
`merged_metrics*.parquet` file.

Use [Merged Metrics Taxonomy](merged-metrics-taxonomy.md) for the semantic
grouping model of the full `metric` namespace.

Canonical file link:

- <https://drive.google.com/file/d/1jKRRU7l9kOMdGI_hIJGg02X3jWTMPJsw/view?usp=sharing>

Profiled file in this repository:

- `merged_metrics_2026-03-15_04-29-57.parquet`

## Physical schema

| Column | Polars dtype | Required | Nulls | Meaning |
| --- | --- | --- | --- | --- |
| `day_utc` | `Date` | yes | no | UTC calendar day for the metric observation. |
| `metric` | `String` | yes | no | Metric key/name (long-format keyspace). |
| `value` | `Float64` | yes | no | Numeric metric value for `(day_utc, metric)`. |

## Dataset profile (current canonical snapshot)

These values were computed directly from
`merged_metrics_2026-03-15_04-29-57.parquet`:

| Property | Value |
| --- | --- |
| Total rows | `236,259,020` |
| Distinct days | `6,274` |
| Distinct metrics | `41,407` |
| Day range | `2009-01-03` to `2026-03-13` |
| Null counts | `day_utc=0`, `metric=0`, `value=0` |
| Finite check | all `value` entries are finite in this snapshot |

## Runtime projection contract

StackSats runtime entrypoints (`load_data`, `StrategyRunner`, CLI strategy lifecycle
commands) consume a normalized BRK-wide parquet with canonical columns such as
`date`, `price_usd`, and optional strategy features.

That runtime parquet is a derived artifact from this canonical long-format dataset.

Minimum projection used by built-in strategy audit tooling:

- `market_cap`
- `supply_btc`
- `mvrv`
- `adjusted_sopr`
- `adjusted_sopr_7d_ema`
- `realized_cap_growth_rate`
- `market_cap_growth_rate`

With:

- `price_usd = market_cap / supply_btc`
- `day_utc` renamed to `date`

## Coverage of runtime-critical metrics (snapshot)

| Metric | Rows | Min day | Max day |
| --- | ---: | --- | --- |
| `market_cap` | `6,274` | `2009-01-03` | `2026-03-13` |
| `supply_btc` | `6,274` | `2009-01-03` | `2026-03-13` |
| `mvrv` | `6,273` | `2009-01-09` | `2026-03-13` |
| `adjusted_sopr` | `5,689` | `2010-08-16` | `2026-03-13` |
| `adjusted_sopr_7d_ema` | `5,689` | `2010-08-16` | `2026-03-13` |
| `realized_cap_growth_rate` | `5,324` | `2011-08-16` | `2026-03-13` |
| `market_cap_growth_rate` | `5,324` | `2011-08-16` | `2026-03-13` |

## Validation query (reproducible)

```bash
python - <<'PY'
import polars as pl
from pathlib import Path

p = Path("merged_metrics_2026-03-15_04-29-57.parquet")
lf = pl.scan_parquet(p)

print(lf.collect_schema())
print(
    lf.select(
        pl.len().alias("rows"),
        pl.col("day_utc").n_unique().alias("unique_days"),
        pl.col("metric").n_unique().alias("unique_metrics"),
        pl.col("day_utc").min().alias("min_day"),
        pl.col("day_utc").max().alias("max_day"),
        pl.col("day_utc").null_count().alias("null_day_utc"),
        pl.col("metric").null_count().alias("null_metric"),
        pl.col("value").null_count().alias("null_value"),
    ).collect()
)
PY
```
