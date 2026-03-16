---
title: BRK Data Source
description: Canonical merged_metrics parquet contract and runtime projection workflow.
---

# BRK Data Source (Canonical Merged Metrics + Runtime Projection)

StackSats canonical dataset is the long-format Google Drive parquet:
`merged_metrics*.parquet`.

Canonical file link:

- <https://drive.google.com/file/d/1jKRRU7l9kOMdGI_hIJGg02X3jWTMPJsw/view?usp=sharing>

Canonical schema page:

- [Merged Metrics Parquet Schema](reference/merged-metrics-parquet-schema.md)
- [Merged Metrics Taxonomy](reference/merged-metrics-taxonomy.md)

## Snapshot scale (current canonical repo snapshot)

The current canonical snapshot is large enough to support long-horizon strategy
research, not just toy examples:

- `236,259,020` rows
- `6,274` daily observations
- `41,407` distinct metric keys
- `284` top-level metric families
- coverage from `2009-01-03` to `2026-03-13`

## Canonical merged_metrics schema

The canonical parquet has exactly:

- `day_utc` (`Date`)
- `metric` (`String`)
- `value` (`Float64`)

This is the source-of-truth dataset for StackSats documentation and data workflow.
The physical long-format schema and the semantic metric taxonomy are documented
separately so metric grouping can evolve without conflating it with the parquet
column contract.

## Runtime ingestion contract

Runtime APIs are strict and deterministic:

- runtime env var: `STACKSATS_ANALYTICS_PARQUET`
- local fallback path when env var is unset: `./bitcoin_analytics.parquet`
- runtime does not auto-download data
- framework loaders retain pre-start history by default for feature warmup; scoring windows still respect requested start/end bounds

Runtime expects a BRK-wide parquet (for example columns like `date`, `price_usd`,
`mvrv`, and optional overlay features).

That BRK-wide parquet is a derived artifact from canonical `merged_metrics`.

Current minimal projection for built-in strategy audit tooling:

- `market_cap`
- `supply_btc`
- `mvrv`
- `adjusted_sopr`
- `adjusted_sopr_7d_ema`
- `realized_cap_growth_rate`
- `market_cap_growth_rate`

With:

- `price_usd = market_cap / supply_btc`
- rename `day_utc` to `date`

## Derive runtime parquet from canonical merged_metrics

```bash
python - <<'PY'
import polars as pl
from pathlib import Path

src = Path("merged_metrics_2026-03-15_04-29-57.parquet")
dst = Path("bitcoin_analytics.parquet")

metrics = [
    "market_cap",
    "supply_btc",
    "mvrv",
    "adjusted_sopr",
    "adjusted_sopr_7d_ema",
    "realized_cap_growth_rate",
    "market_cap_growth_rate",
]

(
    pl.scan_parquet(src)
    .filter(pl.col("metric").is_in(metrics))
    .select("day_utc", "metric", "value")
    .collect()
    .pivot(values="value", index="day_utc", on="metric")
    .with_columns((pl.col("market_cap") / pl.col("supply_btc")).alias("price_usd"))
    .rename({"day_utc": "date"})
    .select(
        "date",
        "price_usd",
        "mvrv",
        "adjusted_sopr",
        "adjusted_sopr_7d_ema",
        "realized_cap_growth_rate",
        "market_cap_growth_rate",
    )
    .filter(pl.col("price_usd").is_finite() & (pl.col("price_usd") > 0))
    .write_parquet(dst)
)
print(f"wrote {dst.resolve()}")
PY

export STACKSATS_ANALYTICS_PARQUET=$(pwd)/bitcoin_analytics.parquet
```

## Canonical Source of Truth

- Google Drive parquet: <https://drive.google.com/file/d/1jKRRU7l9kOMdGI_hIJGg02X3jWTMPJsw/view?usp=sharing>
- Manifest in repo: `data/brk_data_manifest.json`

The manifest defines, for both parquet and schema artifacts:

- `name`
- `file_id`
- `sha256`
- `size_bytes`
- `version`

It also tracks:

- `gdrive_folder_url`
- `updated_at_utc`

## Fetch + Verify Workflow

From repo root:

```bash
venv/bin/python scripts/fetch_brk_data.py --target-dir .
```

Default behavior:

- downloads parquet to `./bitcoin_analytics.parquet`
- downloads schema markdown path from manifest metadata
- verifies `sha256` and exact file size from manifest
- fails closed on missing metadata, hash mismatch, size mismatch, or partial download

If `data/brk_data_manifest.json` still contains placeholder file IDs (for example `REPLACE_WITH_*`), the fetch command will fail by design. In that case, download the parquet manually from the Drive folder and place it at repo root before exporting `STACKSATS_ANALYTICS_PARQUET`.

Then export the runtime path:

```bash
export STACKSATS_ANALYTICS_PARQUET=$(pwd)/bitcoin_analytics.parquet
```

## Refreshing Data Metadata (Maintainers)

When Drive artifacts are refreshed:

1. update `file_id`, `sha256`, `size_bytes`, `version`, `updated_at_utc` in `data/brk_data_manifest.json`
2. run `venv/bin/python scripts/fetch_brk_data.py --target-dir . --overwrite`
3. update [Merged Metrics Parquet Schema](reference/merged-metrics-parquet-schema.md) when canonical schema/profile changes
4. verify docs/tests pass

Do not add network fetches to runtime providers. Keep downloads script-only to preserve deterministic runtime behavior.
