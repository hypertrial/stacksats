---
title: BRK Data Source
description: Canonical data distribution and integrity workflow for BRK parquet artifacts.
---

# BRK Data Source (Parquet + Schema)

StackSats strategy runtime is BRK-only.

- canonical runtime env var: `STACKSATS_ANALYTICS_PARQUET`
- local fallback path when env var is unset: `./bitcoin_analytics.parquet`
- runtime does not auto-download data

## Expected parquet schema

The parquet file must have a daily datetime index (or a `date` column) and at least:

- `price_usd`: required
- `mvrv`: optional; used by overlay and some strategies

Optional overlay columns (when present, used by `brk_overlay_v1`): `adjusted_sopr`, `adjusted_sopr_7d_ema`, `realized_cap_growth_rate`, `market_cap_growth_rate`, `tx_count_pct10`, `annualized_volume_usd`, `hash_rate_1y_sma`, `subsidy_usd_average`, `net_sentiment`, `greed_index`, `pain_index`.

## Canonical Source of Truth

- Google Drive folder: <https://drive.google.com/drive/folders/1SvAwcdegMzgPANM4pnuTH_9DbNEyXt8N?usp=drive_link>
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
- downloads schema markdown to `./docs/reference/bitcoin-analytics-parquet-schema.md`
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
3. verify docs/tests pass

Do not add network fetches to runtime providers. Keep downloads script-only to preserve deterministic runtime behavior.
