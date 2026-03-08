---
title: BRK Data Source
description: Canonical data distribution and integrity workflow for BRK DuckDB artifacts.
---

# BRK Data Source (DuckDB + Schema)

StackSats strategy runtime is BRK-only.

- canonical runtime env var: `STACKSATS_ANALYTICS_DUCKDB`
- local fallback path when env var is unset: `./bitcoin_analytics.duckdb`
- runtime does not auto-download data

## Canonical Source of Truth

- Google Drive folder: <https://drive.google.com/drive/folders/1SvAwcdegMzgPANM4pnuTH_9DbNEyXt8N?usp=drive_link>
- Manifest in repo: `data/brk_data_manifest.json`

The manifest defines, for both DuckDB and schema artifacts:

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

- downloads DuckDB to `./bitcoin_analytics.duckdb`
- downloads schema markdown to `./docs/reference/bitcoin-analytics-duckdb-schema.md`
- verifies `sha256` and exact file size from manifest
- fails closed on missing metadata, hash mismatch, size mismatch, or partial download

Then export the runtime path:

```bash
export STACKSATS_ANALYTICS_DUCKDB=$(pwd)/bitcoin_analytics.duckdb
```

## Refreshing Data Metadata (Maintainers)

When Drive artifacts are refreshed:

1. update `file_id`, `sha256`, `size_bytes`, `version`, `updated_at_utc` in `data/brk_data_manifest.json`
2. run `venv/bin/python scripts/fetch_brk_data.py --target-dir . --overwrite`
3. verify docs/tests pass

Do not add network fetches to runtime providers. Keep downloads script-only to preserve deterministic runtime behavior.
