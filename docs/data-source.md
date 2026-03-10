---
title: BRK Data Source
description: Canonical data distribution and integrity workflow for BRK DuckDB artifacts.
---

# BRK Data Source (DuckDB + Schema)

StackSats strategy runtime is BRK-only.

- canonical runtime env var: `STACKSATS_ANALYTICS_DUCKDB`
- local fallback path when env var is unset: `./bitcoin_analytics.duckdb`
- runtime does not auto-download data

Current DuckDB artifact size is large:

- `bitcoin_analytics.duckdb`: `10,876,104,704` bytes (`~10.13 GiB`)
- plan for at least `~12 GiB` of free local disk space before download

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

Because the DuckDB file is over `10 GiB`, first download can take time depending on bandwidth.

If `data/brk_data_manifest.json` still contains placeholder file IDs (for example `REPLACE_WITH_*`), the fetch command will fail by design. In that case, download the DuckDB manually from the Drive folder and place it at repo root before exporting `STACKSATS_ANALYTICS_DUCKDB`.

Then export the runtime path:

```bash
export STACKSATS_ANALYTICS_DUCKDB=$(pwd)/bitcoin_analytics.duckdb
```

## Refreshing Data Metadata (Maintainers)

When Drive artifacts are refreshed:

1. update `file_id`, `sha256`, `size_bytes`, `version`, `updated_at_utc` in `data/brk_data_manifest.json`
2. run `venv/bin/python scripts/fetch_brk_data.py --target-dir . --overwrite`
3. regenerate schema docs: `venv/bin/python scripts/render_duckdb_schema_doc.py`
4. verify docs/tests pass

Do not add network fetches to runtime providers. Keep downloads script-only to preserve deterministic runtime behavior.
