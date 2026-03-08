# Bitcoin Analytics DuckDB Schema

Source database: `bitcoin_analytics.duckdb`
Schema scope: all user tables in `main`
Snapshot date: `2026-03-08`

All metrics in this database were derived via the BRK library from a local Bitcoin node.

Canonical distribution references:

- Drive folder: <https://drive.google.com/drive/folders/1SvAwcdegMzgPANM4pnuTH_9DbNEyXt8N?usp=drive_link>
- Manifest: `data/brk_data_manifest.json`

## Basic DB Stats

Computed from `bitcoin_analytics.duckdb` on `2026-03-08`.

| Stat | Value |
|---|---|
| File size | `10,876,104,704` bytes (`~10.13 GiB`) |
| Total user tables (`main`) | `15` |
| Metric tables (`metrics_*`) | `13` |
| Total metric rows | `153,616,679` |
| Date coverage (metric tables) | `2015-01-01` to `2025-05-31` |
| Unique metrics across all metric tables | `41,231` |
| Null `value` rows across all metric tables | `0` |
| `_long_load_runs` rows | `1` |
| `_long_load_chunks` rows | `374` |

### Metric Table Summary

| Table | Rows | Date range | Unique metrics | Null `value` rows |
|---|---:|---|---:|---:|
| `metrics_blocks` | `555,384` | `2015-01-01` to `2025-05-31` | `146` | `0` |
| `metrics_cointime` | `441,264` | `2015-01-01` to `2025-05-31` | `116` | `0` |
| `metrics_constants` | `68,472` | `2015-01-01` to `2025-05-31` | `18` | `0` |
| `metrics_distribution` | `108,512,622` | `2015-01-01` to `2025-05-31` | `29,351` | `0` |
| `metrics_indexes` | `26,628` | `2015-01-01` to `2025-05-31` | `7` | `0` |
| `metrics_inputs` | `19,020` | `2015-01-01` to `2025-05-31` | `5` | `0` |
| `metrics_market` | `24,372,257` | `2015-01-01` to `2025-05-31` | `6,430` | `0` |
| `metrics_outputs` | `22,824` | `2015-01-01` to `2025-05-31` | `6` | `0` |
| `metrics_pools` | `18,631,992` | `2015-01-01` to `2025-05-31` | `4,898` | `0` |
| `metrics_price` | `60,864` | `2015-01-01` to `2025-05-31` | `16` | `0` |
| `metrics_scripts` | `623,856` | `2015-01-01` to `2025-05-31` | `164` | `0` |
| `metrics_supply` | `83,688` | `2015-01-01` to `2025-05-31` | `22` | `0` |
| `metrics_transactions` | `197,808` | `2015-01-01` to `2025-05-31` | `52` | `0` |

## Tables

### `main._long_load_chunks`

| Column | Type | Nullable |
|---|---|---|
| `run_id` | `VARCHAR` | yes |
| `group_name` | `VARCHAR` | yes |
| `chunk_start` | `DATE` | yes |
| `chunk_end` | `DATE` | yes |
| `metric_batch_start` | `INTEGER` | yes |
| `metric_batch_end` | `INTEGER` | yes |
| `status` | `VARCHAR` | no |
| `attempt_count` | `INTEGER` | no |
| `row_count` | `BIGINT` | yes |
| `error_message` | `VARCHAR` | yes |
| `started_at` | `TIMESTAMP` | yes |
| `finished_at` | `TIMESTAMP` | yes |

Primary key:
`(run_id, group_name, chunk_start, chunk_end, metric_batch_start, metric_batch_end)`

### `main._long_load_runs`

| Column | Type | Nullable |
|---|---|---|
| `run_id` | `VARCHAR` | no |
| `source_path` | `VARCHAR` | no |
| `start_date` | `DATE` | no |
| `status` | `VARCHAR` | no |
| `error_message` | `VARCHAR` | yes |
| `started_at` | `TIMESTAMP` | no |
| `finished_at` | `TIMESTAMP` | yes |

Primary key:
`(run_id)`

### `main.metrics_blocks`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_cointime`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_constants`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_distribution`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_indexes`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_inputs`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_market`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_outputs`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_pools`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_price`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_scripts`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_supply`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |

### `main.metrics_transactions`

| Column | Type | Nullable |
|---|---|---|
| `date_day` | `DATE` | no |
| `metric` | `VARCHAR` | no |
| `value` | `DOUBLE` | yes |
