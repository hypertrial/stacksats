# Fundamental Objects

StackSats now has two fundamental runtime objects:

- `strategy`: user intent object (`BaseStrategy`)
- `StrategyTimeSeries` / `StrategyTimeSeriesBatch`: validated final output objects

## 1) `strategy`

A strategy subclasses `BaseStrategy` (`stacksats/strategy_types.py`) and defines:

- identity: `strategy_id`, `version`, `description`
- hooks: `transform_features`, `build_signals`
- intent path: `propose_weight(...)` or `build_target_profile(...)`

Framework-owned behavior remains sealed:

- compute kernel (`compute_weights` in `BaseStrategy`)
- clipping and remaining-budget enforcement
- lock semantics and final invariants

## 2) `StrategyTimeSeries`

`StrategyTimeSeries` (`stacksats/strategy_time_series.py`) is a single-window output object.

### Required metadata

- `strategy_id`
- `strategy_version`
- `run_id`
- `config_hash`
- `schema_version`
- `generated_at`
- `window_start`
- `window_end`

### Data columns (generated)

Refresh with:

`python scripts/sync_objects_schema_docs.py`

<!-- BEGIN: STRATEGY_TIMESERIES_SCHEMA_TABLE -->

| name | dtype | required | description | unit | constraints | source | formula |
| --- | --- | --- | --- | --- | --- | --- | --- |
| day_index | int64 | False | Zero-based day index within the allocation window. |  | >=0, strictly increasing by 1 | framework |  |
| date | datetime64[ns] | True | Calendar day for this allocation row. |  | unique, sorted ascending, daily grain | framework |  |
| weight | float64 | True | Final feasible daily allocation after clipping, lock preservation, and remaining-budget constraints. |  | finite, >=0, sum ~= 1.0 | framework |  |
| price_usd | float64 | True | BTC price in USD for the given date when available. | USD | finite when present, nullable for future dates | framework |  |
| locked | bool | False | True when a row belongs to an immutable locked history prefix. |  | boolean values only | framework |  |
| PriceUSD | float64 | False | Raw CoinMetrics BTC price column when preserved in payloads. | USD | finite when present | coinmetrics | raw PriceUSD |
| PriceUSD_coinmetrics | float64 | False | Runtime alias for CoinMetrics BTC price when retained. | USD | finite when present | coinmetrics | PriceUSD -> PriceUSD_coinmetrics |
| CapMVRVCur | float64 | False | CoinMetrics MVRV ratio when retained in strategy payloads. |  | finite when present | coinmetrics |  |
| time | datetime64[ns] | False | CoinMetrics daily timestamp column. |  | valid datetime when present | coinmetrics | raw time |
| AdrActCnt | float64 | False | CoinMetrics active addresses count. |  | finite when present | coinmetrics |  |
| AdrBalCnt | float64 | False | CoinMetrics addresses with non-zero balance. |  | finite when present | coinmetrics |  |
| AssetCompletionTime | datetime64[ns] | False | CoinMetrics ingestion completion timestamp for asset-day data. |  | valid datetime when present | coinmetrics |  |
| AssetEODCompletionTime | datetime64[ns] | False | CoinMetrics end-of-day completion timestamp for asset metrics. |  | valid datetime when present | coinmetrics |  |
| BlkCnt | float64 | False | CoinMetrics blocks mined during the day. |  | finite when present | coinmetrics |  |
| CapMrktCurUSD | float64 | False | CoinMetrics current market capitalization in USD. | USD | finite when present | coinmetrics |  |
| CapMrktEstUSD | float64 | False | CoinMetrics estimated market capitalization in USD. | USD | finite when present | coinmetrics |  |
| FeeTotNtv | float64 | False | CoinMetrics total transaction fees in native BTC units. | BTC | finite when present | coinmetrics |  |
| FlowInExNtv | float64 | False | CoinMetrics exchange inflow in native BTC units. | BTC | finite when present | coinmetrics |  |
| FlowInExUSD | float64 | False | CoinMetrics exchange inflow valued in USD. | USD | finite when present | coinmetrics |  |
| FlowOutExNtv | float64 | False | CoinMetrics exchange outflow in native BTC units. | BTC | finite when present | coinmetrics |  |
| FlowOutExUSD | float64 | False | CoinMetrics exchange outflow valued in USD. | USD | finite when present | coinmetrics |  |
| HashRate | float64 | False | CoinMetrics network hash rate estimate. |  | finite when present | coinmetrics |  |
| IssTotNtv | float64 | False | CoinMetrics total daily issuance in native BTC units. | BTC | finite when present | coinmetrics |  |
| IssTotUSD | float64 | False | CoinMetrics total daily issuance valued in USD. | USD | finite when present | coinmetrics |  |
| PriceBTC | float64 | False | CoinMetrics BTC reference price quoted in BTC. | BTC | finite when present | coinmetrics |  |
| ROI1yr | float64 | False | CoinMetrics trailing 1-year return metric. |  | finite when present | coinmetrics |  |
| ROI30d | float64 | False | CoinMetrics trailing 30-day return metric. |  | finite when present | coinmetrics |  |
| ReferenceRate | float64 | False | CoinMetrics reference rate for BTC. |  | finite when present | coinmetrics |  |
| ReferenceRateETH | float64 | False | CoinMetrics reference rate for BTC quoted in ETH. |  | finite when present | coinmetrics |  |
| ReferenceRateEUR | float64 | False | CoinMetrics reference rate for BTC quoted in EUR. |  | finite when present | coinmetrics |  |
| ReferenceRateUSD | float64 | False | CoinMetrics reference rate for BTC quoted in USD. | USD | finite when present | coinmetrics |  |
| SplyCur | float64 | False | CoinMetrics current circulating BTC supply. | BTC | finite when present | coinmetrics |  |
| SplyExNtv | float64 | False | CoinMetrics supply held on exchanges in native BTC units. | BTC | finite when present | coinmetrics |  |
| SplyExUSD | float64 | False | CoinMetrics supply held on exchanges valued in USD. | USD | finite when present | coinmetrics |  |
| SplyExpFut10yr | float64 | False | CoinMetrics projected BTC supply 10 years ahead. | BTC | finite when present | coinmetrics |  |
| TxCnt | float64 | False | CoinMetrics on-chain transaction count. |  | finite when present | coinmetrics |  |
| TxTfrCnt | float64 | False | CoinMetrics transfer transaction count. |  | finite when present | coinmetrics |  |
| volume_reported_spot_usd_1d | float64 | False | CoinMetrics reported spot exchange volume in USD for 1 day. | USD | finite when present | coinmetrics |  |

<!-- END: STRATEGY_TIMESERIES_SCHEMA_TABLE -->

### CoinMetrics lineage (generated)

This table is the source-of-truth mapping from CoinMetrics input columns to
`StrategyTimeSeries` schema columns.
Every BTC CoinMetrics column used by the project is handwritten in
`stacksats/strategy_time_series.py` and validated in tests/CI.

<!-- BEGIN: STRATEGY_TIMESERIES_COINMETRICS_LINEAGE -->

| source_column | required | description | strategy_column | notes |
| --- | --- | --- | --- | --- |
| time | True | CoinMetrics daily timestamp column. | date | Loaded as index, then represented by StrategyTimeSeries.date. |
| AdrActCnt | False | CoinMetrics active addresses count. | AdrActCnt |  |
| AdrBalCnt | False | CoinMetrics addresses with non-zero balance. | AdrBalCnt |  |
| AssetCompletionTime | False | CoinMetrics ingestion completion timestamp for asset-day data. | AssetCompletionTime |  |
| AssetEODCompletionTime | False | CoinMetrics end-of-day completion timestamp for asset metrics. | AssetEODCompletionTime |  |
| BlkCnt | False | CoinMetrics blocks mined during the day. | BlkCnt |  |
| PriceUSD | True | CoinMetrics BTC close price in USD. | price_usd | Aliased to PriceUSD_coinmetrics before export normalization. |
| CapMVRVCur | False | CoinMetrics current market-value-to-realized-value ratio. | CapMVRVCur |  |
| CapMrktCurUSD | False | CoinMetrics current market capitalization in USD. | CapMrktCurUSD |  |
| CapMrktEstUSD | False | CoinMetrics estimated market capitalization in USD. | CapMrktEstUSD |  |
| FeeTotNtv | False | CoinMetrics total transaction fees in native BTC units. | FeeTotNtv |  |
| FlowInExNtv | False | CoinMetrics exchange inflow in native BTC units. | FlowInExNtv |  |
| FlowInExUSD | False | CoinMetrics exchange inflow valued in USD. | FlowInExUSD |  |
| FlowOutExNtv | False | CoinMetrics exchange outflow in native BTC units. | FlowOutExNtv |  |
| FlowOutExUSD | False | CoinMetrics exchange outflow valued in USD. | FlowOutExUSD |  |
| HashRate | False | CoinMetrics network hash rate estimate. | HashRate |  |
| IssTotNtv | False | CoinMetrics total daily issuance in native BTC units. | IssTotNtv |  |
| IssTotUSD | False | CoinMetrics total daily issuance valued in USD. | IssTotUSD |  |
| PriceBTC | False | CoinMetrics BTC reference price quoted in BTC. | PriceBTC |  |
| PriceUSD_coinmetrics | True | Runtime alias of CoinMetrics PriceUSD. | price_usd | Canonical runtime price input consumed by model and export. |
| ROI1yr | False | CoinMetrics trailing 1-year return metric. | ROI1yr |  |
| ROI30d | False | CoinMetrics trailing 30-day return metric. | ROI30d |  |
| ReferenceRate | False | CoinMetrics reference rate for BTC. | ReferenceRate |  |
| ReferenceRateETH | False | CoinMetrics reference rate for BTC quoted in ETH. | ReferenceRateETH |  |
| ReferenceRateEUR | False | CoinMetrics reference rate for BTC quoted in EUR. | ReferenceRateEUR |  |
| ReferenceRateUSD | False | CoinMetrics reference rate for BTC quoted in USD. | ReferenceRateUSD |  |
| SplyCur | False | CoinMetrics current circulating BTC supply. | SplyCur |  |
| SplyExNtv | False | CoinMetrics supply held on exchanges in native BTC units. | SplyExNtv |  |
| SplyExUSD | False | CoinMetrics supply held on exchanges valued in USD. | SplyExUSD |  |
| SplyExpFut10yr | False | CoinMetrics projected BTC supply 10 years ahead. | SplyExpFut10yr |  |
| TxCnt | False | CoinMetrics on-chain transaction count. | TxCnt |  |
| TxTfrCnt | False | CoinMetrics transfer transaction count. | TxTfrCnt |  |
| volume_reported_spot_usd_1d | False | CoinMetrics reported spot exchange volume in USD for 1 day. | volume_reported_spot_usd_1d |  |

<!-- END: STRATEGY_TIMESERIES_COINMETRICS_LINEAGE -->

### Core methods

- `schema()`
- `schema_markdown()`
- `validate_schema_coverage()`
- `validate()`
- `to_dataframe()`

### Validation guarantees

- required columns exist
- `date` is valid, unique, sorted ascending
- `weight` is finite, non-negative, sums to `1.0` (tolerance)
- `price_usd` is finite when present (nullable for future rows)
- all columns are covered by handwritten schema specs
- `window_start` / `window_end` match series boundaries

## 3) `StrategyTimeSeriesBatch`

`StrategyTimeSeriesBatch` is a multi-window container returned by export APIs.

### Batch guarantees

- contains one or more `StrategyTimeSeries` windows
- each window has a unique `(window_start, window_end)` key
- per-window provenance matches batch-level provenance

### Core methods

- `from_flat_dataframe(...)`
- `to_dataframe()`
- `iter_windows()`
- `for_window(start_date, end_date)`
- `schema_markdown()`

## Export contract

`StrategyRunner.export(...)` returns `StrategyTimeSeriesBatch`.

Export artifacts remain under:

`<output_dir>/<strategy_id>/<version>/<run_id>/`

and include:

- `weights.csv`
- `timeseries_schema.md`
- `artifacts.json`

Canonical `weights.csv` columns:

- `start_date`
- `end_date`
- `day_index`
- `date`
- `price_usd`
- `weight`

## References

- `stacksats/strategy_types.py`
- `stacksats/strategy_time_series.py`
- `stacksats/runner.py`
- `docs/framework.md`
