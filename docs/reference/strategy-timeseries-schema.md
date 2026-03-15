---
title: WeightTimeSeries Schema
description: Generated schema and BRK lineage tables for WeightTimeSeries.
---

# WeightTimeSeries Schema

This page is generated from runtime schema definitions in code.

It documents the core framework schema.
If a strategy exports additional columns, those must be declared explicitly via `extra_schema`; they are merged at runtime but are not listed here unless you render schema markdown from an instance or batch carrying that extension.

Refresh with:

`venv/bin/python scripts/sync_objects_schema_docs.py`

## Data columns (generated)

<!-- BEGIN: TIMESERIES_SCHEMA_TABLE -->

| name | dtype | required | description | unit | constraints | source | formula |
| --- | --- | --- | --- | --- | --- | --- | --- |
| day_index | int64 | False | Zero-based day index within the allocation window. |  | >=0, strictly increasing by 1 | framework |  |
| date | datetime64[ns] | True | Calendar day for this allocation row. |  | unique, sorted ascending, daily grain | framework |  |
| weight | float64 | True | Final feasible daily allocation after clipping, lock preservation, and remaining-budget constraints. |  | finite, >=0, sum ~= 1.0 | framework |  |
| price_usd | float64 | True | BTC price in USD for the given date when available. | USD | finite when present, nullable for future dates | framework |  |
| locked | bool | False | True when a row belongs to an immutable locked history prefix. |  | boolean values only | framework |  |
| PriceUSD | float64 | False | Raw BRK BTC price column when preserved in payloads. | USD | finite when present | brk | raw PriceUSD |
| mvrv | float64 | False | BRK MVRV ratio when retained in strategy payloads. |  | finite when present | brk |  |
| time | datetime64[ns] | False | BRK daily timestamp column. |  | valid datetime when present | brk | raw time |
| AdrActCnt | float64 | False | BRK active addresses count. |  | finite when present | brk |  |
| AdrBalCnt | float64 | False | BRK addresses with non-zero balance. |  | finite when present | brk |  |
| AssetCompletionTime | datetime64[ns] | False | BRK ingestion completion timestamp for asset-day data. |  | valid datetime when present | brk |  |
| AssetEODCompletionTime | datetime64[ns] | False | BRK end-of-day completion timestamp for asset metrics. |  | valid datetime when present | brk |  |
| BlkCnt | float64 | False | BRK blocks mined during the day. |  | finite when present | brk |  |
| CapMrktCurUSD | float64 | False | BRK current market capitalization in USD. | USD | finite when present | brk |  |
| CapMrktEstUSD | float64 | False | BRK estimated market capitalization in USD. | USD | finite when present | brk |  |
| FeeTotNtv | float64 | False | BRK total transaction fees in native BTC units. | BTC | finite when present | brk |  |
| FlowInExNtv | float64 | False | BRK exchange inflow in native BTC units. | BTC | finite when present | brk |  |
| FlowInExUSD | float64 | False | BRK exchange inflow valued in USD. | USD | finite when present | brk |  |
| FlowOutExNtv | float64 | False | BRK exchange outflow in native BTC units. | BTC | finite when present | brk |  |
| FlowOutExUSD | float64 | False | BRK exchange outflow valued in USD. | USD | finite when present | brk |  |
| HashRate | float64 | False | BRK network hash rate estimate. |  | finite when present | brk |  |
| IssTotNtv | float64 | False | BRK total daily issuance in native BTC units. | BTC | finite when present | brk |  |
| IssTotUSD | float64 | False | BRK total daily issuance valued in USD. | USD | finite when present | brk |  |
| PriceBTC | float64 | False | BRK BTC reference price quoted in BTC. | BTC | finite when present | brk |  |
| ROI1yr | float64 | False | BRK trailing 1-year return metric. |  | finite when present | brk |  |
| ROI30d | float64 | False | BRK trailing 30-day return metric. |  | finite when present | brk |  |
| ReferenceRate | float64 | False | BRK reference rate for BTC. |  | finite when present | brk |  |
| ReferenceRateETH | float64 | False | BRK reference rate for BTC quoted in ETH. |  | finite when present | brk |  |
| ReferenceRateEUR | float64 | False | BRK reference rate for BTC quoted in EUR. |  | finite when present | brk |  |
| ReferenceRateUSD | float64 | False | BRK reference rate for BTC quoted in USD. | USD | finite when present | brk |  |
| SplyCur | float64 | False | BRK current circulating BTC supply. | BTC | finite when present | brk |  |
| SplyExNtv | float64 | False | BRK supply held on exchanges in native BTC units. | BTC | finite when present | brk |  |
| SplyExUSD | float64 | False | BRK supply held on exchanges valued in USD. | USD | finite when present | brk |  |
| SplyExpFut10yr | float64 | False | BRK projected BTC supply 10 years ahead. | BTC | finite when present | brk |  |
| TxCnt | float64 | False | BRK on-chain transaction count. |  | finite when present | brk |  |
| TxTfrCnt | float64 | False | BRK transfer transaction count. |  | finite when present | brk |  |
| volume_reported_spot_usd_1d | float64 | False | BRK reported spot exchange volume in USD for 1 day. | USD | finite when present | brk |  |

<!-- END: TIMESERIES_SCHEMA_TABLE -->

## BRK lineage (generated)

<!-- BEGIN: TIMESERIES_BRK_LINEAGE -->

| source_column | required | description | strategy_column | notes |
| --- | --- | --- | --- | --- |
| time | True | BRK daily timestamp column. | date | Loaded as index, then represented by TimeSeries.date. |
| AdrActCnt | False | BRK active addresses count. | AdrActCnt |  |
| AdrBalCnt | False | BRK addresses with non-zero balance. | AdrBalCnt |  |
| AssetCompletionTime | False | BRK ingestion completion timestamp for asset-day data. | AssetCompletionTime |  |
| AssetEODCompletionTime | False | BRK end-of-day completion timestamp for asset metrics. | AssetEODCompletionTime |  |
| BlkCnt | False | BRK blocks mined during the day. | BlkCnt |  |
| PriceUSD | True | BRK BTC close price in USD. | price_usd | Aliased to price_usd before export normalization. |
| mvrv | False | BRK current market-value-to-realized-value ratio. | mvrv |  |
| CapMrktCurUSD | False | BRK current market capitalization in USD. | CapMrktCurUSD |  |
| CapMrktEstUSD | False | BRK estimated market capitalization in USD. | CapMrktEstUSD |  |
| FeeTotNtv | False | BRK total transaction fees in native BTC units. | FeeTotNtv |  |
| FlowInExNtv | False | BRK exchange inflow in native BTC units. | FlowInExNtv |  |
| FlowInExUSD | False | BRK exchange inflow valued in USD. | FlowInExUSD |  |
| FlowOutExNtv | False | BRK exchange outflow in native BTC units. | FlowOutExNtv |  |
| FlowOutExUSD | False | BRK exchange outflow valued in USD. | FlowOutExUSD |  |
| HashRate | False | BRK network hash rate estimate. | HashRate |  |
| IssTotNtv | False | BRK total daily issuance in native BTC units. | IssTotNtv |  |
| IssTotUSD | False | BRK total daily issuance valued in USD. | IssTotUSD |  |
| PriceBTC | False | BRK BTC reference price quoted in BTC. | PriceBTC |  |
| ROI1yr | False | BRK trailing 1-year return metric. | ROI1yr |  |
| ROI30d | False | BRK trailing 30-day return metric. | ROI30d |  |
| ReferenceRate | False | BRK reference rate for BTC. | ReferenceRate |  |
| ReferenceRateETH | False | BRK reference rate for BTC quoted in ETH. | ReferenceRateETH |  |
| ReferenceRateEUR | False | BRK reference rate for BTC quoted in EUR. | ReferenceRateEUR |  |
| ReferenceRateUSD | False | BRK reference rate for BTC quoted in USD. | ReferenceRateUSD |  |
| SplyCur | False | BRK current circulating BTC supply. | SplyCur |  |
| SplyExNtv | False | BRK supply held on exchanges in native BTC units. | SplyExNtv |  |
| SplyExUSD | False | BRK supply held on exchanges valued in USD. | SplyExUSD |  |
| SplyExpFut10yr | False | BRK projected BTC supply 10 years ahead. | SplyExpFut10yr |  |
| TxCnt | False | BRK on-chain transaction count. | TxCnt |  |
| TxTfrCnt | False | BRK transfer transaction count. | TxTfrCnt |  |
| volume_reported_spot_usd_1d | False | BRK reported spot exchange volume in USD for 1 day. | volume_reported_spot_usd_1d |  |

<!-- END: TIMESERIES_BRK_LINEAGE -->
