---
title: Merged Metrics Data Guide
description: What data a new user can access in the canonical BRK merged_metrics parquet.
---

# Merged Metrics Data Guide

> Generated from `scripts/generate_merged_metrics_taxonomy.py` against `merged_metrics_2026-03-15_04-29-57.parquet`.

Use this page first if you want to understand what data you actually have access to through the canonical Bitcoin Research Kit (BRK) `merged_metrics*.parquet` dataset consumed by StackSats.

Reading order:

1. [Merged Metrics Data Guide](merged-metrics-data-guide.md)
2. [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md)
3. [Merged Metrics Taxonomy](merged-metrics-taxonomy.md)

## What This Dataset Is

The canonical BRK dataset consumed by StackSats is a long-format daily fact table with exactly three columns:

- `day_utc`: UTC calendar day
- `metric`: metric key
- `value`: numeric value for that metric on that day

Each row is one daily numeric observation for one Bitcoin analytics metric. The dataset gives you access to thousands of derived BTC time series, not a wide table of fixed columns and not raw transaction-level blockchain records.

Current snapshot scale:

- `236,259,020` rows
- `6,274` daily observations
- `41,407` metric keys
- `2009-01-03` to `2026-03-13` coverage in the current snapshot

## What Data You Can Access

The metric namespace covers these major user-facing domains:

| Domain | What it covers | Representative metrics | Typical use | Coverage notes |
| --- | --- | --- | --- | --- |
| Address balance cohorts | Address cohorts partitioned by balance bucket. | `addrs_above_100btc_under_1k_btc__30d_change`, `addrs_above_100btc_under_1k_btc__30d_change_btc`, `addrs_above_100btc_under_1k_btc__30d_change_usd` | Whale, retail, and cohort-balance distribution analysis. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Benchmarks, path metrics, and technical indicators | Windowed return paths, DCA/lump-sum ladders, and indicator-style metrics. | `10y_cagr`, `10y_dca_average_price`, `10y_dca_average_price_sats` | Benchmarking, path-dependent outcomes, and technical overlays. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Blocks, transactions, and network activity | Blocks, transactions, throughput, activity, address counts, and network utilization. | `_30d_change`, `_30d_change_btc`, `_30d_change_usd` | On-chain activity monitoring and network-usage context. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Holder cohorts | Short-term and long-term holder slices and holder-behavior metrics. | `lth__30d_change`, `lth__30d_change_btc`, `lth__30d_change_usd` | Compare STH and LTH behavior across market regimes. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Market and valuation | Price, valuation bands, market value, realized value, and valuation ratios. | `active_cap`, `active_price`, `active_price_ratio` | Price context, valuation regimes, and long-horizon market state modeling. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Mining pools and miner economics | Mining-pool shares plus miner economics such as hash-price and fee flows. | `aaopool_1m_blocks_mined`, `aaopool_1m_dominance`, `aaopool_1w_blocks_mined` | Miner revenue, pool share, and mining-economics monitoring. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Profitability and SOPR | Realized and unrealized profit or loss, SOPR-style metrics, and spending pressure. | `adjusted_sopr`, `adjusted_sopr_30d_ema`, `adjusted_sopr_7d_ema` | Profit-taking, capitulation, and spending-behavior analysis. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Script and output types | Output/script-type cohorts including p2* families, unknown, empty, and OP_RETURN. | `empty_addr_count`, `empty_addr_count_30d_change`, `empty_outputs__30d_change` | Track script adoption, output composition, and script-specific cohorts. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Supply, issuance, and scarcity | Circulating supply, issuance, subsidy, inflation, and scarcity metrics. | `active_supply`, `active_supply_btc`, `active_supply_usd` | Supply-side modeling and issuance or dilution analysis. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| UTXO age cohorts | Age-bucketed UTXO metrics by holding period cohort. | `utxos_10y_to_12y_old__30d_change`, `utxos_10y_to_12y_old__30d_change_btc`, `utxos_10y_to_12y_old__30d_change_usd` | Analyze age-distributed supply, spentness, and conviction. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |
| Vintage and halving cohorts | Year-vintage and halving-epoch cohort metrics. | `epoch_0__30d_change`, `epoch_0__30d_change_btc`, `epoch_0__30d_change_usd` | Cycle-aware cohort comparisons and halving-era analysis. | 2009-01-03 to 2026-03-13 across at least some metrics; per-metric coverage still varies. |

## Major Data Domains

High-level examples of what a new user can query from the metric namespace:

- Market and valuation: `price_*`, `market_*`, `realized_*`, `mvrv`, `investor_*`, `cost_*`
- Profitability and SOPR: `*_sopr*`, `*_profit*`, `*_loss*`, `capitulation_*`, `pain_*`
- Supply and scarcity: `supply_*`, `circulating_*`, `subsidy_*`, `inflation_*`
- Holder cohorts: `sth_*`, `lth_*`
- UTXO age cohorts: `utxos_<age_bucket>_*`
- Address balance cohorts: `addrs_<balance_bucket>_*`
- Vintage and halving cohorts: `year_<yyyy>_*`, `epoch_<n>_*`
- Mining pools and miner economics: `<pool>_blocks_mined`, `<pool>_dominance`, `hash_price_*`, `coinbase_*`, `fee_*`
- Script and output types: `p2*_*`, `unknown_*`, `empty_*`, `opreturn_*`, `segwit_*`, `taproot_*`
- Blocks, transactions, and network activity: `block_*`, `tx_*`, `hash_rate`, `difficulty*`, `sent*`, `received*`
- Benchmarks and path metrics: `1m_*`, `1y_*`, `10y_*`, `dca_*`, `rsi_*`, `macd_*`

## What This Dataset Does Not Contain

- It does not expose raw transaction rows, raw block rows, or raw address ledgers.
- It does not provide intraday timestamps; observations are daily.
- It does not make every metric a dedicated parquet column; access happens through metric keys in long format.
- StackSats runtime does not consume all `41,407` metrics directly. Runtime uses a smaller derived BRK-wide projection.

## Coverage Caveats

- Coverage is metric-specific. Some metrics begin much later than `2009-01-03`.
- Newer transforms and ratios can have shorter history because they depend on warmup windows or derived inputs.
- Use `data/brk_merged_metrics_catalog.json` to inspect `coverage_rows`, `first_day`, and `last_day` for each metric.

## Metrics Used By StackSats Runtime

These runtime-critical metrics are the minimum projection used by built-in strategy audit tooling:

| Metric | Coverage rows | First day | Last day |
| --- | ---: | --- | --- |
| `market_cap` | `6,274` | `2009-01-03` | `2026-03-13` |
| `supply_btc` | `6,274` | `2009-01-03` | `2026-03-13` |
| `mvrv` | `6,273` | `2009-01-09` | `2026-03-13` |
| `adjusted_sopr` | `5,689` | `2010-08-16` | `2026-03-13` |
| `adjusted_sopr_7d_ema` | `5,689` | `2010-08-16` | `2026-03-13` |
| `realized_cap_growth_rate` | `5,324` | `2011-08-16` | `2026-03-13` |
| `market_cap_growth_rate` | `5,324` | `2011-08-16` | `2026-03-13` |

The runtime projection renames `day_utc` to `date` and derives `price_usd` from `market_cap / supply_btc`. This runtime projection is a StackSats-derived artifact built from canonical BRK source data. See [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md) and [BRK Data Source](../data-source.md).

## How To Search The Catalog And Taxonomy

- Use `data/brk_merged_metrics_catalog.json` when you want per-metric access metadata and coverage.
- Use [Merged Metrics Taxonomy](merged-metrics-taxonomy.md) when you want family-level naming patterns.
- Search by `access_category` to find domains, by `family` to find namespaces, and by `display_label` or `metric` when you already know the concept or key.
- If a metric name looks inconsistent, the catalog `notes` field explains metadata-only normalization such as collapsed double underscores or family aliases.

## Related Pages

- [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md)
- [Merged Metrics Taxonomy](merged-metrics-taxonomy.md)
- [BRK Data Source](../data-source.md)
