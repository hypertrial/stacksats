---
title: Merged Metrics Taxonomy
description: Generated semantic taxonomy for the BRK merged_metrics parquet metric namespace.
---

# Merged Metrics Taxonomy

> Generated from `scripts/generate_merged_metrics_taxonomy.py` against `merged_metrics_2026-03-15_04-29-57.parquet`.

This page documents the semantic metric taxonomy for the canonical long-format `merged_metrics*.parquet` dataset.

Dataset scale in the current canonical snapshot:

- `236,259,020` total rows
- `6,274` daily observations
- `41,407` distinct metric keys
- `284` top-level metric families

Use [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md) for the physical parquet schema and runtime projection contract.

Canonical generated artifacts:

- markdown page: `docs/reference/merged-metrics-taxonomy.md`
- JSON taxonomy: `data/brk_merged_metrics_taxonomy.json`
- refresh command: `python scripts/generate_merged_metrics_taxonomy.py`

## Snapshot

| Property | Value |
| --- | --- |
| Parquet file | `merged_metrics_2026-03-15_04-29-57.parquet` |
| Total rows | `236,259,020` |
| Distinct days | `6,274` |
| Distinct metrics | `41,407` |
| Day range | `2009-01-03` to `2026-03-13` |
| Top-level families | `284` |

## Semantic Classes

| Class | Namespace count | Metric count | Meaning | Rule |
| --- | ---: | ---: | --- | --- |
| `address_activity_aggregates` | `1` | `48` | Address activity distribution aggregates. | Top-level prefix is address. |
| `address_balance_cohorts` | `1` | `5249` | Address balance-bucket cohort metrics. | Top-level prefix is addrs. |
| `benchmark_class_metrics` | `1` | `120` | Benchmark class path metrics such as DCA cohort ladders. | Top-level prefix is dca. |
| `block_aggregates` | `1` | `49` | Block production, size, interval, weight, and fullness aggregates. | Top-level prefix is block. |
| `core_market_metrics` | `31` | `6503` | Standalone market, valuation, supply, and realized-value metric families. | Known standalone prefixes such as price, realized, investor, cost, invested, subsidy, supply, market, sopr, mvrv, and similar domains. |
| `halving_epoch_cohorts` | `1` | `599` | Halving-epoch cohort metrics partitioned by epoch id. | Top-level prefix is epoch with epoch_<n>_* metric patterns. |
| `holder_cohorts` | `2` | `842` | Short-term and long-term holder cohort metrics. | Top-level prefix is sth or lth. |
| `mining_pool_metrics` | `157` | `4867` | Per-pool mining production, dominance, fee, coinbase, and subsidy metrics. | Exact top-level family exposes the mining-pool signature (blocks_mined, dominance, coinbase, fee, subsidy). |
| `other_standalone_metrics` | `61` | `368` | Fallback class for documented families that do not match a more specific rule. | All remaining top-level families. |
| `script_output_type_cohorts` | `12` | `2042` | Script type and output-type cohort metrics. | Top-level prefix starts with p2 or is unknown, empty, or opreturn. |
| `utxo_cohorts` | `1` | `18339` | UTXO age-bucket cohort metrics. | Top-level prefix is utxos. |
| `vintage_year_cohorts` | `1` | `2135` | Vintage cohort metrics partitioned by originating year. | Top-level prefix is year with year_<yyyy>_* metric patterns. |
| `windowed_return_and_path_metrics` | `14` | `246` | Duration-led rolling return, DCA, lump-sum, and path-statistic metrics. | Top-level prefix matches a duration token such as 1m, 1y, or 10y. |

## Suffix And Unit Registry

| Suffix | Count | Meaning |
| --- | ---: | --- |
| `_usd` | `5847` | USD-denominated value |
| `_sats` | `5312` | Satoshi-denominated value |
| `_cumulative` | `4976` | Cumulative running total |
| `_btc` | `2455` | BTC-denominated value |
| `_ema` | `1804` | Exponential moving average |
| `_ratio` | `578` | Ratio or relative-value transform |
| `_sma` | `360` | Simple moving average |
| `_cents` | `356` | Cent-denominated fiat value |
| `_zscore` | `224` | Z-score normalization |
| `_pct10` | `148` | Percentile-style suffix. |
| `_pct25` | `148` | Percentile-style suffix. |
| `_pct75` | `148` | Percentile-style suffix. |
| `_pct90` | `148` | Percentile-style suffix. |
| `_pct95` | `132` | Percentile-style suffix. |
| `_pct98` | `84` | Percentile-style suffix. |
| `_pct99` | `84` | Percentile-style suffix. |
| `_pct1` | `76` | Percentile-style suffix. |
| `_pct2` | `76` | Percentile-style suffix. |
| `_pct5` | `76` | Percentile-style suffix. |
| `_pct05` | `48` | Percentile-style suffix. |
| `_pct15` | `48` | Percentile-style suffix. |
| `_pct20` | `48` | Percentile-style suffix. |
| `_pct30` | `48` | Percentile-style suffix. |
| `_pct35` | `48` | Percentile-style suffix. |
| `_pct40` | `48` | Percentile-style suffix. |
| `_pct45` | `48` | Percentile-style suffix. |
| `_pct50` | `48` | Percentile-style suffix. |
| `_pct55` | `48` | Percentile-style suffix. |
| `_pct60` | `48` | Percentile-style suffix. |
| `_pct65` | `48` | Percentile-style suffix. |
| `_pct70` | `48` | Percentile-style suffix. |
| `_pct80` | `48` | Percentile-style suffix. |
| `_pct85` | `48` | Percentile-style suffix. |

## Top-Level Families

| Family | Metric count |
| --- | ---: |
| `utxos` | `18339` |
| `price` | `6050` |
| `addrs` | `5249` |
| `year` | `2135` |
| `epoch` | `599` |
| `sth` | `425` |
| `lth` | `417` |
| `p2a` | `201` |
| `p2pk33` | `201` |
| `p2sh` | `201` |
| `p2tr` | `201` |
| `p2wpkh` | `201` |
| `p2wsh` | `201` |
| `p2pk65` | `200` |
| `p2pkh` | `200` |
| `investor` | `189` |
| `unknown` | `146` |
| `p2ms` | `126` |
| `dca` | `120` |
| `empty` | `118` |
| `block` | `49` |
| `address` | `48` |
| `opreturn` | `46` |
| `invested` | `42` |
| `realized` | `40` |
| `cost` | `37` |
| `subsidy` | `32` |
| `aaopool` | `31` |
| `antpool` | `31` |
| `arkpool` | `31` |
| `asicminer` | `31` |
| `axbt` | `31` |
| `batpool` | `31` |
| `bcmonster` | `31` |
| `bcpoolio` | `31` |
| `binancepool` | `31` |
| `bitalo` | `31` |
| `bitclub` | `31` |
| `bitcoinaffiliatenetwork` | `31` |
| `bitcoincom` | `31` |
| `bitcoinindia` | `31` |
| `bitcoinrussia` | `31` |
| `bitcoinukraine` | `31` |
| `bitfarms` | `31` |
| `bitfufupool` | `31` |
| `bitfury` | `31` |
| `bitminter` | `31` |
| `bitparking` | `31` |
| `bitsolo` | `31` |
| `bixin` | `31` |
| `blockfills` | `31` |
| `braiinspool` | `31` |
| `bravomining` | `31` |
| `btcc` | `31` |
| `btccom` | `31` |
| `btcdig` | `31` |
| `btcguild` | `31` |
| `btclab` | `31` |
| `btcmp` | `31` |
| `btcnuggets` | `31` |
| `btcpoolparty` | `31` |
| `btcserv` | `31` |
| `btctop` | `31` |
| `btpool` | `31` |
| `bwpool` | `31` |
| `bytepool` | `31` |
| `canoe` | `31` |
| `canoepool` | `31` |
| `carbonnegative` | `31` |
| `ckpool` | `31` |
| `cloudhashing` | `31` |
| `coinlab` | `31` |
| `cointerra` | `31` |
| `cointime` | `31` |
| `connectbtc` | `31` |
| `dcex` | `31` |
| `dcexploration` | `31` |
| `digitalbtc` | `31` |
| `digitalxmintsy` | `31` |
| `dpool` | `31` |
| `eclipsemc` | `31` |
| `eightbaochi` | `31` |
| `ekanembtc` | `31` |
| `eligius` | `31` |
| `emcdpool` | `31` |
| `entrustcharitypool` | `31` |
| `eobot` | `31` |
| `exxbw` | `31` |
| `f2pool` | `31` |
| `fiftyeightcoin` | `31` |
| `foundryusa` | `31` |
| `futurebitapollosolo` | `31` |
| `gbminers` | `31` |
| `ghashio` | `31` |
| `givemecoins` | `31` |
| `gogreenlight` | `31` |
| `haominer` | `31` |
| `haozhuzhu` | `31` |
| `hashbx` | `31` |
| `hashpool` | `31` |
| `helix` | `31` |
| `hhtt` | `31` |
| `hotpool` | `31` |
| `hummerpool` | `31` |
| `huobipool` | `31` |
| `innopolistech` | `31` |
| `kanopool` | `31` |
| `kncminer` | `31` |
| `kucoinpool` | `31` |
| `lubiancom` | `31` |
| `luckypool` | `31` |
| `luxor` | `31` |
| `marapool` | `31` |
| `maxbtc` | `31` |
| `maxipool` | `31` |
| `megabigpower` | `31` |
| `minerium` | `31` |
| `miningcity` | `31` |
| `miningdutch` | `31` |
| `miningkings` | `31` |
| `miningsquared` | `31` |
| `mmpool` | `31` |
| `mtred` | `31` |
| `multicoinco` | `31` |
| `multipool` | `31` |
| `mybtccoinpool` | `31` |
| `neopool` | `31` |
| `nexious` | `31` |
| `nicehash` | `31` |
| `nmcbit` | `31` |
| `novablock` | `31` |
| `ocean` | `31` |
| `okexpool` | `31` |
| `okkong` | `31` |
| `okminer` | `31` |
| `okpooltop` | `31` |
| `onehash` | `31` |
| `onem1x` | `31` |
| `onethash` | `31` |
| `ozcoin` | `31` |
| `parasite` | `31` |
| `patels` | `31` |
| `pegapool` | `31` |
| `phashio` | `31` |
| `phoenix` | `31` |
| `polmine` | `31` |
| `pool175btc` | `31` |
| `pool50btc` | `31` |
| `poolin` | `31` |
| `portlandhodl` | `31` |
| `publicpool` | `31` |
| `purebtccom` | `31` |
| `rawpool` | `31` |
| `rigpool` | `31` |
| `sbicrypto` | `31` |
| `secpool` | `31` |
| `secretsuperstar` | `31` |
| `sevenpool` | `31` |
| `shawnp0wers` | `31` |
| `sigmapoolcom` | `31` |
| `simplecoinus` | `31` |
| `solock` | `31` |
| `spiderpool` | `31` |
| `stminingcorp` | `31` |
| `tangpool` | `31` |
| `tatmaspool` | `31` |
| `tbdice` | `31` |
| `telco214` | `31` |
| `terrapool` | `31` |
| `tiger` | `31` |
| `tigerpoolnet` | `31` |
| `titan` | `31` |
| `transactioncoinmining` | `31` |
| `trickysbtcpool` | `31` |
| `triplemining` | `31` |
| `twentyoneinc` | `31` |
| `ultimuspool` | `31` |
| `unomp` | `31` |
| `viabtc` | `31` |
| `waterhole` | `31` |
| `wayicn` | `31` |
| `whitepool` | `31` |
| `wk057` | `31` |
| `yourbtcnet` | `31` |
| `zulupool` | `31` |
| `coinbase` | `30` |
| `sent` | `30` |
| `active` | `25` |
| `vaulted` | `25` |
| `tx` | `23` |
| `2y` | `21` |
| `3y` | `21` |
| `4y` | `21` |
| `5y` | `21` |
| `6y` | `21` |
| `8y` | `21` |
| `true` | `21` |
| `10y` | `20` |
| `1m` | `20` |
| `1w` | `20` |
| `1y` | `20` |
| `3m` | `19` |
| `6m` | `19` |
| `fee` | `19` |
| `constant` | `18` |
| `hash` | `17` |
| `supply` | `14` |
| `net` | `12` |
| `segwit` | `12` |
| `emptyoutput` | `10` |
| `new` | `10` |
| `unknownoutput` | `10` |
| `growth` | `8` |
| `unrealized` | `8` |
| `downside` | `7` |
| `rsi` | `7` |
| `coinblocks` | `6` |
| `unclaimed` | `6` |
| `unspendable` | `6` |
| `adjusted` | `5` |
| `input` | `5` |
| `neg` | `5` |
| `output` | `5` |
| `stoch` | `5` |
| `max` | `4` |
| `oracle` | `4` |
| `` | `3` |
| `annualized` | `3` |
| `circulating` | `3` |
| `days` | `3` |
| `difficulty` | `3` |
| `macd` | `3` |
| `profit` | `3` |
| `received` | `3` |
| `sell` | `3` |
| `sopr` | `3` |
| `sortino` | `3` |
| `total` | `3` |
| `vocdd` | `3` |
| `addr` | `2` |
| `blocks` | `2` |
| `coindays` | `2` |
| `loss` | `2` |
| `lower` | `2` |
| `market` | `2` |
| `spot` | `2` |
| `taproot` | `2` |
| `upper` | `2` |
| `utxo` | `2` |
| `value` | `2` |
| `1d` | `1` |
| `24h` | `1` |
| `activity` | `1` |
| `btc` | `1` |
| `cap` | `1` |
| `capitulation` | `1` |
| `dateindex` | `1` |
| `difficultyepoch` | `1` |
| `exact` | `1` |
| `first` | `1` |
| `gini` | `1` |
| `greed` | `1` |
| `halvingepoch` | `1` |
| `height` | `1` |
| `hodl` | `1` |
| `inflation` | `1` |
| `inputs` | `1` |
| `liveliness` | `1` |
| `min` | `1` |
| `monthindex` | `1` |
| `mvrv` | `1` |
| `nupl` | `1` |
| `nvt` | `1` |
| `outputs` | `1` |
| `pain` | `1` |
| `peak` | `1` |
| `pi` | `1` |
| `puell` | `1` |
| `reserve` | `1` |
| `thermo` | `1` |
| `usd` | `1` |
| `vaultedness` | `1` |
| `weekindex` | `1` |
| `years` | `1` |

## Namespace Registry

This table is exhaustive at the namespace/family level. The JSON artifact at `data/brk_merged_metrics_taxonomy.json` carries the full metric lists.

| Family | Class | Metric count | Pattern | Examples |
| --- | --- | ---: | --- | --- |
| `` | `other_standalone_metrics` | `3` | `_<measure>` | `_30d_change`, `_30d_change_btc`, `_30d_change_usd` |
| `10y` | `windowed_return_and_path_metrics` | `20` | `10y_<measure>` | `10y_cagr`, `10y_dca_average_price`, `10y_dca_average_price_sats` |
| `1d` | `windowed_return_and_path_metrics` | `1` | `1d_<measure>` | `1d_price_returns` |
| `1m` | `windowed_return_and_path_metrics` | `20` | `1m_<measure>` | `1m_block_count`, `1m_dca_average_price`, `1m_dca_average_price_sats` |
| `1w` | `windowed_return_and_path_metrics` | `20` | `1w_<measure>` | `1w_block_count`, `1w_dca_average_price`, `1w_dca_average_price_sats` |
| `1y` | `windowed_return_and_path_metrics` | `20` | `1y_<measure>` | `1y_block_count`, `1y_dca_average_price`, `1y_dca_average_price_sats` |
| `24h` | `windowed_return_and_path_metrics` | `1` | `24h_<measure>` | `24h_block_count` |
| `2y` | `windowed_return_and_path_metrics` | `21` | `2y_<measure>` | `2y_cagr`, `2y_dca_average_price`, `2y_dca_average_price_sats` |
| `3m` | `windowed_return_and_path_metrics` | `19` | `3m_<measure>` | `3m_dca_average_price`, `3m_dca_average_price_sats`, `3m_dca_days_in_loss` |
| `3y` | `windowed_return_and_path_metrics` | `21` | `3y_<measure>` | `3y_cagr`, `3y_dca_average_price`, `3y_dca_average_price_sats` |
| `4y` | `windowed_return_and_path_metrics` | `21` | `4y_<measure>` | `4y_cagr`, `4y_dca_average_price`, `4y_dca_average_price_sats` |
| `5y` | `windowed_return_and_path_metrics` | `21` | `5y_<measure>` | `5y_cagr`, `5y_dca_average_price`, `5y_dca_average_price_sats` |
| `6m` | `windowed_return_and_path_metrics` | `19` | `6m_<measure>` | `6m_dca_average_price`, `6m_dca_average_price_sats`, `6m_dca_days_in_loss` |
| `6y` | `windowed_return_and_path_metrics` | `21` | `6y_<measure>` | `6y_cagr`, `6y_dca_average_price`, `6y_dca_average_price_sats` |
| `8y` | `windowed_return_and_path_metrics` | `21` | `8y_<measure>` | `8y_cagr`, `8y_dca_average_price`, `8y_dca_average_price_sats` |
| `aaopool` | `mining_pool_metrics` | `31` | `aaopool_<measure>` | `aaopool_1m_blocks_mined`, `aaopool_1m_dominance`, `aaopool_1w_blocks_mined` |
| `active` | `other_standalone_metrics` | `25` | `active_<measure>` | `active_cap`, `active_price`, `active_price_ratio` |
| `activity` | `other_standalone_metrics` | `1` | `activity_<measure>` | `activity_to_vaultedness_ratio` |
| `addr` | `other_standalone_metrics` | `2` | `addr_<measure>` | `addr_count`, `addr_count_30d_change` |
| `address` | `address_activity_aggregates` | `48` | `address_<measure>` | `address_activity_balance_decreased_average`, `address_activity_balance_decreased_max`, `address_activity_balance_decreased_median` |
| `addrs` | `address_balance_cohorts` | `5249` | `addrs_<balance_bucket>_<measure>` | `addrs_above_100btc_under_1k_btc__30d_change`, `addrs_above_100btc_under_1k_btc__30d_change_btc`, `addrs_above_100btc_under_1k_btc__30d_change_usd` |
| `adjusted` | `core_market_metrics` | `5` | `adjusted_<measure>` | `adjusted_sopr`, `adjusted_sopr_30d_ema`, `adjusted_sopr_7d_ema` |
| `annualized` | `other_standalone_metrics` | `3` | `annualized_<measure>` | `annualized_volume`, `annualized_volume_btc`, `annualized_volume_usd` |
| `antpool` | `mining_pool_metrics` | `31` | `antpool_<measure>` | `antpool_1m_blocks_mined`, `antpool_1m_dominance`, `antpool_1w_blocks_mined` |
| `arkpool` | `mining_pool_metrics` | `31` | `arkpool_<measure>` | `arkpool_1m_blocks_mined`, `arkpool_1m_dominance`, `arkpool_1w_blocks_mined` |
| `asicminer` | `mining_pool_metrics` | `31` | `asicminer_<measure>` | `asicminer_1m_blocks_mined`, `asicminer_1m_dominance`, `asicminer_1w_blocks_mined` |
| `axbt` | `mining_pool_metrics` | `31` | `axbt_<measure>` | `axbt_1m_blocks_mined`, `axbt_1m_dominance`, `axbt_1w_blocks_mined` |
| `batpool` | `mining_pool_metrics` | `31` | `batpool_<measure>` | `batpool_1m_blocks_mined`, `batpool_1m_dominance`, `batpool_1w_blocks_mined` |
| `bcmonster` | `mining_pool_metrics` | `31` | `bcmonster_<measure>` | `bcmonster_1m_blocks_mined`, `bcmonster_1m_dominance`, `bcmonster_1w_blocks_mined` |
| `bcpoolio` | `mining_pool_metrics` | `31` | `bcpoolio_<measure>` | `bcpoolio_1m_blocks_mined`, `bcpoolio_1m_dominance`, `bcpoolio_1w_blocks_mined` |
| `binancepool` | `mining_pool_metrics` | `31` | `binancepool_<measure>` | `binancepool_1m_blocks_mined`, `binancepool_1m_dominance`, `binancepool_1w_blocks_mined` |
| `bitalo` | `mining_pool_metrics` | `31` | `bitalo_<measure>` | `bitalo_1m_blocks_mined`, `bitalo_1m_dominance`, `bitalo_1w_blocks_mined` |
| `bitclub` | `mining_pool_metrics` | `31` | `bitclub_<measure>` | `bitclub_1m_blocks_mined`, `bitclub_1m_dominance`, `bitclub_1w_blocks_mined` |
| `bitcoinaffiliatenetwork` | `mining_pool_metrics` | `31` | `bitcoinaffiliatenetwork_<measure>` | `bitcoinaffiliatenetwork_1m_blocks_mined`, `bitcoinaffiliatenetwork_1m_dominance`, `bitcoinaffiliatenetwork_1w_blocks_mined` |
| `bitcoincom` | `mining_pool_metrics` | `31` | `bitcoincom_<measure>` | `bitcoincom_1m_blocks_mined`, `bitcoincom_1m_dominance`, `bitcoincom_1w_blocks_mined` |
| `bitcoinindia` | `mining_pool_metrics` | `31` | `bitcoinindia_<measure>` | `bitcoinindia_1m_blocks_mined`, `bitcoinindia_1m_dominance`, `bitcoinindia_1w_blocks_mined` |
| `bitcoinrussia` | `mining_pool_metrics` | `31` | `bitcoinrussia_<measure>` | `bitcoinrussia_1m_blocks_mined`, `bitcoinrussia_1m_dominance`, `bitcoinrussia_1w_blocks_mined` |
| `bitcoinukraine` | `mining_pool_metrics` | `31` | `bitcoinukraine_<measure>` | `bitcoinukraine_1m_blocks_mined`, `bitcoinukraine_1m_dominance`, `bitcoinukraine_1w_blocks_mined` |
| `bitfarms` | `mining_pool_metrics` | `31` | `bitfarms_<measure>` | `bitfarms_1m_blocks_mined`, `bitfarms_1m_dominance`, `bitfarms_1w_blocks_mined` |
| `bitfufupool` | `mining_pool_metrics` | `31` | `bitfufupool_<measure>` | `bitfufupool_1m_blocks_mined`, `bitfufupool_1m_dominance`, `bitfufupool_1w_blocks_mined` |
| `bitfury` | `mining_pool_metrics` | `31` | `bitfury_<measure>` | `bitfury_1m_blocks_mined`, `bitfury_1m_dominance`, `bitfury_1w_blocks_mined` |
| `bitminter` | `mining_pool_metrics` | `31` | `bitminter_<measure>` | `bitminter_1m_blocks_mined`, `bitminter_1m_dominance`, `bitminter_1w_blocks_mined` |
| `bitparking` | `mining_pool_metrics` | `31` | `bitparking_<measure>` | `bitparking_1m_blocks_mined`, `bitparking_1m_dominance`, `bitparking_1w_blocks_mined` |
| `bitsolo` | `mining_pool_metrics` | `31` | `bitsolo_<measure>` | `bitsolo_1m_blocks_mined`, `bitsolo_1m_dominance`, `bitsolo_1w_blocks_mined` |
| `bixin` | `mining_pool_metrics` | `31` | `bixin_<measure>` | `bixin_1m_blocks_mined`, `bixin_1m_dominance`, `bixin_1w_blocks_mined` |
| `block` | `block_aggregates` | `49` | `block_<measure>` | `block_count`, `block_count_cumulative`, `block_count_target` |
| `blockfills` | `mining_pool_metrics` | `31` | `blockfills_<measure>` | `blockfills_1m_blocks_mined`, `blockfills_1m_dominance`, `blockfills_1w_blocks_mined` |
| `blocks` | `other_standalone_metrics` | `2` | `blocks_<measure>` | `blocks_before_next_difficulty_adjustment`, `blocks_before_next_halving` |
| `braiinspool` | `mining_pool_metrics` | `31` | `braiinspool_<measure>` | `braiinspool_1m_blocks_mined`, `braiinspool_1m_dominance`, `braiinspool_1w_blocks_mined` |
| `bravomining` | `mining_pool_metrics` | `31` | `bravomining_<measure>` | `bravomining_1m_blocks_mined`, `bravomining_1m_dominance`, `bravomining_1w_blocks_mined` |
| `btc` | `other_standalone_metrics` | `1` | `btc_<measure>` | `btc_velocity` |
| `btcc` | `mining_pool_metrics` | `31` | `btcc_<measure>` | `btcc_1m_blocks_mined`, `btcc_1m_dominance`, `btcc_1w_blocks_mined` |
| `btccom` | `mining_pool_metrics` | `31` | `btccom_<measure>` | `btccom_1m_blocks_mined`, `btccom_1m_dominance`, `btccom_1w_blocks_mined` |
| `btcdig` | `mining_pool_metrics` | `31` | `btcdig_<measure>` | `btcdig_1m_blocks_mined`, `btcdig_1m_dominance`, `btcdig_1w_blocks_mined` |
| `btcguild` | `mining_pool_metrics` | `31` | `btcguild_<measure>` | `btcguild_1m_blocks_mined`, `btcguild_1m_dominance`, `btcguild_1w_blocks_mined` |
| `btclab` | `mining_pool_metrics` | `31` | `btclab_<measure>` | `btclab_1m_blocks_mined`, `btclab_1m_dominance`, `btclab_1w_blocks_mined` |
| `btcmp` | `mining_pool_metrics` | `31` | `btcmp_<measure>` | `btcmp_1m_blocks_mined`, `btcmp_1m_dominance`, `btcmp_1w_blocks_mined` |
| `btcnuggets` | `mining_pool_metrics` | `31` | `btcnuggets_<measure>` | `btcnuggets_1m_blocks_mined`, `btcnuggets_1m_dominance`, `btcnuggets_1w_blocks_mined` |
| `btcpoolparty` | `mining_pool_metrics` | `31` | `btcpoolparty_<measure>` | `btcpoolparty_1m_blocks_mined`, `btcpoolparty_1m_dominance`, `btcpoolparty_1w_blocks_mined` |
| `btcserv` | `mining_pool_metrics` | `31` | `btcserv_<measure>` | `btcserv_1m_blocks_mined`, `btcserv_1m_dominance`, `btcserv_1w_blocks_mined` |
| `btctop` | `mining_pool_metrics` | `31` | `btctop_<measure>` | `btctop_1m_blocks_mined`, `btctop_1m_dominance`, `btctop_1w_blocks_mined` |
| `btpool` | `mining_pool_metrics` | `31` | `btpool_<measure>` | `btpool_1m_blocks_mined`, `btpool_1m_dominance`, `btpool_1w_blocks_mined` |
| `bwpool` | `mining_pool_metrics` | `31` | `bwpool_<measure>` | `bwpool_1m_blocks_mined`, `bwpool_1m_dominance`, `bwpool_1w_blocks_mined` |
| `bytepool` | `mining_pool_metrics` | `31` | `bytepool_<measure>` | `bytepool_1m_blocks_mined`, `bytepool_1m_dominance`, `bytepool_1w_blocks_mined` |
| `canoe` | `mining_pool_metrics` | `31` | `canoe_<measure>` | `canoe_1m_blocks_mined`, `canoe_1m_dominance`, `canoe_1w_blocks_mined` |
| `canoepool` | `mining_pool_metrics` | `31` | `canoepool_<measure>` | `canoepool_1m_blocks_mined`, `canoepool_1m_dominance`, `canoepool_1w_blocks_mined` |
| `cap` | `other_standalone_metrics` | `1` | `cap_<measure>` | `cap_growth_rate_diff` |
| `capitulation` | `core_market_metrics` | `1` | `capitulation_<measure>` | `capitulation_flow` |
| `carbonnegative` | `mining_pool_metrics` | `31` | `carbonnegative_<measure>` | `carbonnegative_1m_blocks_mined`, `carbonnegative_1m_dominance`, `carbonnegative_1w_blocks_mined` |
| `circulating` | `other_standalone_metrics` | `3` | `circulating_<measure>` | `circulating_supply`, `circulating_supply_btc`, `circulating_supply_usd` |
| `ckpool` | `mining_pool_metrics` | `31` | `ckpool_<measure>` | `ckpool_1m_blocks_mined`, `ckpool_1m_dominance`, `ckpool_1w_blocks_mined` |
| `cloudhashing` | `mining_pool_metrics` | `31` | `cloudhashing_<measure>` | `cloudhashing_1m_blocks_mined`, `cloudhashing_1m_dominance`, `cloudhashing_1w_blocks_mined` |
| `coinbase` | `other_standalone_metrics` | `30` | `coinbase_<measure>` | `coinbase_average`, `coinbase_btc_average`, `coinbase_btc_cumulative` |
| `coinblocks` | `core_market_metrics` | `6` | `coinblocks_<measure>` | `coinblocks_created`, `coinblocks_created_cumulative`, `coinblocks_destroyed` |
| `coindays` | `core_market_metrics` | `2` | `coindays_<measure>` | `coindays_destroyed`, `coindays_destroyed_cumulative` |
| `coinlab` | `mining_pool_metrics` | `31` | `coinlab_<measure>` | `coinlab_1m_blocks_mined`, `coinlab_1m_dominance`, `coinlab_1w_blocks_mined` |
| `cointerra` | `mining_pool_metrics` | `31` | `cointerra_<measure>` | `cointerra_1m_blocks_mined`, `cointerra_1m_dominance`, `cointerra_1w_blocks_mined` |
| `cointime` | `core_market_metrics` | `31` | `cointime_<measure>` | `cointime_adj_inflation_rate`, `cointime_adj_tx_btc_velocity`, `cointime_adj_tx_usd_velocity` |
| `connectbtc` | `mining_pool_metrics` | `31` | `connectbtc_<measure>` | `connectbtc_1m_blocks_mined`, `connectbtc_1m_dominance`, `connectbtc_1w_blocks_mined` |
| `constant` | `other_standalone_metrics` | `18` | `constant_<measure>` | `constant_0`, `constant_1`, `constant_100` |
| `cost` | `core_market_metrics` | `37` | `cost_<measure>` | `cost_basis_pct05`, `cost_basis_pct10`, `cost_basis_pct10_sats` |
| `dateindex` | `other_standalone_metrics` | `1` | `dateindex_<measure>` | `dateindex` |
| `days` | `other_standalone_metrics` | `3` | `days_<measure>` | `days_before_next_difficulty_adjustment`, `days_before_next_halving`, `days_since_price_ath` |
| `dca` | `benchmark_class_metrics` | `120` | `dca_class_<yyyy>_<measure>` | `dca_class_2015_average_price`, `dca_class_2015_average_price_sats`, `dca_class_2015_days_in_loss` |
| `dcex` | `mining_pool_metrics` | `31` | `dcex_<measure>` | `dcex_1m_blocks_mined`, `dcex_1m_dominance`, `dcex_1w_blocks_mined` |
| `dcexploration` | `mining_pool_metrics` | `31` | `dcexploration_<measure>` | `dcexploration_1m_blocks_mined`, `dcexploration_1m_dominance`, `dcexploration_1w_blocks_mined` |
| `difficulty` | `other_standalone_metrics` | `3` | `difficulty_<measure>` | `difficulty`, `difficulty_adjustment`, `difficulty_as_hash` |
| `difficultyepoch` | `other_standalone_metrics` | `1` | `difficultyepoch_<measure>` | `difficultyepoch` |
| `digitalbtc` | `mining_pool_metrics` | `31` | `digitalbtc_<measure>` | `digitalbtc_1m_blocks_mined`, `digitalbtc_1m_dominance`, `digitalbtc_1w_blocks_mined` |
| `digitalxmintsy` | `mining_pool_metrics` | `31` | `digitalxmintsy_<measure>` | `digitalxmintsy_1m_blocks_mined`, `digitalxmintsy_1m_dominance`, `digitalxmintsy_1w_blocks_mined` |
| `downside` | `other_standalone_metrics` | `7` | `downside_<measure>` | `downside_1m_sd_sd`, `downside_1m_sd_sma`, `downside_1w_sd_sd` |
| `dpool` | `mining_pool_metrics` | `31` | `dpool_<measure>` | `dpool_1m_blocks_mined`, `dpool_1m_dominance`, `dpool_1w_blocks_mined` |
| `eclipsemc` | `mining_pool_metrics` | `31` | `eclipsemc_<measure>` | `eclipsemc_1m_blocks_mined`, `eclipsemc_1m_dominance`, `eclipsemc_1w_blocks_mined` |
| `eightbaochi` | `mining_pool_metrics` | `31` | `eightbaochi_<measure>` | `eightbaochi_1m_blocks_mined`, `eightbaochi_1m_dominance`, `eightbaochi_1w_blocks_mined` |
| `ekanembtc` | `mining_pool_metrics` | `31` | `ekanembtc_<measure>` | `ekanembtc_1m_blocks_mined`, `ekanembtc_1m_dominance`, `ekanembtc_1w_blocks_mined` |
| `eligius` | `mining_pool_metrics` | `31` | `eligius_<measure>` | `eligius_1m_blocks_mined`, `eligius_1m_dominance`, `eligius_1w_blocks_mined` |
| `emcdpool` | `mining_pool_metrics` | `31` | `emcdpool_<measure>` | `emcdpool_1m_blocks_mined`, `emcdpool_1m_dominance`, `emcdpool_1w_blocks_mined` |
| `empty` | `script_output_type_cohorts` | `118` | `empty_<measure>` | `empty_addr_count`, `empty_addr_count_30d_change`, `empty_outputs__30d_change` |
| `emptyoutput` | `other_standalone_metrics` | `10` | `emptyoutput_<measure>` | `emptyoutput_count_average`, `emptyoutput_count_cumulative`, `emptyoutput_count_max` |
| `entrustcharitypool` | `mining_pool_metrics` | `31` | `entrustcharitypool_<measure>` | `entrustcharitypool_1m_blocks_mined`, `entrustcharitypool_1m_dominance`, `entrustcharitypool_1w_blocks_mined` |
| `eobot` | `mining_pool_metrics` | `31` | `eobot_<measure>` | `eobot_1m_blocks_mined`, `eobot_1m_dominance`, `eobot_1w_blocks_mined` |
| `epoch` | `halving_epoch_cohorts` | `599` | `epoch_<n>_<measure>` | `epoch_0__30d_change`, `epoch_0__30d_change_btc`, `epoch_0__30d_change_usd` |
| `exact` | `other_standalone_metrics` | `1` | `exact_<measure>` | `exact_utxo_count` |
| `exxbw` | `mining_pool_metrics` | `31` | `exxbw_<measure>` | `exxbw_1m_blocks_mined`, `exxbw_1m_dominance`, `exxbw_1w_blocks_mined` |
| `f2pool` | `mining_pool_metrics` | `31` | `f2pool_<measure>` | `f2pool_1m_blocks_mined`, `f2pool_1m_dominance`, `f2pool_1w_blocks_mined` |
| `fee` | `other_standalone_metrics` | `19` | `fee_<measure>` | `fee_average`, `fee_btc_average`, `fee_btc_cumulative` |
| `fiftyeightcoin` | `mining_pool_metrics` | `31` | `fiftyeightcoin_<measure>` | `fiftyeightcoin_1m_blocks_mined`, `fiftyeightcoin_1m_dominance`, `fiftyeightcoin_1w_blocks_mined` |
| `first` | `other_standalone_metrics` | `1` | `first_<measure>` | `first_height` |
| `foundryusa` | `mining_pool_metrics` | `31` | `foundryusa_<measure>` | `foundryusa_1m_blocks_mined`, `foundryusa_1m_dominance`, `foundryusa_1w_blocks_mined` |
| `futurebitapollosolo` | `mining_pool_metrics` | `31` | `futurebitapollosolo_<measure>` | `futurebitapollosolo_1m_blocks_mined`, `futurebitapollosolo_1m_dominance`, `futurebitapollosolo_1w_blocks_mined` |
| `gbminers` | `mining_pool_metrics` | `31` | `gbminers_<measure>` | `gbminers_1m_blocks_mined`, `gbminers_1m_dominance`, `gbminers_1w_blocks_mined` |
| `ghashio` | `mining_pool_metrics` | `31` | `ghashio_<measure>` | `ghashio_1m_blocks_mined`, `ghashio_1m_dominance`, `ghashio_1w_blocks_mined` |
| `gini` | `other_standalone_metrics` | `1` | `gini_<measure>` | `gini` |
| `givemecoins` | `mining_pool_metrics` | `31` | `givemecoins_<measure>` | `givemecoins_1m_blocks_mined`, `givemecoins_1m_dominance`, `givemecoins_1w_blocks_mined` |
| `gogreenlight` | `mining_pool_metrics` | `31` | `gogreenlight_<measure>` | `gogreenlight_1m_blocks_mined`, `gogreenlight_1m_dominance`, `gogreenlight_1w_blocks_mined` |
| `greed` | `core_market_metrics` | `1` | `greed_<measure>` | `greed_index` |
| `growth` | `other_standalone_metrics` | `8` | `growth_<measure>` | `growth_rate_average`, `growth_rate_max`, `growth_rate_median` |
| `halvingepoch` | `other_standalone_metrics` | `1` | `halvingepoch_<measure>` | `halvingepoch` |
| `haominer` | `mining_pool_metrics` | `31` | `haominer_<measure>` | `haominer_1m_blocks_mined`, `haominer_1m_dominance`, `haominer_1w_blocks_mined` |
| `haozhuzhu` | `mining_pool_metrics` | `31` | `haozhuzhu_<measure>` | `haozhuzhu_1m_blocks_mined`, `haozhuzhu_1m_dominance`, `haozhuzhu_1w_blocks_mined` |
| `hash` | `other_standalone_metrics` | `17` | `hash_<measure>` | `hash_price_phs`, `hash_price_phs_min`, `hash_price_rebound` |
| `hashbx` | `mining_pool_metrics` | `31` | `hashbx_<measure>` | `hashbx_1m_blocks_mined`, `hashbx_1m_dominance`, `hashbx_1w_blocks_mined` |
| `hashpool` | `mining_pool_metrics` | `31` | `hashpool_<measure>` | `hashpool_1m_blocks_mined`, `hashpool_1m_dominance`, `hashpool_1w_blocks_mined` |
| `height` | `other_standalone_metrics` | `1` | `height_<measure>` | `height_count` |
| `helix` | `mining_pool_metrics` | `31` | `helix_<measure>` | `helix_1m_blocks_mined`, `helix_1m_dominance`, `helix_1w_blocks_mined` |
| `hhtt` | `mining_pool_metrics` | `31` | `hhtt_<measure>` | `hhtt_1m_blocks_mined`, `hhtt_1m_dominance`, `hhtt_1w_blocks_mined` |
| `hodl` | `other_standalone_metrics` | `1` | `hodl_<measure>` | `hodl_bank` |
| `hotpool` | `mining_pool_metrics` | `31` | `hotpool_<measure>` | `hotpool_1m_blocks_mined`, `hotpool_1m_dominance`, `hotpool_1w_blocks_mined` |
| `hummerpool` | `mining_pool_metrics` | `31` | `hummerpool_<measure>` | `hummerpool_1m_blocks_mined`, `hummerpool_1m_dominance`, `hummerpool_1w_blocks_mined` |
| `huobipool` | `mining_pool_metrics` | `31` | `huobipool_<measure>` | `huobipool_1m_blocks_mined`, `huobipool_1m_dominance`, `huobipool_1w_blocks_mined` |
| `inflation` | `other_standalone_metrics` | `1` | `inflation_<measure>` | `inflation_rate` |
| `innopolistech` | `mining_pool_metrics` | `31` | `innopolistech_<measure>` | `innopolistech_1m_blocks_mined`, `innopolistech_1m_dominance`, `innopolistech_1w_blocks_mined` |
| `input` | `other_standalone_metrics` | `5` | `input_<measure>` | `input_count_average`, `input_count_cumulative`, `input_count_max` |
| `inputs` | `other_standalone_metrics` | `1` | `inputs_<measure>` | `inputs_per_sec` |
| `invested` | `core_market_metrics` | `42` | `invested_<measure>` | `invested_capital_in_loss`, `invested_capital_in_loss_pct`, `invested_capital_in_profit` |
| `investor` | `core_market_metrics` | `189` | `investor_<measure>` | `investor_cap`, `investor_price`, `investor_price_cents` |
| `kanopool` | `mining_pool_metrics` | `31` | `kanopool_<measure>` | `kanopool_1m_blocks_mined`, `kanopool_1m_dominance`, `kanopool_1w_blocks_mined` |
| `kncminer` | `mining_pool_metrics` | `31` | `kncminer_<measure>` | `kncminer_1m_blocks_mined`, `kncminer_1m_dominance`, `kncminer_1w_blocks_mined` |
| `kucoinpool` | `mining_pool_metrics` | `31` | `kucoinpool_<measure>` | `kucoinpool_1m_blocks_mined`, `kucoinpool_1m_dominance`, `kucoinpool_1w_blocks_mined` |
| `liveliness` | `other_standalone_metrics` | `1` | `liveliness_<measure>` | `liveliness` |
| `loss` | `core_market_metrics` | `2` | `loss_<measure>` | `loss_value_created`, `loss_value_destroyed` |
| `lower` | `core_market_metrics` | `2` | `lower_<measure>` | `lower_price_band`, `lower_price_band_sats` |
| `lth` | `holder_cohorts` | `417` | `lth_<measure>` | `lth__30d_change`, `lth__30d_change_btc`, `lth__30d_change_usd` |
| `lubiancom` | `mining_pool_metrics` | `31` | `lubiancom_<measure>` | `lubiancom_1m_blocks_mined`, `lubiancom_1m_dominance`, `lubiancom_1w_blocks_mined` |
| `luckypool` | `mining_pool_metrics` | `31` | `luckypool_<measure>` | `luckypool_1m_blocks_mined`, `luckypool_1m_dominance`, `luckypool_1w_blocks_mined` |
| `luxor` | `mining_pool_metrics` | `31` | `luxor_<measure>` | `luxor_1m_blocks_mined`, `luxor_1m_dominance`, `luxor_1w_blocks_mined` |
| `macd` | `other_standalone_metrics` | `3` | `macd_<measure>` | `macd_histogram`, `macd_line`, `macd_signal` |
| `marapool` | `mining_pool_metrics` | `31` | `marapool_<measure>` | `marapool_1m_blocks_mined`, `marapool_1m_dominance`, `marapool_1w_blocks_mined` |
| `market` | `core_market_metrics` | `2` | `market_<measure>` | `market_cap`, `market_cap_growth_rate` |
| `max` | `core_market_metrics` | `4` | `max_<measure>` | `max_cost_basis`, `max_cost_basis_sats`, `max_days_between_price_aths` |
| `maxbtc` | `mining_pool_metrics` | `31` | `maxbtc_<measure>` | `maxbtc_1m_blocks_mined`, `maxbtc_1m_dominance`, `maxbtc_1w_blocks_mined` |
| `maxipool` | `mining_pool_metrics` | `31` | `maxipool_<measure>` | `maxipool_1m_blocks_mined`, `maxipool_1m_dominance`, `maxipool_1w_blocks_mined` |
| `megabigpower` | `mining_pool_metrics` | `31` | `megabigpower_<measure>` | `megabigpower_1m_blocks_mined`, `megabigpower_1m_dominance`, `megabigpower_1w_blocks_mined` |
| `min` | `core_market_metrics` | `1` | `min_<measure>` | `min_cost_basis` |
| `minerium` | `mining_pool_metrics` | `31` | `minerium_<measure>` | `minerium_1m_blocks_mined`, `minerium_1m_dominance`, `minerium_1w_blocks_mined` |
| `miningcity` | `mining_pool_metrics` | `31` | `miningcity_<measure>` | `miningcity_1m_blocks_mined`, `miningcity_1m_dominance`, `miningcity_1w_blocks_mined` |
| `miningdutch` | `mining_pool_metrics` | `31` | `miningdutch_<measure>` | `miningdutch_1m_blocks_mined`, `miningdutch_1m_dominance`, `miningdutch_1w_blocks_mined` |
| `miningkings` | `mining_pool_metrics` | `31` | `miningkings_<measure>` | `miningkings_1m_blocks_mined`, `miningkings_1m_dominance`, `miningkings_1w_blocks_mined` |
| `miningsquared` | `mining_pool_metrics` | `31` | `miningsquared_<measure>` | `miningsquared_1m_blocks_mined`, `miningsquared_1m_dominance`, `miningsquared_1w_blocks_mined` |
| `mmpool` | `mining_pool_metrics` | `31` | `mmpool_<measure>` | `mmpool_1m_blocks_mined`, `mmpool_1m_dominance`, `mmpool_1w_blocks_mined` |
| `monthindex` | `other_standalone_metrics` | `1` | `monthindex_<measure>` | `monthindex` |
| `mtred` | `mining_pool_metrics` | `31` | `mtred_<measure>` | `mtred_1m_blocks_mined`, `mtred_1m_dominance`, `mtred_1w_blocks_mined` |
| `multicoinco` | `mining_pool_metrics` | `31` | `multicoinco_<measure>` | `multicoinco_1m_blocks_mined`, `multicoinco_1m_dominance`, `multicoinco_1w_blocks_mined` |
| `multipool` | `mining_pool_metrics` | `31` | `multipool_<measure>` | `multipool_1m_blocks_mined`, `multipool_1m_dominance`, `multipool_1w_blocks_mined` |
| `mvrv` | `core_market_metrics` | `1` | `mvrv_<measure>` | `mvrv` |
| `mybtccoinpool` | `mining_pool_metrics` | `31` | `mybtccoinpool_<measure>` | `mybtccoinpool_1m_blocks_mined`, `mybtccoinpool_1m_dominance`, `mybtccoinpool_1w_blocks_mined` |
| `neg` | `other_standalone_metrics` | `5` | `neg_<measure>` | `neg_realized_loss`, `neg_realized_loss_cumulative`, `neg_unrealized_loss` |
| `neopool` | `mining_pool_metrics` | `31` | `neopool_<measure>` | `neopool_1m_blocks_mined`, `neopool_1m_dominance`, `neopool_1w_blocks_mined` |
| `net` | `core_market_metrics` | `12` | `net_<measure>` | `net_realized_pnl`, `net_realized_pnl_7d_ema`, `net_realized_pnl_cumulative` |
| `new` | `other_standalone_metrics` | `10` | `new_<measure>` | `new_addr_count_average`, `new_addr_count_cumulative`, `new_addr_count_max` |
| `nexious` | `mining_pool_metrics` | `31` | `nexious_<measure>` | `nexious_1m_blocks_mined`, `nexious_1m_dominance`, `nexious_1w_blocks_mined` |
| `nicehash` | `mining_pool_metrics` | `31` | `nicehash_<measure>` | `nicehash_1m_blocks_mined`, `nicehash_1m_dominance`, `nicehash_1w_blocks_mined` |
| `nmcbit` | `mining_pool_metrics` | `31` | `nmcbit_<measure>` | `nmcbit_1m_blocks_mined`, `nmcbit_1m_dominance`, `nmcbit_1w_blocks_mined` |
| `novablock` | `mining_pool_metrics` | `31` | `novablock_<measure>` | `novablock_1m_blocks_mined`, `novablock_1m_dominance`, `novablock_1w_blocks_mined` |
| `nupl` | `core_market_metrics` | `1` | `nupl_<measure>` | `nupl` |
| `nvt` | `other_standalone_metrics` | `1` | `nvt_<measure>` | `nvt` |
| `ocean` | `mining_pool_metrics` | `31` | `ocean_<measure>` | `ocean_1m_blocks_mined`, `ocean_1m_dominance`, `ocean_1w_blocks_mined` |
| `okexpool` | `mining_pool_metrics` | `31` | `okexpool_<measure>` | `okexpool_1m_blocks_mined`, `okexpool_1m_dominance`, `okexpool_1w_blocks_mined` |
| `okkong` | `mining_pool_metrics` | `31` | `okkong_<measure>` | `okkong_1m_blocks_mined`, `okkong_1m_dominance`, `okkong_1w_blocks_mined` |
| `okminer` | `mining_pool_metrics` | `31` | `okminer_<measure>` | `okminer_1m_blocks_mined`, `okminer_1m_dominance`, `okminer_1w_blocks_mined` |
| `okpooltop` | `mining_pool_metrics` | `31` | `okpooltop_<measure>` | `okpooltop_1m_blocks_mined`, `okpooltop_1m_dominance`, `okpooltop_1w_blocks_mined` |
| `onehash` | `mining_pool_metrics` | `31` | `onehash_<measure>` | `onehash_1m_blocks_mined`, `onehash_1m_dominance`, `onehash_1w_blocks_mined` |
| `onem1x` | `mining_pool_metrics` | `31` | `onem1x_<measure>` | `onem1x_1m_blocks_mined`, `onem1x_1m_dominance`, `onem1x_1w_blocks_mined` |
| `onethash` | `mining_pool_metrics` | `31` | `onethash_<measure>` | `onethash_1m_blocks_mined`, `onethash_1m_dominance`, `onethash_1w_blocks_mined` |
| `opreturn` | `script_output_type_cohorts` | `46` | `opreturn_<measure>` | `opreturn_count_average`, `opreturn_count_cumulative`, `opreturn_count_max` |
| `oracle` | `other_standalone_metrics` | `4` | `oracle_<measure>` | `oracle_price_close`, `oracle_price_high`, `oracle_price_low` |
| `output` | `other_standalone_metrics` | `5` | `output_<measure>` | `output_count_average`, `output_count_cumulative`, `output_count_max` |
| `outputs` | `other_standalone_metrics` | `1` | `outputs_<measure>` | `outputs_per_sec` |
| `ozcoin` | `mining_pool_metrics` | `31` | `ozcoin_<measure>` | `ozcoin_1m_blocks_mined`, `ozcoin_1m_dominance`, `ozcoin_1w_blocks_mined` |
| `p2a` | `script_output_type_cohorts` | `201` | `p2a_<measure>` | `p2a__30d_change`, `p2a__30d_change_btc`, `p2a__30d_change_usd` |
| `p2ms` | `script_output_type_cohorts` | `126` | `p2ms_<measure>` | `p2ms__30d_change`, `p2ms__30d_change_btc`, `p2ms__30d_change_usd` |
| `p2pk33` | `script_output_type_cohorts` | `201` | `p2pk33_<measure>` | `p2pk33__30d_change`, `p2pk33__30d_change_btc`, `p2pk33__30d_change_usd` |
| `p2pk65` | `script_output_type_cohorts` | `200` | `p2pk65_<measure>` | `p2pk65__30d_change`, `p2pk65__30d_change_btc`, `p2pk65__30d_change_usd` |
| `p2pkh` | `script_output_type_cohorts` | `200` | `p2pkh_<measure>` | `p2pkh__30d_change`, `p2pkh__30d_change_btc`, `p2pkh__30d_change_usd` |
| `p2sh` | `script_output_type_cohorts` | `201` | `p2sh_<measure>` | `p2sh__30d_change`, `p2sh__30d_change_btc`, `p2sh__30d_change_usd` |
| `p2tr` | `script_output_type_cohorts` | `201` | `p2tr_<measure>` | `p2tr__30d_change`, `p2tr__30d_change_btc`, `p2tr__30d_change_usd` |
| `p2wpkh` | `script_output_type_cohorts` | `201` | `p2wpkh_<measure>` | `p2wpkh__30d_change`, `p2wpkh__30d_change_btc`, `p2wpkh__30d_change_usd` |
| `p2wsh` | `script_output_type_cohorts` | `201` | `p2wsh_<measure>` | `p2wsh__30d_change`, `p2wsh__30d_change_btc`, `p2wsh__30d_change_usd` |
| `pain` | `core_market_metrics` | `1` | `pain_<measure>` | `pain_index` |
| `parasite` | `mining_pool_metrics` | `31` | `parasite_<measure>` | `parasite_1m_blocks_mined`, `parasite_1m_dominance`, `parasite_1w_blocks_mined` |
| `patels` | `mining_pool_metrics` | `31` | `patels_<measure>` | `patels_1m_blocks_mined`, `patels_1m_dominance`, `patels_1w_blocks_mined` |
| `peak` | `core_market_metrics` | `1` | `peak_<measure>` | `peak_regret_rel_to_realized_cap` |
| `pegapool` | `mining_pool_metrics` | `31` | `pegapool_<measure>` | `pegapool_1m_blocks_mined`, `pegapool_1m_dominance`, `pegapool_1w_blocks_mined` |
| `phashio` | `mining_pool_metrics` | `31` | `phashio_<measure>` | `phashio_1m_blocks_mined`, `phashio_1m_dominance`, `phashio_1w_blocks_mined` |
| `phoenix` | `mining_pool_metrics` | `31` | `phoenix_<measure>` | `phoenix_1m_blocks_mined`, `phoenix_1m_dominance`, `phoenix_1w_blocks_mined` |
| `pi` | `other_standalone_metrics` | `1` | `pi_<measure>` | `pi_cycle` |
| `polmine` | `mining_pool_metrics` | `31` | `polmine_<measure>` | `polmine_1m_blocks_mined`, `polmine_1m_dominance`, `polmine_1w_blocks_mined` |
| `pool175btc` | `mining_pool_metrics` | `31` | `pool175btc_<measure>` | `pool175btc_1m_blocks_mined`, `pool175btc_1m_dominance`, `pool175btc_1w_blocks_mined` |
| `pool50btc` | `mining_pool_metrics` | `31` | `pool50btc_<measure>` | `pool50btc_1m_blocks_mined`, `pool50btc_1m_dominance`, `pool50btc_1w_blocks_mined` |
| `poolin` | `mining_pool_metrics` | `31` | `poolin_<measure>` | `poolin_1m_blocks_mined`, `poolin_1m_dominance`, `poolin_1w_blocks_mined` |
| `portlandhodl` | `mining_pool_metrics` | `31` | `portlandhodl_<measure>` | `portlandhodl_1m_blocks_mined`, `portlandhodl_1m_dominance`, `portlandhodl_1w_blocks_mined` |
| `price` | `core_market_metrics` | `6050` | `price_<measure>` | `price_10y_ago`, `price_10y_ago_sats`, `price_111d_sma` |
| `profit` | `core_market_metrics` | `3` | `profit_<measure>` | `profit_flow`, `profit_value_created`, `profit_value_destroyed` |
| `publicpool` | `mining_pool_metrics` | `31` | `publicpool_<measure>` | `publicpool_1m_blocks_mined`, `publicpool_1m_dominance`, `publicpool_1w_blocks_mined` |
| `puell` | `other_standalone_metrics` | `1` | `puell_<measure>` | `puell_multiple` |
| `purebtccom` | `mining_pool_metrics` | `31` | `purebtccom_<measure>` | `purebtccom_1m_blocks_mined`, `purebtccom_1m_dominance`, `purebtccom_1w_blocks_mined` |
| `rawpool` | `mining_pool_metrics` | `31` | `rawpool_<measure>` | `rawpool_1m_blocks_mined`, `rawpool_1m_dominance`, `rawpool_1w_blocks_mined` |
| `realized` | `core_market_metrics` | `40` | `realized_<measure>` | `realized_cap`, `realized_cap_30d_delta`, `realized_cap_cents` |
| `received` | `other_standalone_metrics` | `3` | `received_<measure>` | `received_sum`, `received_sum_btc`, `received_sum_usd` |
| `reserve` | `other_standalone_metrics` | `1` | `reserve_<measure>` | `reserve_risk` |
| `rigpool` | `mining_pool_metrics` | `31` | `rigpool_<measure>` | `rigpool_1m_blocks_mined`, `rigpool_1m_dominance`, `rigpool_1w_blocks_mined` |
| `rsi` | `other_standalone_metrics` | `7` | `rsi_<measure>` | `rsi_14d`, `rsi_14d_max`, `rsi_14d_min` |
| `sbicrypto` | `mining_pool_metrics` | `31` | `sbicrypto_<measure>` | `sbicrypto_1m_blocks_mined`, `sbicrypto_1m_dominance`, `sbicrypto_1w_blocks_mined` |
| `secpool` | `mining_pool_metrics` | `31` | `secpool_<measure>` | `secpool_1m_blocks_mined`, `secpool_1m_dominance`, `secpool_1w_blocks_mined` |
| `secretsuperstar` | `mining_pool_metrics` | `31` | `secretsuperstar_<measure>` | `secretsuperstar_1m_blocks_mined`, `secretsuperstar_1m_dominance`, `secretsuperstar_1w_blocks_mined` |
| `segwit` | `other_standalone_metrics` | `12` | `segwit_<measure>` | `segwit_adoption_cumulative`, `segwit_adoption_sum`, `segwit_count_average` |
| `sell` | `core_market_metrics` | `3` | `sell_<measure>` | `sell_side_risk_ratio`, `sell_side_risk_ratio_30d_ema`, `sell_side_risk_ratio_7d_ema` |
| `sent` | `other_standalone_metrics` | `30` | `sent_<measure>` | `sent`, `sent_14d_ema`, `sent_14d_ema_btc` |
| `sevenpool` | `mining_pool_metrics` | `31` | `sevenpool_<measure>` | `sevenpool_1m_blocks_mined`, `sevenpool_1m_dominance`, `sevenpool_1w_blocks_mined` |
| `shawnp0wers` | `mining_pool_metrics` | `31` | `shawnp0wers_<measure>` | `shawnp0wers_1m_blocks_mined`, `shawnp0wers_1m_dominance`, `shawnp0wers_1w_blocks_mined` |
| `sigmapoolcom` | `mining_pool_metrics` | `31` | `sigmapoolcom_<measure>` | `sigmapoolcom_1m_blocks_mined`, `sigmapoolcom_1m_dominance`, `sigmapoolcom_1w_blocks_mined` |
| `simplecoinus` | `mining_pool_metrics` | `31` | `simplecoinus_<measure>` | `simplecoinus_1m_blocks_mined`, `simplecoinus_1m_dominance`, `simplecoinus_1w_blocks_mined` |
| `solock` | `mining_pool_metrics` | `31` | `solock_<measure>` | `solock_1m_blocks_mined`, `solock_1m_dominance`, `solock_1w_blocks_mined` |
| `sopr` | `core_market_metrics` | `3` | `sopr_<measure>` | `sopr`, `sopr_30d_ema`, `sopr_7d_ema` |
| `sortino` | `other_standalone_metrics` | `3` | `sortino_<measure>` | `sortino_1m`, `sortino_1w`, `sortino_1y` |
| `spiderpool` | `mining_pool_metrics` | `31` | `spiderpool_<measure>` | `spiderpool_1m_blocks_mined`, `spiderpool_1m_dominance`, `spiderpool_1w_blocks_mined` |
| `spot` | `core_market_metrics` | `2` | `spot_<measure>` | `spot_cost_basis_percentile`, `spot_invested_capital_percentile` |
| `sth` | `holder_cohorts` | `425` | `sth_<measure>` | `sth__30d_change`, `sth__30d_change_btc`, `sth__30d_change_usd` |
| `stminingcorp` | `mining_pool_metrics` | `31` | `stminingcorp_<measure>` | `stminingcorp_1m_blocks_mined`, `stminingcorp_1m_dominance`, `stminingcorp_1w_blocks_mined` |
| `stoch` | `other_standalone_metrics` | `5` | `stoch_<measure>` | `stoch_d`, `stoch_k`, `stoch_rsi` |
| `subsidy` | `core_market_metrics` | `32` | `subsidy_<measure>` | `subsidy_average`, `subsidy_btc_average`, `subsidy_btc_cumulative` |
| `supply` | `core_market_metrics` | `14` | `supply_<measure>` | `supply`, `supply_btc`, `supply_halved` |
| `tangpool` | `mining_pool_metrics` | `31` | `tangpool_<measure>` | `tangpool_1m_blocks_mined`, `tangpool_1m_dominance`, `tangpool_1w_blocks_mined` |
| `taproot` | `other_standalone_metrics` | `2` | `taproot_<measure>` | `taproot_adoption_cumulative`, `taproot_adoption_sum` |
| `tatmaspool` | `mining_pool_metrics` | `31` | `tatmaspool_<measure>` | `tatmaspool_1m_blocks_mined`, `tatmaspool_1m_dominance`, `tatmaspool_1w_blocks_mined` |
| `tbdice` | `mining_pool_metrics` | `31` | `tbdice_<measure>` | `tbdice_1m_blocks_mined`, `tbdice_1m_dominance`, `tbdice_1w_blocks_mined` |
| `telco214` | `mining_pool_metrics` | `31` | `telco214_<measure>` | `telco214_1m_blocks_mined`, `telco214_1m_dominance`, `telco214_1w_blocks_mined` |
| `terrapool` | `mining_pool_metrics` | `31` | `terrapool_<measure>` | `terrapool_1m_blocks_mined`, `terrapool_1m_dominance`, `terrapool_1w_blocks_mined` |
| `thermo` | `other_standalone_metrics` | `1` | `thermo_<measure>` | `thermo_cap` |
| `tiger` | `mining_pool_metrics` | `31` | `tiger_<measure>` | `tiger_1m_blocks_mined`, `tiger_1m_dominance`, `tiger_1w_blocks_mined` |
| `tigerpoolnet` | `mining_pool_metrics` | `31` | `tigerpoolnet_<measure>` | `tigerpoolnet_1m_blocks_mined`, `tigerpoolnet_1m_dominance`, `tigerpoolnet_1w_blocks_mined` |
| `titan` | `mining_pool_metrics` | `31` | `titan_<measure>` | `titan_1m_blocks_mined`, `titan_1m_dominance`, `titan_1w_blocks_mined` |
| `total` | `core_market_metrics` | `3` | `total_<measure>` | `total_addr_count`, `total_realized_pnl`, `total_unrealized_pnl` |
| `transactioncoinmining` | `mining_pool_metrics` | `31` | `transactioncoinmining_<measure>` | `transactioncoinmining_1m_blocks_mined`, `transactioncoinmining_1m_dominance`, `transactioncoinmining_1w_blocks_mined` |
| `trickysbtcpool` | `mining_pool_metrics` | `31` | `trickysbtcpool_<measure>` | `trickysbtcpool_1m_blocks_mined`, `trickysbtcpool_1m_dominance`, `trickysbtcpool_1w_blocks_mined` |
| `triplemining` | `mining_pool_metrics` | `31` | `triplemining_<measure>` | `triplemining_1m_blocks_mined`, `triplemining_1m_dominance`, `triplemining_1w_blocks_mined` |
| `true` | `other_standalone_metrics` | `21` | `true_<measure>` | `true_market_mean`, `true_market_mean_ratio`, `true_market_mean_ratio_pct1` |
| `twentyoneinc` | `mining_pool_metrics` | `31` | `twentyoneinc_<measure>` | `twentyoneinc_1m_blocks_mined`, `twentyoneinc_1m_dominance`, `twentyoneinc_1w_blocks_mined` |
| `tx` | `other_standalone_metrics` | `23` | `tx_<measure>` | `tx_count_average`, `tx_count_cumulative`, `tx_count_max` |
| `ultimuspool` | `mining_pool_metrics` | `31` | `ultimuspool_<measure>` | `ultimuspool_1m_blocks_mined`, `ultimuspool_1m_dominance`, `ultimuspool_1w_blocks_mined` |
| `unclaimed` | `other_standalone_metrics` | `6` | `unclaimed_<measure>` | `unclaimed_rewards`, `unclaimed_rewards_btc`, `unclaimed_rewards_btc_cumulative` |
| `unknown` | `script_output_type_cohorts` | `146` | `unknown_<measure>` | `unknown_1m_blocks_mined`, `unknown_1m_dominance`, `unknown_1w_blocks_mined` |
| `unknownoutput` | `other_standalone_metrics` | `10` | `unknownoutput_<measure>` | `unknownoutput_count_average`, `unknownoutput_count_cumulative`, `unknownoutput_count_max` |
| `unomp` | `mining_pool_metrics` | `31` | `unomp_<measure>` | `unomp_1m_blocks_mined`, `unomp_1m_dominance`, `unomp_1w_blocks_mined` |
| `unrealized` | `core_market_metrics` | `8` | `unrealized_<measure>` | `unrealized_loss`, `unrealized_loss_rel_to_market_cap`, `unrealized_loss_rel_to_own_total_unrealized_pnl` |
| `unspendable` | `other_standalone_metrics` | `6` | `unspendable_<measure>` | `unspendable_supply`, `unspendable_supply_btc`, `unspendable_supply_btc_cumulative` |
| `upper` | `core_market_metrics` | `2` | `upper_<measure>` | `upper_price_band`, `upper_price_band_sats` |
| `usd` | `other_standalone_metrics` | `1` | `usd_<measure>` | `usd_velocity` |
| `utxo` | `other_standalone_metrics` | `2` | `utxo_<measure>` | `utxo_count`, `utxo_count_30d_change` |
| `utxos` | `utxo_cohorts` | `18339` | `utxos_<age_bucket>_<measure>` | `utxos_10y_to_12y_old__30d_change`, `utxos_10y_to_12y_old__30d_change_btc`, `utxos_10y_to_12y_old__30d_change_usd` |
| `value` | `core_market_metrics` | `2` | `value_<measure>` | `value_created`, `value_destroyed` |
| `vaulted` | `other_standalone_metrics` | `25` | `vaulted_<measure>` | `vaulted_cap`, `vaulted_price`, `vaulted_price_ratio` |
| `vaultedness` | `other_standalone_metrics` | `1` | `vaultedness_<measure>` | `vaultedness` |
| `viabtc` | `mining_pool_metrics` | `31` | `viabtc_<measure>` | `viabtc_1m_blocks_mined`, `viabtc_1m_dominance`, `viabtc_1w_blocks_mined` |
| `vocdd` | `other_standalone_metrics` | `3` | `vocdd_<measure>` | `vocdd`, `vocdd_365d_median`, `vocdd_cumulative` |
| `waterhole` | `mining_pool_metrics` | `31` | `waterhole_<measure>` | `waterhole_1m_blocks_mined`, `waterhole_1m_dominance`, `waterhole_1w_blocks_mined` |
| `wayicn` | `mining_pool_metrics` | `31` | `wayicn_<measure>` | `wayicn_1m_blocks_mined`, `wayicn_1m_dominance`, `wayicn_1w_blocks_mined` |
| `weekindex` | `other_standalone_metrics` | `1` | `weekindex_<measure>` | `weekindex` |
| `whitepool` | `mining_pool_metrics` | `31` | `whitepool_<measure>` | `whitepool_1m_blocks_mined`, `whitepool_1m_dominance`, `whitepool_1w_blocks_mined` |
| `wk057` | `mining_pool_metrics` | `31` | `wk057_<measure>` | `wk057_1m_blocks_mined`, `wk057_1m_dominance`, `wk057_1w_blocks_mined` |
| `year` | `vintage_year_cohorts` | `2135` | `year_<yyyy>_<measure>` | `year_2009__30d_change`, `year_2009__30d_change_btc`, `year_2009__30d_change_usd` |
| `years` | `other_standalone_metrics` | `1` | `years_<measure>` | `years_since_price_ath` |
| `yourbtcnet` | `mining_pool_metrics` | `31` | `yourbtcnet_<measure>` | `yourbtcnet_1m_blocks_mined`, `yourbtcnet_1m_dominance`, `yourbtcnet_1w_blocks_mined` |
| `zulupool` | `mining_pool_metrics` | `31` | `zulupool_<measure>` | `zulupool_1m_blocks_mined`, `zulupool_1m_dominance`, `zulupool_1w_blocks_mined` |

## Runtime Projection Notes

StackSats runtime does not consume the full metric namespace directly. It projects a small BRK-wide subset into runtime columns such as `date`, `price_usd`, `mvrv`, and selected overlay features.

Canonical runtime subset documentation remains on [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md) and [BRK Data Source](../data-source.md).
