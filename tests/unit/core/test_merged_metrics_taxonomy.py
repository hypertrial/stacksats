from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from stacksats.docs_merged_metrics_taxonomy import (
    build_taxonomy_from_metrics,
    build_taxonomy_from_parquet,
)


def _taxonomy_for(metrics: list[str]):
    return build_taxonomy_from_metrics(
        parquet_name="synthetic.parquet",
        row_count=100,
        distinct_days=10,
        distinct_metrics=len(metrics),
        min_day="2020-01-01",
        max_day="2020-01-10",
        metrics=metrics,
    )


def test_taxonomy_classifies_requested_metric_families() -> None:
    metrics = [
        "1m_dca_stack",
        "1y_price_returns",
        "10y_cagr",
        "year_2024_mvrv",
        "epoch_0_mvrv",
        "utxos_10y_to_12y_old_mvrv",
        "addrs_above_100btc_under_1k_btc_mvrv",
        "sth_adjusted_sopr",
        "lth_mvrv",
        "p2wpkh_count_average",
        "unknown_outputs_mvrv",
        "empty_outputs_mvrv",
        "opreturn_count_sum",
        "ckpool_dominance",
        "ckpool_blocks_mined",
        "ckpool_coinbase",
        "ckpool_fee",
        "ckpool_subsidy",
        "foundryusa_dominance",
        "foundryusa_blocks_mined",
        "foundryusa_coinbase",
        "foundryusa_fee",
        "foundryusa_subsidy",
        "address_activity_both_average",
        "block_weight_sum",
        "price_usd",
        "realized_cap",
        "dca_class_2024_stack",
        "mystery_metric",
    ]

    taxonomy = _taxonomy_for(metrics)
    registry = {
        item["family"]: item["semantic_class"]
        for item in taxonomy["namespace_registry"]
    }

    assert registry["1m"] == "windowed_return_and_path_metrics"
    assert registry["1y"] == "windowed_return_and_path_metrics"
    assert registry["10y"] == "windowed_return_and_path_metrics"
    assert registry["year"] == "vintage_year_cohorts"
    assert registry["epoch"] == "halving_epoch_cohorts"
    assert registry["utxos"] == "utxo_cohorts"
    assert registry["addrs"] == "address_balance_cohorts"
    assert registry["sth"] == "holder_cohorts"
    assert registry["lth"] == "holder_cohorts"
    assert registry["p2wpkh"] == "script_output_type_cohorts"
    assert registry["unknown"] == "script_output_type_cohorts"
    assert registry["empty"] == "script_output_type_cohorts"
    assert registry["opreturn"] == "script_output_type_cohorts"
    assert registry["ckpool"] == "mining_pool_metrics"
    assert registry["foundryusa"] == "mining_pool_metrics"
    assert registry["address"] == "address_activity_aggregates"
    assert registry["block"] == "block_aggregates"
    assert registry["price"] == "core_market_metrics"
    assert registry["realized"] == "core_market_metrics"
    assert registry["dca"] == "benchmark_class_metrics"
    assert registry["mystery"] == "other_standalone_metrics"


def test_taxonomy_snapshot_and_suffix_registry_are_deterministic() -> None:
    metrics = [
        "price_usd",
        "price_sats",
        "realized_cap_cumulative",
        "investor_price_ratio",
        "mvrv_zscore",
        "sth_adjusted_sopr_7d_ema",
        "block_size_pct90",
        "block_size_pct10",
        "block_size_pct25",
        "block_size_pct75",
        "cost_basis_pct05",
        "cost_basis_pct95",
    ]

    taxonomy = _taxonomy_for(metrics)
    snapshot = taxonomy["dataset_snapshot"]
    assert snapshot["distinct_metrics"] == len(metrics)
    assert snapshot["top_level_family_count"] == 7

    suffix_counts = {
        item["suffix"]: item["count"] for item in taxonomy["suffix_registry"]
    }
    assert suffix_counts["_usd"] == 1
    assert suffix_counts["_sats"] == 1
    assert suffix_counts["_cumulative"] == 1
    assert suffix_counts["_ratio"] == 1
    assert suffix_counts["_zscore"] == 1
    assert suffix_counts["_ema"] == 1
    assert suffix_counts["_pct10"] == 1
    assert suffix_counts["_pct25"] == 1
    assert suffix_counts["_pct75"] == 1
    assert suffix_counts["_pct90"] == 1
    assert suffix_counts["_pct05"] == 1
    assert suffix_counts["_pct95"] == 1

    namespace_metrics = [
        metric_name
        for item in taxonomy["namespace_registry"]
        for metric_name in item["metrics"]
    ]
    assert sorted(namespace_metrics) == sorted(metrics)
    assert sum(item["metric_count"] for item in taxonomy["semantic_classes"]) == len(metrics)


def test_taxonomy_year_and_epoch_patterns_capture_cohort_values() -> None:
    taxonomy = _taxonomy_for(
        [
            "year_2009_mvrv",
            "year_2024_realized_cap",
            "epoch_0_mvrv",
            "epoch_4_realized_cap",
        ]
    )
    registry = {item["family"]: item for item in taxonomy["namespace_registry"]}

    assert registry["year"]["pattern_summary"]["pattern"] == "year_<yyyy>_<measure>"
    assert registry["year"]["pattern_summary"]["cohort_values"] == [2009, 2024]
    assert registry["epoch"]["pattern_summary"]["pattern"] == "epoch_<n>_<measure>"
    assert registry["epoch"]["pattern_summary"]["cohort_values"] == [0, 4]


def test_build_taxonomy_from_parquet_requires_long_format_schema(tmp_path: Path) -> None:
    parquet_path = tmp_path / "wrong_schema.parquet"
    pl.DataFrame(
        {
            "date": [1],
            "metric_name": ["price_usd"],
            "value": [1.0],
        }
    ).write_parquet(parquet_path)

    with pytest.raises(ValueError, match="requires long-format parquet columns"):
        build_taxonomy_from_parquet(parquet_path)
