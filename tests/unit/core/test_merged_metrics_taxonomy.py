from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from stacksats.docs_merged_metrics_taxonomy import (
    _is_mining_pool_family,
    build_taxonomy_from_metrics,
    build_taxonomy_from_parquet,
    classify_family,
    render_taxonomy_docs,
    render_taxonomy_json,
    resolve_default_parquet_path,
    taxonomy_docs_path,
    taxonomy_json_path,
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


def test_taxonomy_path_helpers_and_default_parquet_resolution(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "docs" / "reference").mkdir(parents=True)
    older = tmp_path / "merged_metrics_2026-01-01.parquet"
    newer = tmp_path / "merged_metrics_2026-02-01.parquet"
    pl.DataFrame({"day_utc": [], "metric": [], "value": []}).write_parquet(older)
    pl.DataFrame({"day_utc": [], "metric": [], "value": []}).write_parquet(newer)

    assert taxonomy_json_path(tmp_path) == tmp_path / "data" / "brk_merged_metrics_taxonomy.json"
    assert taxonomy_docs_path(tmp_path) == tmp_path / "docs" / "reference" / "merged-metrics-taxonomy.md"
    assert resolve_default_parquet_path(tmp_path) == newer


def test_resolve_default_parquet_path_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No merged_metrics"):
        resolve_default_parquet_path(tmp_path)


def test_classify_family_fallback_and_core_market_cases() -> None:
    assert classify_family("price", ["price_usd"]) == "core_market_metrics"
    assert classify_family("mystery", ["mystery_metric"]) == "other_standalone_metrics"
    assert classify_family("p2tr", ["p2tr_outputs"]) == "script_output_type_cohorts"
    assert classify_family(
        "poolx",
        [
            "poolx_blocks_mined",
            "poolx_dominance",
            "poolx_coinbase",
            "poolx_fee",
            "poolx_subsidy",
        ],
    ) == "mining_pool_metrics"
    assert _is_mining_pool_family("p2tr", ["p2tr_blocks_mined"]) is False


def test_render_taxonomy_docs_and_json_include_expected_sections() -> None:
    taxonomy = _taxonomy_for(
        [
            "price_usd",
            "utxos_1d_to_1w_mvrv",
            "mystery_metric",
        ]
    )

    docs = render_taxonomy_docs(taxonomy)
    payload = render_taxonomy_json(taxonomy)

    assert "Dataset scale in the current canonical snapshot" in docs
    assert "## Namespace Registry" in docs
    assert "`data/brk_merged_metrics_taxonomy.json`" in docs
    assert '"dataset_snapshot"' in payload
    assert '"namespace_registry"' in payload


def test_build_taxonomy_from_parquet_success_path(tmp_path: Path) -> None:
    parquet_path = tmp_path / "merged_metrics_sample.parquet"
    pl.DataFrame(
        {
            "day_utc": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "metric": ["price_usd", "utxos_1d_to_1w_mvrv"],
            "value": [1.0, 2.0],
        }
    ).write_parquet(parquet_path)

    taxonomy = build_taxonomy_from_parquet(parquet_path)
    assert taxonomy["dataset_snapshot"]["parquet_name"] == parquet_path.name
    assert taxonomy["dataset_snapshot"]["distinct_metrics"] == 2
