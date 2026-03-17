from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from stacksats.docs_merged_metrics_taxonomy import (
    _is_mining_pool_family,
    build_artifacts_from_parquet,
    build_metric_catalog_from_metrics,
    build_metric_catalog_from_parquet,
    build_taxonomy_from_metrics,
    build_taxonomy_from_parquet,
    catalog_json_path,
    classify_family,
    data_guide_docs_path,
    render_data_guide_docs,
    render_metric_catalog_json,
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


def _catalog_for(metrics: list[str]):
    return build_metric_catalog_from_metrics(
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


def test_taxonomy_exposes_access_categories_and_dimension_registries() -> None:
    metrics = [
        "price_usd",
        "realized_loss_cumulative",
        "supply_btc",
        "sth_adjusted_sopr_7d_ema",
        "utxos_10y_to_12y_old_mvrv",
        "addrs_above_100btc_under_1k_btc__30d_change_usd",
        "year_2024_realized_cap",
        "epoch_4_mvrv",
        "ckpool_blocks_mined",
        "p2wpkh_count_average",
        "block_count_pct90",
        "1y_price_returns",
    ]

    taxonomy = _taxonomy_for(metrics)

    categories = {
        item["label"]: item["metric_count"]
        for item in taxonomy["access_category_registry"]
    }
    assert "Market and valuation" in categories
    assert "Profitability and SOPR" in categories
    assert "Supply, issuance, and scarcity" in categories
    assert "Holder cohorts" in categories
    assert "UTXO age cohorts" in categories
    assert "Address balance cohorts" in categories
    assert "Vintage and halving cohorts" in categories
    assert "Mining pools and miner economics" in categories
    assert "Script and output types" in categories
    assert "Blocks, transactions, and network activity" in categories
    assert "Benchmarks, path metrics, and technical indicators" in categories

    dimension_registries = taxonomy["dimension_registries"]
    assert {"name": "usd", "count": 2} in dimension_registries["units"]
    assert {"name": "btc", "count": 1} in dimension_registries["units"]
    assert {"name": "cumulative", "count": 1} in dimension_registries["transforms"]
    assert {"name": "ema", "count": 1} in dimension_registries["transforms"]
    assert {"name": "change", "count": 1} in dimension_registries["transforms"]
    assert {"name": "pct90", "count": 1} in dimension_registries["statistics"]
    assert {"name": "average", "count": 1} in dimension_registries["statistics"]
    assert {"name": "7d", "count": 1} in dimension_registries["windows"]
    assert {"name": "1y", "count": 1} in dimension_registries["windows"]


def test_metric_catalog_parses_dimensions_and_notes() -> None:
    catalog = _catalog_for(
        [
            "_30d_change_usd",
            "addrs_above_100btc_under_1k_btc__30d_change_usd",
            "price_111d_sma_ratio_1m_sma",
            "emptyoutput_count_average",
            "ckpool_blocks_mined",
        ]
    )
    entries = {item["metric"]: item for item in catalog["metrics"]}

    assert entries["_30d_change_usd"]["unit"] == "usd"
    assert entries["_30d_change_usd"]["transform"] == "change"
    assert "metadata parsing normalizes" in entries["_30d_change_usd"]["notes"]

    addrs_entry = entries["addrs_above_100btc_under_1k_btc__30d_change_usd"]
    assert addrs_entry["cohort_scheme"] == "address_balance"
    assert addrs_entry["window"] == "30d"
    assert addrs_entry["transform"] == "change"
    assert addrs_entry["notes"]

    price_entry = entries["price_111d_sma_ratio_1m_sma"]
    assert price_entry["window"] == "111d, 1m"
    assert price_entry["transform"] == "ratio, sma"

    emptyoutput_entry = entries["emptyoutput_count_average"]
    assert emptyoutput_entry["statistic"] == "average"
    assert "family is grouped with `empty`" in emptyoutput_entry["notes"]

    ckpool_entry = entries["ckpool_blocks_mined"]
    assert ckpool_entry["access_category"] == "Mining pools and miner economics"
    assert ckpool_entry["entity_scope"] == "mining_pool:ckpool"


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
    assert catalog_json_path(tmp_path) == tmp_path / "data" / "brk_merged_metrics_catalog.json"
    assert data_guide_docs_path(tmp_path) == tmp_path / "docs" / "reference" / "merged-metrics-data-guide.md"
    assert resolve_default_parquet_path(tmp_path) == newer


def test_resolve_default_parquet_path_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No merged_metrics"):
        resolve_default_parquet_path(tmp_path)


def test_classify_family_fallback_and_core_market_cases() -> None:
    assert classify_family("price", ["price_usd"]) == "core_market_metrics"
    assert classify_family("mystery", ["mystery_metric"]) == "other_standalone_metrics"
    assert classify_family("p2tr", ["p2tr_outputs"]) == "script_output_type_cohorts"
    assert (
        classify_family(
            "poolx",
            [
                "poolx_blocks_mined",
                "poolx_dominance",
                "poolx_coinbase",
                "poolx_fee",
                "poolx_subsidy",
            ],
        )
        == "mining_pool_metrics"
    )
    assert _is_mining_pool_family("p2tr", ["p2tr_blocks_mined"]) is False


def test_render_taxonomy_catalog_and_guide_include_expected_sections() -> None:
    taxonomy = _taxonomy_for(
        [
            "price_usd",
            "utxos_1d_to_1w_old_mvrv",
            "mystery_metric",
            "market_cap",
            "supply_btc",
            "mvrv",
            "adjusted_sopr",
            "adjusted_sopr_7d_ema",
            "realized_cap_growth_rate",
            "market_cap_growth_rate",
        ]
    )
    catalog = _catalog_for(
        [
            "price_usd",
            "utxos_1d_to_1w_old_mvrv",
            "mystery_metric",
            "market_cap",
            "supply_btc",
            "mvrv",
            "adjusted_sopr",
            "adjusted_sopr_7d_ema",
            "realized_cap_growth_rate",
            "market_cap_growth_rate",
        ]
    )

    docs = render_taxonomy_docs(taxonomy)
    payload = render_taxonomy_json(taxonomy)
    catalog_payload = render_metric_catalog_json(catalog)
    guide = render_data_guide_docs(taxonomy, catalog)

    assert "## User-Facing Access Categories" in docs
    assert "## Metric Dimension Registry" in docs
    assert "`data/brk_merged_metrics_catalog.json`" in docs
    assert '"access_category_registry"' in payload
    assert '"metrics"' in catalog_payload
    assert "## What Data You Can Access" in guide
    assert "## What This Dataset Does Not Contain" in guide
    assert "## Metrics Used By StackSats Runtime" in guide


def test_build_taxonomy_and_catalog_from_parquet_success_path(tmp_path: Path) -> None:
    parquet_path = tmp_path / "merged_metrics_sample.parquet"
    pl.DataFrame(
        {
            "day_utc": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 2),
            ],
            "metric": [
                "price_usd",
                "price_usd",
                "utxos_1d_to_1w_old_mvrv",
            ],
            "value": [1.0, 2.0, 3.0],
        }
    ).write_parquet(parquet_path)

    artifacts = build_artifacts_from_parquet(parquet_path)
    taxonomy = artifacts["taxonomy"]
    catalog = artifacts["catalog"]

    assert taxonomy["dataset_snapshot"]["parquet_name"] == parquet_path.name
    assert taxonomy["dataset_snapshot"]["distinct_metrics"] == 2
    assert catalog["dataset_snapshot"]["parquet_name"] == parquet_path.name

    catalog_lookup = {item["metric"]: item for item in catalog["metrics"]}
    assert catalog_lookup["price_usd"]["coverage_rows"] == 2
    assert catalog_lookup["price_usd"]["first_day"] == "2024-01-01"
    assert catalog_lookup["price_usd"]["last_day"] == "2024-01-02"


def test_build_metric_catalog_from_parquet_success_path(tmp_path: Path) -> None:
    parquet_path = tmp_path / "merged_metrics_sample.parquet"
    pl.DataFrame(
        {
            "day_utc": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "metric": ["price_usd", "utxos_1d_to_1w_old_mvrv"],
            "value": [1.0, 2.0],
        }
    ).write_parquet(parquet_path)

    catalog = build_metric_catalog_from_parquet(parquet_path)
    assert catalog["dataset_snapshot"]["distinct_metrics"] == 2
    assert len(catalog["metrics"]) == 2
