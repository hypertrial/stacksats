from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import polars as pl
import pytest

import stacksats.eda as eda
from stacksats.eda import load_metric_catalog, open_merged_metrics


def _write_canonical_parquet(path: Path) -> Path:
    rows = [
        {"day_utc": "2024-01-01", "metric": "market_cap", "value": 100.0},
        {"day_utc": "2024-01-01", "metric": "adjusted_sopr", "value": 1.1},
        {"day_utc": "2024-01-01", "metric": "10y_cagr", "value": 0.5},
        {"day_utc": "2024-01-02", "metric": "market_cap", "value": 105.0},
        {"day_utc": "2024-01-02", "metric": "adjusted_sopr", "value": 1.2},
        {"day_utc": "2024-01-02", "metric": "mvrv", "value": 1.4},
        {"day_utc": "2024-01-03", "metric": "market_cap", "value": 110.0},
        {"day_utc": "2024-01-03", "metric": "10y_cagr", "value": 0.7},
        {"day_utc": "2024-01-03", "metric": "supply_btc", "value": 19_000_000.0},
    ]
    pl.DataFrame(rows).with_columns(pl.col("day_utc").str.to_date()).write_parquet(path)
    return path


def test_open_merged_metrics_uses_explicit_path(tmp_path: Path) -> None:
    parquet_path = _write_canonical_parquet(tmp_path / "merged_metrics.parquet")

    dataset = open_merged_metrics(parquet_path)

    assert dataset.parquet_path == parquet_path.resolve()
    assert dataset.summary()["row_count"] == 9


def test_open_merged_metrics_uses_latest_managed_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = _write_canonical_parquet(tmp_path / "merged_metrics_older.parquet")
    newer = _write_canonical_parquet(tmp_path / "merged_metrics_newer.parquet")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    monkeypatch.setattr(eda, "MANAGED_BRK_DIR", tmp_path)

    dataset = open_merged_metrics()

    assert dataset.parquet_path == newer.resolve()


def test_open_merged_metrics_prefers_canonical_named_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _write_canonical_parquet(tmp_path / "merged_metrics_2024.parquet")
    unrelated = tmp_path / "zzz_other.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [1.0]}).write_parquet(unrelated)
    os.utime(canonical, (1, 1))
    os.utime(unrelated, (2, 2))
    monkeypatch.setattr(eda, "MANAGED_BRK_DIR", tmp_path)

    dataset = open_merged_metrics()

    assert dataset.parquet_path == canonical.resolve()


def test_open_merged_metrics_raises_when_no_canonical_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eda, "MANAGED_BRK_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Run `stacksats data fetch`"):
        open_merged_metrics()


def test_open_merged_metrics_ignores_unrelated_parquet_when_none_are_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = tmp_path / "other.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [1.0]}).write_parquet(unrelated)
    monkeypatch.setattr(eda, "MANAGED_BRK_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Run `stacksats data fetch`"):
        open_merged_metrics()


def test_open_merged_metrics_rejects_runtime_parquet(tmp_path: Path) -> None:
    runtime_path = tmp_path / "bitcoin_analytics.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [1.0]}).write_parquet(runtime_path)

    with pytest.raises(ValueError, match="supports canonical merged_metrics parquet only"):
        open_merged_metrics(runtime_path)


def test_open_merged_metrics_rejects_unsupported_schema(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.parquet"
    pl.DataFrame({"x": [1], "y": [2]}).write_parquet(bad_path)

    with pytest.raises(ValueError, match="Expected canonical merged_metrics columns"):
        open_merged_metrics(bad_path)


def test_dataset_summary_and_collect_shape(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    summary = dataset.summary()
    frame = dataset.collect()

    assert summary == {
        "row_count": 9,
        "distinct_days": 3,
        "distinct_metrics": 5,
        "first_day": "2024-01-01",
        "last_day": "2024-01-03",
        "parquet_path": str(dataset.parquet_path),
    }
    assert frame.columns == ["day_utc", "metric", "value"]
    assert frame.height == 9


def test_head_and_sample_use_current_filtered_slice(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))
    filtered = dataset.filter_dates(start="2024-01-02")

    head = filtered.head(2)
    sample_a = filtered.sample(2, seed=123)
    sample_b = filtered.sample(2, seed=123)

    assert head.height == 2
    assert head["day_utc"].to_list() == [dt.date(2024, 1, 2), dt.date(2024, 1, 2)]
    assert sample_a.equals(sample_b)
    assert set(sample_a["day_utc"].to_list()).issubset({dt.date(2024, 1, 2), dt.date(2024, 1, 3)})


def test_sample_small_preview_does_not_use_full_collect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    def _raise_if_collect(*args: object, **kwargs: object) -> pl.DataFrame:
        raise AssertionError("sample() should not call collect() for bounded previews")

    monkeypatch.setattr(type(dataset), "collect", _raise_if_collect)

    sample = dataset.sample(2, seed=7)

    assert sample.height == 2


def test_filter_dates_is_inclusive_and_chainable(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    filtered = dataset.filter_dates(start="2024-01-02").filter_dates(end="2024-01-02")

    assert filtered.summary()["row_count"] == 3
    assert filtered.summary()["first_day"] == "2024-01-02"
    assert dataset.filter_dates() is dataset


def test_filter_dates_rejects_reversed_bounds(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    with pytest.raises(ValueError, match="start must be on or before end"):
        dataset.filter_dates(start="2024-01-03", end="2024-01-01")


def test_filter_metrics_uses_union_across_selectors(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    filtered = dataset.filter_metrics(
        metrics=["market_cap"],
        prefixes=["adjusted_"],
        regex=r"^10y_",
        families=["mvrv"],
        categories=["Supply, issuance, and scarcity"],
    )

    assert filtered.summary()["distinct_metrics"] == 5
    assert dataset.filter_metrics() is dataset


def test_filter_metrics_validates_unknown_family_and_category(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    with pytest.raises(ValueError, match="Unknown families: bogus"):
        dataset.filter_metrics(families=["bogus"])
    with pytest.raises(ValueError, match="Unknown categories: bogus"):
        dataset.filter_metrics(categories=["bogus"])


def test_available_metrics_and_metric_counts_reflect_current_slice(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))
    filtered = dataset.filter_dates(start="2024-01-02", end="2024-01-03")

    assert filtered.available_metrics() == ["10y_cagr", "adjusted_sopr", "market_cap", "mvrv", "supply_btc"]

    counts = filtered.metric_counts()
    rows = counts.to_dicts()
    assert rows[0]["metric"] == "market_cap"
    assert rows[0]["coverage_rows"] == 2
    assert rows[-1]["metric"] == "supply_btc"


def test_metric_series_sorts_and_validates_metric(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    series = dataset.metric_series("market_cap")

    assert series["day_utc"].to_list() == [
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
    ]
    assert series["value"].to_list() == [100.0, 105.0, 110.0]

    with pytest.raises(ValueError, match="Unknown metric: not_real"):
        dataset.metric_series("not_real")


def test_metric_series_empty_slice_behavior_is_opt_in(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))
    filtered = dataset.filter_dates(start="2024-01-02", end="2024-01-03")

    empty = filtered.metric_series("adjusted_sopr", error_if_empty=False)
    assert empty.height == 1

    later = dataset.filter_dates(start="2024-01-03", end="2024-01-03")
    assert later.metric_series("adjusted_sopr", error_if_empty=False).is_empty()
    with pytest.raises(ValueError, match="Current filtered dataset has no rows for metric 'adjusted_sopr'"):
        later.metric_series("adjusted_sopr", error_if_empty=True)


def test_metric_coverage_uses_current_dataset_window(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    coverage = dataset.filter_dates(start="2024-01-02", end="2024-01-03").metric_coverage()

    rows = {row["metric"]: row for row in coverage.to_dicts()}
    assert rows["market_cap"]["coverage_rows"] == 2
    assert rows["adjusted_sopr"]["coverage_rows"] == 1
    assert rows["market_cap"]["first_day"] == dt.date(2024, 1, 2)


def test_pivot_wide_supports_metric_subset_and_fill_null(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    wide = dataset.pivot_wide(metrics=["market_cap", "10y_cagr"], fill_null=0.0)

    assert wide.columns == ["day_utc", "10y_cagr", "market_cap"]
    assert wide.height == 3
    assert wide.filter(pl.col("day_utc") == dt.date(2024, 1, 2))["10y_cagr"][0] == 0.0
    assert wide.filter(pl.col("day_utc") == dt.date(2024, 1, 2))["market_cap"][0] == 105.0


def test_filter_search_uses_catalog_search_and_empty_query_is_noop(tmp_path: Path) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))

    assert dataset.filter_search("") is dataset

    searched = dataset.filter_search("sopr")
    assert searched.available_metrics() == ["adjusted_sopr"]
    assert searched.summary()["row_count"] == 2

    empty = dataset.filter_search("not a real search query")
    assert empty.summary()["row_count"] == 0


def test_load_metric_catalog_summary_and_lists() -> None:
    catalog = load_metric_catalog()

    summary = catalog.summary()

    assert summary["metric_count"] > 40_000
    assert summary["family_count"] > 200
    assert summary["category_count"] >= 10
    assert "market" in catalog.families()
    assert "Market and valuation" in catalog.categories()


def test_metric_catalog_search_filter_coverage_describe_and_suggest() -> None:
    catalog = load_metric_catalog()

    search = catalog.search("sopr")
    filtered = catalog.filter(
        metrics=["market_cap"],
        prefixes=["adjusted_"],
        regex=r"^10y_",
        families=["mvrv"],
        categories=["Supply, issuance, and scarcity"],
    )
    coverage = catalog.coverage(["market_cap"])
    describe = catalog.describe_metric("adjusted_sopr")
    suggest_exact = catalog.suggest_metrics("market_cap", limit=3)
    suggest_prefix = catalog.suggest_metrics("adjusted sop", limit=5)

    assert "adjusted_sopr" in search["metric"].to_list()
    filtered_metrics = set(filtered["metric"].to_list())
    assert {"market_cap", "adjusted_sopr", "10y_cagr", "mvrv", "supply_btc"}.issubset(
        filtered_metrics
    )
    assert coverage.to_dicts()[0]["metric"] == "market_cap"
    assert describe["metric"] == "adjusted_sopr"
    assert describe["access_category"] == "Profitability and SOPR"
    assert suggest_exact.to_dicts()[0]["metric"] == "market_cap"
    assert any(row["metric"] == "adjusted_sopr" for row in suggest_prefix.to_dicts())


def test_metric_catalog_suggest_ranking_prefers_exact_then_prefix_then_substring() -> None:
    catalog = load_metric_catalog()

    exact = catalog.suggest_metrics("market_cap", limit=5)
    prefix = catalog.suggest_metrics("market", limit=10)
    substring = catalog.suggest_metrics("sopr", limit=10)

    assert exact.to_dicts()[0]["metric"] == "market_cap"
    assert prefix.to_dicts()[0]["metric"].startswith("market")
    assert substring.height > 0
    assert substring["metric"].to_list()[:3] == ["sopr", "sopr_30d_ema", "sopr_7d_ema"]


def test_metric_catalog_describe_metric_rejects_unknown_metric() -> None:
    catalog = load_metric_catalog()

    with pytest.raises(ValueError, match="Unknown metric: not_real"):
        catalog.describe_metric("not_real")


def test_metric_catalog_and_dataset_reject_unknown_explicit_metrics(
    tmp_path: Path,
) -> None:
    dataset = open_merged_metrics(_write_canonical_parquet(tmp_path / "merged_metrics.parquet"))
    catalog = load_metric_catalog()

    with pytest.raises(ValueError, match="Unknown metrics: not_real"):
        catalog.filter(metrics=["not_real"])
    with pytest.raises(ValueError, match="Unknown metrics: not_real"):
        catalog.coverage(["not_real"])
    with pytest.raises(ValueError, match="Unknown metrics: not_real"):
        dataset.filter_metrics(metrics=["not_real"])
    with pytest.raises(ValueError, match="Unknown metrics: not_real"):
        dataset.metric_coverage(["not_real"])
    with pytest.raises(ValueError, match="Unknown metrics: not_real"):
        dataset.pivot_wide(metrics=["not_real"])


def test_packaged_catalog_asset_matches_repo_copy() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_catalog = (repo_root / "data" / "brk_merged_metrics_catalog.json").read_text(
        encoding="utf-8"
    )
    packaged_catalog = (
        repo_root / "stacksats" / "assets" / "brk_merged_metrics_catalog.json"
    ).read_text(encoding="utf-8")

    assert json.loads(repo_catalog) == json.loads(packaged_catalog)
