from __future__ import annotations

import pytest

from stacksats.docs_objects_schema import objects_docs_path, render_objects_docs
from stacksats.strategy_time_series import CoinMetricsLineageSpec, StrategyTimeSeries


def test_coinmetrics_lineage_targets_documented_columns() -> None:
    StrategyTimeSeries.validate_coinmetrics_lineage_coverage()


def test_coinmetrics_passthrough_columns_have_schema_specs() -> None:
    specs = StrategyTimeSeries.schema_dict()
    for column in StrategyTimeSeries.COINMETRICS_BTC_CSV_COLUMNS:
        assert column in specs


def test_coinmetrics_btc_columns_have_lineage_rows() -> None:
    lineage_sources = {item.source_column for item in StrategyTimeSeries.COINMETRICS_LINEAGE}
    for column in StrategyTimeSeries.COINMETRICS_BTC_CSV_COLUMNS:
        assert column in lineage_sources


def test_objects_docs_schema_sections_are_synced() -> None:
    doc_path = objects_docs_path()
    content = doc_path.read_text(encoding="utf-8")
    assert render_objects_docs(content) == content


def test_render_objects_docs_raises_when_markers_missing() -> None:
    with pytest.raises(ValueError, match="Missing or invalid marker section"):
        render_objects_docs("no schema markers here")


def test_lineage_coverage_raises_when_target_column_not_documented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        StrategyTimeSeries,
        "COINMETRICS_LINEAGE",
        (
            CoinMetricsLineageSpec(
                source_column="time",
                required=True,
                description="timestamp",
                strategy_column="not_in_schema",
            ),
        ),
    )
    with pytest.raises(ValueError, match="reference undocumented StrategyTimeSeries columns"):
        StrategyTimeSeries.validate_coinmetrics_lineage_coverage()


def test_lineage_coverage_raises_when_source_column_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        StrategyTimeSeries,
        "COINMETRICS_BTC_CSV_COLUMNS",
        StrategyTimeSeries.COINMETRICS_BTC_CSV_COLUMNS + ("MissingSourceColumn",),
    )
    with pytest.raises(ValueError, match="CoinMetrics lineage missing BTC CSV source columns"):
        StrategyTimeSeries.validate_coinmetrics_lineage_coverage()
