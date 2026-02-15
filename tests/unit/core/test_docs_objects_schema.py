from __future__ import annotations

from stacksats.docs_objects_schema import objects_docs_path, render_objects_docs
from stacksats.strategy_time_series import StrategyTimeSeries


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
