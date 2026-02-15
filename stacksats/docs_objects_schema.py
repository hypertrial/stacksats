"""Helpers for keeping StrategyTimeSeries schema docs in sync with code."""

from __future__ import annotations

from pathlib import Path

from .strategy_time_series import StrategyTimeSeries

STRATEGY_SCHEMA_BEGIN = "<!-- BEGIN: STRATEGY_TIMESERIES_SCHEMA_TABLE -->"
STRATEGY_SCHEMA_END = "<!-- END: STRATEGY_TIMESERIES_SCHEMA_TABLE -->"
COINMETRICS_LINEAGE_BEGIN = "<!-- BEGIN: STRATEGY_TIMESERIES_COINMETRICS_LINEAGE -->"
COINMETRICS_LINEAGE_END = "<!-- END: STRATEGY_TIMESERIES_COINMETRICS_LINEAGE -->"


def objects_docs_path(root_dir: Path | None = None) -> Path:
    """Return generated StrategyTimeSeries schema docs path for repository root."""
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "docs" / "reference" / "strategy-timeseries-schema.md"


def _replace_section(content: str, begin: str, end: str, body: str) -> str:
    """Replace marker-delimited section body."""
    start = content.find(begin)
    finish = content.find(end)
    if start == -1 or finish == -1 or finish < start:
        raise ValueError(f"Missing or invalid marker section: {begin} .. {end}")
    start_body = start + len(begin)
    rendered = "\n\n" + body.rstrip() + "\n\n"
    return content[:start_body] + rendered + content[finish:]


def render_objects_docs(content: str) -> str:
    """Return schema docs content with generated StrategyTimeSeries sections."""
    updated = _replace_section(
        content,
        STRATEGY_SCHEMA_BEGIN,
        STRATEGY_SCHEMA_END,
        StrategyTimeSeries.schema_markdown_table(),
    )
    return _replace_section(
        updated,
        COINMETRICS_LINEAGE_BEGIN,
        COINMETRICS_LINEAGE_END,
        StrategyTimeSeries.coinmetrics_lineage_markdown(),
    )
