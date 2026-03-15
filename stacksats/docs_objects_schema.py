"""Helpers for keeping TimeSeries schema docs in sync with code."""

from __future__ import annotations

from pathlib import Path

from .strategy_time_series import TimeSeries

TIMESERIES_SCHEMA_BEGIN = "<!-- BEGIN: TIMESERIES_SCHEMA_TABLE -->"
TIMESERIES_SCHEMA_END = "<!-- END: TIMESERIES_SCHEMA_TABLE -->"
TIMESERIES_LINEAGE_BEGIN = "<!-- BEGIN: TIMESERIES_BRK_LINEAGE -->"
TIMESERIES_LINEAGE_END = "<!-- END: TIMESERIES_BRK_LINEAGE -->"


def objects_docs_path(root_dir: Path | None = None) -> Path:
    """Return generated TimeSeries schema docs path for repository root."""
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
    """Return schema docs content with generated TimeSeries sections."""
    updated = _replace_section(
        content,
        TIMESERIES_SCHEMA_BEGIN,
        TIMESERIES_SCHEMA_END,
        TimeSeries.schema_markdown_table(),
    )
    return _replace_section(
        updated,
        TIMESERIES_LINEAGE_BEGIN,
        TIMESERIES_LINEAGE_END,
        TimeSeries.brk_lineage_markdown(),
    )
