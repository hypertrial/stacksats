from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_mkdocs_nav_includes_merged_metrics_data_guide() -> None:
    mkdocs_text = (_repo_root() / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Merged Metrics Data Guide: reference/merged-metrics-data-guide.md" in mkdocs_text
    assert "EDA Quickstart: start/eda-quickstart.md" in mkdocs_text
    assert "eda: reference/api/eda.md" in mkdocs_text


def test_core_docs_cross_link_to_new_merged_metrics_data_guide() -> None:
    root = _repo_root()
    schema_text = (
        root / "docs" / "reference" / "merged-metrics-parquet-schema.md"
    ).read_text(encoding="utf-8")
    taxonomy_text = (
        root / "docs" / "reference" / "merged-metrics-taxonomy.md"
    ).read_text(encoding="utf-8")
    data_source_text = (root / "docs" / "data-source.md").read_text(encoding="utf-8")
    eda_quickstart_text = (root / "docs" / "start" / "eda-quickstart.md").read_text(
        encoding="utf-8"
    )

    assert "(merged-metrics-data-guide.md)" in schema_text
    assert "(merged-metrics-data-guide.md)" in taxonomy_text
    assert "(reference/merged-metrics-data-guide.md)" in data_source_text
    assert "(../data-source.md)" in eda_quickstart_text
    assert "(../reference/merged-metrics-data-guide.md)" in eda_quickstart_text
    assert "(../reference/api/eda.md)" in eda_quickstart_text


def test_data_guide_states_available_and_excluded_data() -> None:
    guide_text = (
        _repo_root() / "docs" / "reference" / "merged-metrics-data-guide.md"
    ).read_text(encoding="utf-8")

    assert "## What Data You Can Access" in guide_text
    assert "## What This Dataset Does Not Contain" in guide_text
    assert "other_standalone_metrics" not in guide_text
