#!/usr/bin/env python3
"""Generate merged-metrics taxonomy, catalog, and guide artifacts from parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from stacksats.docs_merged_metrics_taxonomy import (
        build_artifacts_from_parquet,
        catalog_json_path,
        data_guide_docs_path,
        packaged_catalog_json_path,
        render_data_guide_docs,
        render_metric_catalog_json,
        render_taxonomy_docs,
        render_taxonomy_json,
        taxonomy_docs_path,
        taxonomy_json_path,
        resolve_default_parquet_path,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic taxonomy, catalog, and guide artifacts for the BRK "
            "merged_metrics parquet."
        )
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Path to merged_metrics parquet. Defaults to the latest merged_metrics*.parquet in repo root.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Path to write taxonomy JSON. Defaults to data/brk_merged_metrics_taxonomy.json.",
    )
    parser.add_argument(
        "--doc-output",
        type=Path,
        default=None,
        help="Path to write taxonomy markdown. Defaults to docs/reference/merged-metrics-taxonomy.md.",
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=None,
        help="Path to write metric catalog JSON. Defaults to data/brk_merged_metrics_catalog.json.",
    )
    parser.add_argument(
        "--guide-output",
        type=Path,
        default=None,
        help="Path to write user-facing data guide markdown. Defaults to docs/reference/merged-metrics-data-guide.md.",
    )
    parser.add_argument(
        "--packaged-catalog-output",
        type=Path,
        default=None,
        help=(
            "Path to write the packaged catalog JSON. Defaults to "
            "stacksats/assets/brk_merged_metrics_catalog.json."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated outputs differ from the committed files.",
    )
    args = parser.parse_args()

    parquet_path = (
        args.parquet_path.expanduser().resolve()
        if args.parquet_path is not None
        else resolve_default_parquet_path(repo_root).resolve()
    )
    json_output = (
        args.json_output.expanduser().resolve()
        if args.json_output is not None
        else taxonomy_json_path(repo_root).resolve()
    )
    doc_output = (
        args.doc_output.expanduser().resolve()
        if args.doc_output is not None
        else taxonomy_docs_path(repo_root).resolve()
    )
    catalog_output = (
        args.catalog_output.expanduser().resolve()
        if args.catalog_output is not None
        else catalog_json_path(repo_root).resolve()
    )
    packaged_catalog_output = (
        args.packaged_catalog_output.expanduser().resolve()
        if args.packaged_catalog_output is not None
        else packaged_catalog_json_path(repo_root).resolve()
    )
    guide_output = (
        args.guide_output.expanduser().resolve()
        if args.guide_output is not None
        else data_guide_docs_path(repo_root).resolve()
    )

    artifacts = build_artifacts_from_parquet(parquet_path)
    taxonomy = artifacts["taxonomy"]
    catalog = artifacts["catalog"]
    rendered_json = render_taxonomy_json(taxonomy)
    rendered_docs = render_taxonomy_docs(taxonomy)
    rendered_catalog = render_metric_catalog_json(catalog)
    rendered_guide = render_data_guide_docs(taxonomy, catalog)

    if args.check:
        failures: list[str] = []
        if not json_output.exists() or json_output.read_text(encoding="utf-8") != rendered_json:
            failures.append(str(json_output))
        if not doc_output.exists() or doc_output.read_text(encoding="utf-8") != rendered_docs:
            failures.append(str(doc_output))
        if not catalog_output.exists() or catalog_output.read_text(encoding="utf-8") != rendered_catalog:
            failures.append(str(catalog_output))
        if (
            not packaged_catalog_output.exists()
            or packaged_catalog_output.read_text(encoding="utf-8") != rendered_catalog
        ):
            failures.append(str(packaged_catalog_output))
        if not guide_output.exists() or guide_output.read_text(encoding="utf-8") != rendered_guide:
            failures.append(str(guide_output))
        if failures:
            joined = ", ".join(failures)
            print(
                f"Merged-metrics generated outputs are out of date: {joined}. "
                "Run: python scripts/generate_merged_metrics_taxonomy.py"
            )
            return 1
        print(
            "Merged-metrics generated outputs are up to date: "
            f"{json_output}, {doc_output}, {catalog_output}, {packaged_catalog_output}, "
            f"{guide_output}"
        )
        return 0

    json_output.parent.mkdir(parents=True, exist_ok=True)
    doc_output.parent.mkdir(parents=True, exist_ok=True)
    catalog_output.parent.mkdir(parents=True, exist_ok=True)
    packaged_catalog_output.parent.mkdir(parents=True, exist_ok=True)
    guide_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(rendered_json, encoding="utf-8")
    doc_output.write_text(rendered_docs, encoding="utf-8")
    catalog_output.write_text(rendered_catalog, encoding="utf-8")
    packaged_catalog_output.write_text(rendered_catalog, encoding="utf-8")
    guide_output.write_text(rendered_guide, encoding="utf-8")
    print(f"Updated {json_output}")
    print(f"Updated {doc_output}")
    print(f"Updated {catalog_output}")
    print(f"Updated {packaged_catalog_output}")
    print(f"Updated {guide_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
