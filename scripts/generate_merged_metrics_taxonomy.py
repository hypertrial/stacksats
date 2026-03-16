#!/usr/bin/env python3
"""Generate merged-metrics taxonomy JSON and docs from the canonical parquet."""

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
        build_taxonomy_from_parquet,
        render_taxonomy_docs,
        render_taxonomy_json,
        resolve_default_parquet_path,
        taxonomy_docs_path,
        taxonomy_json_path,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic taxonomy docs and JSON for the BRK merged_metrics parquet."
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

    taxonomy = build_taxonomy_from_parquet(parquet_path)
    rendered_json = render_taxonomy_json(taxonomy)
    rendered_docs = render_taxonomy_docs(taxonomy)

    if args.check:
        failures: list[str] = []
        if not json_output.exists() or json_output.read_text(encoding="utf-8") != rendered_json:
            failures.append(str(json_output))
        if not doc_output.exists() or doc_output.read_text(encoding="utf-8") != rendered_docs:
            failures.append(str(doc_output))
        if failures:
            joined = ", ".join(failures)
            print(
                f"Merged-metrics taxonomy outputs are out of date: {joined}. "
                "Run: python scripts/generate_merged_metrics_taxonomy.py"
            )
            return 1
        print(
            "Merged-metrics taxonomy outputs are up to date: "
            f"{json_output}, {doc_output}"
        )
        return 0

    json_output.parent.mkdir(parents=True, exist_ok=True)
    doc_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(rendered_json, encoding="utf-8")
    doc_output.write_text(rendered_docs, encoding="utf-8")
    print(f"Updated {json_output}")
    print(f"Updated {doc_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
