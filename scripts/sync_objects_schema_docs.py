#!/usr/bin/env python3
"""Sync generated schema sections in StrategyTimeSeries schema docs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo_root = _repo_root()
    # Ensure local package imports resolve to this repository, not site-packages.
    sys.path.insert(0, str(repo_root))

    from stacksats.docs_objects_schema import objects_docs_path, render_objects_docs

    parser = argparse.ArgumentParser(
        description=(
            "Sync StrategyTimeSeries schema docs in "
            "docs/reference/strategy-timeseries-schema.md."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Exit non-zero if docs/reference/strategy-timeseries-schema.md is out of sync."
        ),
    )
    args = parser.parse_args()

    doc_path = objects_docs_path(root_dir=repo_root)
    if not doc_path.exists():
        print(
            "Expected schema doc path is missing under repository root: "
            f"{doc_path}"
        )
        return 1
    original = doc_path.read_text(encoding="utf-8")
    rendered = render_objects_docs(original)

    if args.check:
        if rendered != original:
            print(
                f"{doc_path} is out of date. "
                "Run: python scripts/sync_objects_schema_docs.py"
            )
            return 1
        print(f"{doc_path} schema sections are up to date.")
        return 0

    if rendered != original:
        doc_path.write_text(rendered, encoding="utf-8")
        print(f"Updated {doc_path}")
    else:
        print(f"No changes needed for {doc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
