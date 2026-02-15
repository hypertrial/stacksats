#!/usr/bin/env python3
"""Sync generated schema sections in docs/objects.md."""

from __future__ import annotations

import argparse
import sys

from stacksats.docs_objects_schema import objects_docs_path, render_objects_docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync StrategyTimeSeries schema docs in docs/objects.md."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if docs/objects.md is out of sync.",
    )
    args = parser.parse_args()

    doc_path = objects_docs_path()
    original = doc_path.read_text(encoding="utf-8")
    rendered = render_objects_docs(original)

    if args.check:
        if rendered != original:
            print(
                "docs/objects.md is out of date. "
                "Run: python scripts/sync_objects_schema_docs.py"
            )
            return 1
        print("docs/objects.md schema sections are up to date.")
        return 0

    if rendered != original:
        doc_path.write_text(rendered, encoding="utf-8")
        print(f"Updated {doc_path}")
    else:
        print(f"No changes needed for {doc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

