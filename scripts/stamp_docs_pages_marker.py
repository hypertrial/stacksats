#!/usr/bin/env python3
"""Inject deterministic deployment markers into built docs HTML."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

COMMIT_META_NAME = "stacksats-docs-commit"
BUILT_AT_META_NAME = "stacksats-docs-built-at"
MARKER_BLOCK_START = "<!-- stacksats-docs-pages-marker:start -->"
MARKER_BLOCK_END = "<!-- stacksats-docs-pages-marker:end -->"
MARKER_BLOCK_RE = re.compile(
    rf"\s*{re.escape(MARKER_BLOCK_START)}.*?{re.escape(MARKER_BLOCK_END)}\s*",
    flags=re.DOTALL,
)
HEAD_CLOSE_RE = re.compile(r"</head\s*>", flags=re.IGNORECASE)


def _build_marker_block(*, commit: str, built_at: str) -> str:
    if not commit.strip():
        raise ValueError("Commit marker must be non-empty.")
    if not built_at.strip():
        raise ValueError("Built-at marker must be non-empty.")
    return (
        f"  {MARKER_BLOCK_START}\n"
        f'  <meta name="{COMMIT_META_NAME}" content="{commit}">\n'
        f'  <meta name="{BUILT_AT_META_NAME}" content="{built_at}">\n'
        f"  {MARKER_BLOCK_END}\n"
    )


def stamp_html(html: str, *, commit: str, built_at: str) -> str:
    marker_block = _build_marker_block(commit=commit, built_at=built_at)
    normalized = MARKER_BLOCK_RE.sub("", html)
    match = HEAD_CLOSE_RE.search(normalized)
    if match is None:
        raise ValueError("HTML document is missing a closing </head> tag.")
    return normalized[: match.start()] + marker_block + normalized[match.start() :]


def stamp_html_file(html_path: Path, *, commit: str, built_at: str) -> None:
    if not html_path.exists():
        raise FileNotFoundError(f"Built docs HTML file not found: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    stamped = stamp_html(html, commit=commit, built_at=built_at)
    html_path.write_text(stamped, encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject commit and build timestamp markers into built docs HTML.",
    )
    parser.add_argument(
        "--html-path",
        default="site/index.html",
        help="Path to the generated docs homepage HTML.",
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="Commit SHA to stamp into the built docs homepage.",
    )
    parser.add_argument(
        "--built-at",
        required=True,
        help="UTC build timestamp to stamp into the built docs homepage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stamp_html_file(
        Path(args.html_path),
        commit=str(args.commit),
        built_at=str(args.built_at),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
