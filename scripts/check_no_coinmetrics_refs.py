#!/usr/bin/env python3
"""Fail CI if legacy CoinMetrics tokens reappear in active code paths."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_TOKENS: tuple[str, ...] = (
    "coinmetrics",
    "PriceUSD_coinmetrics",
    "CapMVRVCur",
    "coinmetrics_overlay_v1",
    "coinmetrics_btc_csv",
)

# Historical narrative is allowed outside active runtime/test/workflow paths.
ALLOWLIST_PATH_PREFIXES: tuple[str, ...] = (
    "CHANGELOG.md",
    "docs/migration.md",
    "docs/whats-new.md",
)


def _tracked_files() -> list[Path]:
    cmd = ["git", "ls-files"]
    raw = subprocess.check_output(cmd, cwd=ROOT, text=True)
    files: list[Path] = []
    for line in raw.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel.startswith(ALLOWLIST_PATH_PREFIXES):
            continue
        if (
            rel.startswith("stacksats/")
            or rel.startswith("scripts/")
            or rel.startswith("tests/")
            or rel.startswith(".github/workflows/")
        ):
            if rel == "scripts/check_no_coinmetrics_refs.py":
                continue
            files.append(ROOT / rel)
    return files


def main() -> int:
    failures: list[str] = []
    for path in _tracked_files():
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lower = text.lower()
        for token in BANNED_TOKENS:
            token_lower = token.lower()
            if token_lower not in lower:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if token_lower in line.lower():
                    if "check_no_coinmetrics_refs.py" in line:
                        continue
                    rel = path.relative_to(ROOT)
                    failures.append(f"{rel}:{line_no}: contains banned token '{token}'")

    if failures:
        print("CoinMetrics reference guard failed:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("CoinMetrics reference guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
