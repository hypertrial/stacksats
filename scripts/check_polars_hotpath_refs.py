"""Fail when core Polars hot-path files add unreviewed eager/row-wise escapes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

HOTPATH_FILES = (
    "stacksats/data_btc.py",
    "stacksats/column_map_provider.py",
    "stacksats/model_development_features.py",
    "stacksats/model_development_helpers.py",
    "stacksats/model_development_weights.py",
    "stacksats/prelude.py",
    "stacksats/feature_providers.py",
    "stacksats/feature_registry.py",
)

PATTERN = re.compile(r"\bto_numpy\s*\(|\biter_rows\s*\(|\bpl\.read_parquet\s*\(")

ALLOWLIST: dict[str, tuple[str, ...]] = {
    "stacksats/model_development_features.py": (
        'df["price_vs_ma"].to_numpy()',
        'df["mvrv_zscore"].to_numpy()',
        'df["mvrv_gradient"].to_numpy()',
        'df["mvrv_percentile"].to_numpy()',
        'df["mvrv_acceleration"].to_numpy()',
        'df["mvrv_volatility"].to_numpy()',
        'df["signal_confidence"].to_numpy()',
    ),
    "stacksats/model_development_helpers.py": (
        "arr = series.to_numpy()",
        "values = window_series.to_numpy()",
        "arr = vol.to_numpy()",
    ),
    "stacksats/model_development_weights.py": (
        'merged["_pref"].fill_null(0.0).to_numpy()',
        'weights["weight"].to_numpy().astype(float)',
    ),
    "stacksats/prelude.py": (
        "dynamic_pct.to_numpy()",
        "uniform_pct.to_numpy()",
    ),
}


@dataclass(frozen=True)
class Match:
    path: str
    line: int
    text: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_hotpath_refs(root: Path) -> list[Match]:
    """Return non-allowlisted eager/row-wise escapes in core Polars hot-path files."""
    matches: list[Match] = []
    for rel_path in HOTPATH_FILES:
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        allowlist = ALLOWLIST.get(rel_path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not PATTERN.search(line):
                continue
            if any(allowed in line for allowed in allowlist):
                continue
            matches.append(Match(path=rel_path, line=line_number, text=line.strip()))
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=_repo_root(),
        help="Repository root to scan.",
    )
    args = parser.parse_args(argv)

    matches = find_hotpath_refs(args.root.resolve())
    if not matches:
        print("No unreviewed Polars hot-path escape hatches found.")
        return 0

    for match in matches:
        print(f"{match.path}:{match.line}: {match.text}")
    print(
        "\nFound non-allowlisted eager or row-wise escapes in core Polars hot-path files.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
