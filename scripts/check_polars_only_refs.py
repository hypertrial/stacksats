"""Fail when active repo files contain legacy dataframe-compatibility leftovers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

DEFAULT_ROOTS = ("stacksats", "tests", "docs", "scripts")
DEFAULT_ALLOWLIST = ("CHANGELOG.md",)
_LEGACY_LIB = "".join(["pan", "das"])
LEGACY_TOKENS = (_LEGACY_LIB, "to_" + _LEGACY_LIB, "from_" + _LEGACY_LIB)
PATTERN = re.compile("|".join(re.escape(token) for token in LEGACY_TOKENS), re.IGNORECASE)


@dataclass(frozen=True)
class Match:
    path: str
    line: int
    text: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_allowed(path: Path, allowlist: tuple[str, ...]) -> bool:
    normalized = path.as_posix()
    return any(
        normalized == allowed or normalized.endswith(f"/{allowed}")
        for allowed in allowlist
    )


def find_disallowed_refs(
    root: Path,
    *,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST,
) -> list[Match]:
    """Return all non-allowlisted legacy dataframe references under the target roots."""
    matches: list[Match] = []
    for relative_root in roots:
        base = root / relative_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or _is_allowed(path.relative_to(root), allowlist):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if PATTERN.search(line):
                    matches.append(
                        Match(
                            path=path.relative_to(root).as_posix(),
                            line=line_number,
                            text=line.strip(),
                        )
                    )
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

    matches = find_disallowed_refs(args.root.resolve())
    if not matches:
        print("No legacy dataframe references found in active code/docs/tests/scripts.")
        return 0

    for match in matches:
        print(f"{match.path}:{match.line}: {match.text}")
    print(
        "\nFound legacy dataframe references outside the allowlist. "
        "Migrate them to Polars-native contracts.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
