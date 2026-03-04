#!/usr/bin/env python3
"""Validate release-facing docs stay aligned with the changelog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

CHANGELOG_RELEASE_RE = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})$",
    flags=re.MULTILINE,
)
WHATS_NEW_RELEASE_RE = re.compile(
    r"^## (?P<version>\d+\.\d+\.\d+) highlights$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class ReleaseEntry:
    version: str
    date: str


def latest_changelog_release(text: str) -> ReleaseEntry:
    matches = list(CHANGELOG_RELEASE_RE.finditer(text))
    if not matches:
        raise ValueError("CHANGELOG.md does not contain a released version section")
    match = matches[0]
    return ReleaseEntry(version=match.group("version"), date=match.group("date"))


def whats_new_release_versions(text: str) -> list[str]:
    return [match.group("version") for match in WHATS_NEW_RELEASE_RE.finditer(text)]


def validate_release_docs(changelog_text: str, whats_new_text: str) -> list[str]:
    errors: list[str] = []
    latest = latest_changelog_release(changelog_text)
    whats_new_versions = whats_new_release_versions(whats_new_text)

    if not whats_new_versions:
        errors.append("docs/whats-new.md is missing a '<version> highlights' section")
        return errors

    if latest.version not in whats_new_versions:
        errors.append(
            "docs/whats-new.md latest release does not match CHANGELOG.md: "
            f"{latest.version}"
        )

    if whats_new_versions[0] != latest.version:
        errors.append(
            "docs/whats-new.md first release section is not the latest changelog "
            f"release: expected {latest.version}, found {whats_new_versions[0]}"
        )

    return errors


def main() -> int:
    changelog_path = ROOT / "CHANGELOG.md"
    whats_new_path = ROOT / "docs" / "whats-new.md"

    changelog_text = changelog_path.read_text(encoding="utf-8")
    whats_new_text = whats_new_path.read_text(encoding="utf-8")

    errors = validate_release_docs(changelog_text, whats_new_text)
    if errors:
        print("Release docs sync check failed:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("Release docs sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
