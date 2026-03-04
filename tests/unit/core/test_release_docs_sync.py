from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_script_module(name: str):
    root = _repo_root()
    path = root / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release_docs_sync = _load_script_module("check_release_docs_sync")


def test_validate_release_docs_passes_for_current_repo_files() -> None:
    root = _repo_root()
    changelog_text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    whats_new_text = (root / "docs" / "whats-new.md").read_text(encoding="utf-8")

    assert release_docs_sync.validate_release_docs(changelog_text, whats_new_text) == []


def test_validate_release_docs_fails_when_whats_new_section_is_missing() -> None:
    changelog_text = """
## [Unreleased]

## [0.6.0] - 2026-03-04
"""
    whats_new_text = "# What's New\n\nNo release section here.\n"

    errors = release_docs_sync.validate_release_docs(changelog_text, whats_new_text)

    assert errors == ["docs/whats-new.md is missing a '<version> highlights' section"]


def test_validate_release_docs_fails_when_latest_version_is_not_present() -> None:
    changelog_text = """
## [Unreleased]

## [0.6.0] - 2026-03-04
## [0.5.2] - 2026-02-28
"""
    whats_new_text = """
# What's New

## 0.5.2 highlights
"""

    errors = release_docs_sync.validate_release_docs(changelog_text, whats_new_text)

    assert any("0.6.0" in error for error in errors)


def test_latest_changelog_release_uses_first_released_section() -> None:
    changelog_text = """
## [Unreleased]

## [0.6.0] - 2026-03-04
## [0.5.2] - 2026-02-28
"""

    latest = release_docs_sync.latest_changelog_release(changelog_text)

    assert latest.version == "0.6.0"
    assert latest.date == "2026-03-04"
