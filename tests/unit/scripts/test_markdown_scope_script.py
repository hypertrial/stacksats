from __future__ import annotations

from pathlib import Path
import subprocess


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _markdown_scope() -> list[str]:
    root = _repo_root()
    completed = subprocess.run(
        ["bash", "scripts/check_markdown_scope.sh"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def test_markdown_scope_includes_root_docs_and_github_templates() -> None:
    files = _markdown_scope()

    assert "README.md" in files
    assert "docs/release.md" in files
    assert ".github/pull_request_template.md" in files


def test_markdown_scope_matches_tracked_markdown_files() -> None:
    root = _repo_root()
    completed = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked_files = sorted(line for line in completed.stdout.splitlines() if line)

    assert _markdown_scope() == tracked_files


def test_markdown_scope_is_sorted() -> None:
    files = _markdown_scope()

    assert files == sorted(files)
