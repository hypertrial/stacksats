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


guard = _load_script_module("check_polars_only_refs")


def test_find_disallowed_refs_passes_for_current_repo() -> None:
    matches = guard.find_disallowed_refs(_repo_root())
    assert matches == []


def test_find_disallowed_refs_ignores_allowlisted_changelog(tmp_path: Path) -> None:
    legacy = "".join(["pan", "das"])
    (tmp_path / "CHANGELOG.md").write_text(f"{legacy} compatibility note\n", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("Polars only\n", encoding="utf-8")

    matches = guard.find_disallowed_refs(
        tmp_path,
        roots=("docs", "."),
        allowlist=("CHANGELOG.md",),
    )
    assert matches == []


def test_find_disallowed_refs_reports_active_docs_and_tests(tmp_path: Path) -> None:
    legacy = "".join(["pan", "das"])
    docs_dir = tmp_path / "docs"
    tests_dir = tmp_path / "tests"
    docs_dir.mkdir()
    tests_dir.mkdir()
    (docs_dir / "guide.md").write_text(f"This mentions {legacy}.\n", encoding="utf-8")
    (tests_dir / "test_guard.py").write_text(f"from_{legacy} should not appear\n", encoding="utf-8")

    matches = guard.find_disallowed_refs(
        tmp_path,
        roots=("docs", "tests"),
        allowlist=("CHANGELOG.md",),
    )

    assert [(match.path, match.line) for match in matches] == [
        ("docs/guide.md", 1),
        ("tests/test_guard.py", 1),
    ]
