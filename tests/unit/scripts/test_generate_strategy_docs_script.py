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


generate_strategy_docs = _load_script_module("generate_strategy_docs")


def test_generate_strategy_docs_matches_checked_in_reference() -> None:
    root = _repo_root()
    expected = (root / "docs" / "reference" / "strategies.md").read_text(encoding="utf-8")
    rendered = generate_strategy_docs.generate_strategy_docs()
    assert rendered == expected


def test_generate_strategy_docs_mentions_catalog_ids() -> None:
    rendered = generate_strategy_docs.generate_strategy_docs()
    assert "`simple-zscore`" in rendered
    assert "`mvrv-plus`" in rendered
