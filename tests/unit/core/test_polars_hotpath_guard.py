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


guard = _load_script_module("check_polars_hotpath_refs")


def test_find_hotpath_refs_passes_for_current_repo() -> None:
    matches = guard.find_hotpath_refs(_repo_root())
    assert matches == []


def test_find_hotpath_refs_reports_non_allowlisted_escape(tmp_path: Path) -> None:
    target = tmp_path / "stacksats"
    target.mkdir()
    hotpath = target / "feature_registry.py"
    hotpath.write_text(
        'frame = pl.read_parquet("x.parquet")\nvalue = df.to_numpy()\n',
        encoding="utf-8",
    )

    matches = guard.find_hotpath_refs(tmp_path)

    assert [(match.path, match.line) for match in matches] == [
        ("stacksats/feature_registry.py", 1),
        ("stacksats/feature_registry.py", 2),
    ]


def test_find_hotpath_refs_allows_explicit_allowlisted_lines(tmp_path: Path) -> None:
    target = tmp_path / "stacksats"
    target.mkdir()
    hotpath = target / "model_development_weights.py"
    hotpath.write_text(
        'raw = merged["_raw"].to_numpy()\n'
        'assert_final_invariants_fn(weights["weight"].to_numpy().astype(float))\n',
        encoding="utf-8",
    )

    matches = guard.find_hotpath_refs(tmp_path)

    assert matches == []
