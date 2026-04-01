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


new_strategy = _load_script_module("new_strategy")


def test_scaffold_strategy_creates_module_test_and_catalog_stub(tmp_path: Path) -> None:
    catalog_path = tmp_path / "stacksats" / "strategies" / "catalog.py"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        "from dataclasses import dataclass\n\n_CATALOG = (\n)\n\n_CATALOG_BY_ID = {}\n",
        encoding="utf-8",
    )

    module_path, test_path = new_strategy.scaffold_strategy(
        root=tmp_path,
        tier="experimental",
        family="overlays",
        strategy_id="alpha-beta",
        class_name="AlphaBetaStrategy",
        intent="profile",
    )

    assert module_path.exists()
    assert test_path.exists()
    text = catalog_path.read_text(encoding="utf-8")
    assert 'strategy_id="alpha-beta"' in text
    assert 'class_name="AlphaBetaStrategy"' in text
