from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys

import pytest
from setuptools import find_packages


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


def _write_catalog_stub(root: Path) -> Path:
    catalog_path = root / "stacksats" / "strategies" / "catalog.py"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        "from dataclasses import dataclass\n\n_CATALOG = (\n)\n\n_CATALOG_BY_ID = {}\n",
        encoding="utf-8",
    )
    return catalog_path


def _write_strategy_types_stub(root: Path) -> None:
    path = root / "stacksats" / "strategy_types.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "class BaseStrategy:\n"
        "    pass\n\n"
        "class DayState:\n"
        "    pass\n\n"
        "class StrategyContext:\n"
        "    pass\n",
        encoding="utf-8",
    )


new_strategy = _load_script_module("new_strategy")


def test_scaffold_strategy_creates_module_test_catalog_and_package_markers(
    tmp_path: Path,
) -> None:
    catalog_path = _write_catalog_stub(tmp_path)

    module_path, test_path = new_strategy.scaffold_strategy(
        root=tmp_path,
        tier="experimental",
        family="research_models",
        strategy_id="alpha-beta",
        class_name="AlphaBetaStrategy",
        intent="profile",
    )

    assert module_path.exists()
    assert test_path.exists()
    assert (tmp_path / "stacksats" / "__init__.py").exists()
    assert (tmp_path / "stacksats" / "strategies" / "__init__.py").exists()
    assert (tmp_path / "stacksats" / "strategies" / "experimental" / "__init__.py").exists()
    assert (
        tmp_path
        / "stacksats"
        / "strategies"
        / "experimental"
        / "research_models"
        / "__init__.py"
    ).exists()
    assert (tmp_path / "tests" / "__init__.py").exists()
    assert (tmp_path / "tests" / "unit" / "__init__.py").exists()
    assert (tmp_path / "tests" / "unit" / "strategies" / "__init__.py").exists()

    text = catalog_path.read_text(encoding="utf-8")
    assert 'strategy_id="alpha-beta"' in text
    assert 'class_name="AlphaBetaStrategy"' in text
    assert 'family="research_models"' in text


def test_scaffold_strategy_supports_import_for_new_family(
    tmp_path: Path,
) -> None:
    _write_catalog_stub(tmp_path)
    _write_strategy_types_stub(tmp_path)

    new_strategy.scaffold_strategy(
        root=tmp_path,
        tier="stable",
        family="research_models",
        strategy_id="alpha-beta",
        class_name="AlphaBetaStrategy",
        intent="propose",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from stacksats.strategies.stable.research_models.alpha_beta "
                "import AlphaBetaStrategy; print(AlphaBetaStrategy.__name__)"
            ),
        ],
        cwd=str(tmp_path),
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "AlphaBetaStrategy"


def test_scaffolded_family_is_discoverable_by_setuptools(tmp_path: Path) -> None:
    _write_catalog_stub(tmp_path)

    new_strategy.scaffold_strategy(
        root=tmp_path,
        tier="stable",
        family="research_models",
        strategy_id="alpha-beta",
        class_name="AlphaBetaStrategy",
        intent="profile",
    )

    packages = find_packages(where=str(tmp_path), include=["stacksats", "stacksats.*"])
    assert "stacksats.strategies.stable.research_models" in packages


def test_scaffold_strategy_rejects_existing_targets(tmp_path: Path) -> None:
    _write_catalog_stub(tmp_path)

    new_strategy.scaffold_strategy(
        root=tmp_path,
        tier="experimental",
        family="overlays",
        strategy_id="alpha-beta",
        class_name="AlphaBetaStrategy",
        intent="profile",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        new_strategy.scaffold_strategy(
            root=tmp_path,
            tier="experimental",
            family="overlays",
            strategy_id="alpha-beta",
            class_name="AlphaBetaStrategy",
            intent="profile",
        )


def test_scaffold_strategy_requires_catalog_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing strategy catalog"):
        new_strategy.scaffold_strategy(
            root=tmp_path,
            tier="stable",
            family="signals",
            strategy_id="alpha-beta",
            class_name="AlphaBetaStrategy",
            intent="profile",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "tier": "private",
                "family": "signals",
                "strategy_id": "alpha-beta",
                "class_name": "AlphaBetaStrategy",
            },
            "Unsupported tier",
        ),
        (
            {
                "tier": "stable",
                "family": "",
                "strategy_id": "alpha-beta",
                "class_name": "AlphaBetaStrategy",
            },
            "family must be a non-empty snake_case identifier",
        ),
        (
            {
                "tier": "stable",
                "family": "research-models",
                "strategy_id": "alpha-beta",
                "class_name": "AlphaBetaStrategy",
            },
            "family must be a non-empty snake_case identifier",
        ),
        (
            {
                "tier": "stable",
                "family": "signals",
                "strategy_id": "AlphaBeta",
                "class_name": "AlphaBetaStrategy",
            },
            "strategy_id must match",
        ),
        (
            {
                "tier": "stable",
                "family": "signals",
                "strategy_id": "alpha-beta",
                "class_name": "alphaBetaStrategy",
            },
            "class_name must be a valid PascalCase Python class name",
        ),
    ],
)
def test_scaffold_strategy_validates_inputs(
    tmp_path: Path,
    kwargs: dict[str, str],
    message: str,
) -> None:
    _write_catalog_stub(tmp_path)

    with pytest.raises(ValueError, match=message):
        new_strategy.scaffold_strategy(
            root=tmp_path,
            intent="profile",
            **kwargs,
        )
