from __future__ import annotations

import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

FACADE_MODULES = [
    "stacksats/api/__init__.py",
    "stacksats/backtest/__init__.py",
    "stacksats/model_development/__init__.py",
    "stacksats/runner/__init__.py",
    "stacksats/statistical_validation/__init__.py",
    "stacksats/strategy_time_series/__init__.py",
    "stacksats/strategy_types/__init__.py",
]

DISALLOWED_IMPORTS = {
    "stacksats/data": ("stacksats.runner", "stacksats.service"),
    "stacksats/features": ("stacksats.runner", "stacksats.service"),
    "stacksats/model_development": ("stacksats.runner", "stacksats.service"),
    "stacksats/strategy_time_series": ("stacksats.service",),
    "stacksats/viz": ("stacksats.runner", "stacksats.service"),
    "stacksats/service": ("stacksats.viz",),
}


def _iter_python_files(package_path: str) -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / package_path).rglob("*.py")
        if "__pycache__" not in path.parts and not path.name.startswith("_legacy")
    )


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    package_parts = _module_name(path).split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            if node.level > 0:
                base_parts = package_parts[:-node.level + 1] if node.level > 1 else package_parts
                module = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if module:
                imported.add(module)
    return imported


def test_facade_modules_are_import_only() -> None:
    for rel_path in FACADE_MODULES:
        path = REPO_ROOT / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        disallowed = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert disallowed == [], f"{rel_path} should stay façade-only"


def test_domain_packages_respect_import_boundaries() -> None:
    for package_path, blocked_prefixes in DISALLOWED_IMPORTS.items():
        for path in _iter_python_files(package_path):
            imports = _imported_modules(path)
            for blocked in blocked_prefixes:
                assert all(
                    imported != blocked and not imported.startswith(f"{blocked}.")
                    for imported in imports
                ), f"{path.relative_to(REPO_ROOT)} must not import {blocked}"


def test_split_module_imports_smoke() -> None:
    modules = [
        "stacksats.api.backtest",
        "stacksats.api.daily",
        "stacksats.api.execution",
        "stacksats.backtest.runtime",
        "stacksats.cli.parser",
        "stacksats.cli.runtime",
        "stacksats.cli.commands.lifecycle",
        "stacksats.cli.commands.daily",
        "stacksats.cli.commands.service",
        "stacksats.eda.catalog",
        "stacksats.eda.dataset",
        "stacksats.eda.parquet",
        "stacksats.runner.adapter",
        "stacksats.runner.backtest",
        "stacksats.runner.core",
        "stacksats.runner.daily",
        "stacksats.runner.export",
        "stacksats.runner.provenance",
        "stacksats.statistical_validation.core",
        "stacksats.strategy_time_series.series",
        "stacksats.strategy_types.config",
        "stacksats.strategy_types.contracts",
        "stacksats.strategy_types.metadata",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
