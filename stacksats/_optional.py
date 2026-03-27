"""Helpers for optional dependency loading."""

from __future__ import annotations

import importlib


def import_optional(module_name: str, *, extra: str, feature: str):
    """Import an optional dependency or raise a helpful error."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not exc.name.startswith(f"{module_name}."):
            raise
        raise ImportError(
            f"Missing optional dependency '{module_name}'. "
            f"Install '{extra}' extras with: pip install \"stacksats[{extra}]\" "
            f"to use {feature}."
        ) from exc


def missing_dependency_error(*, dependency: str, extra: str, feature: str) -> ImportError:
    """Build a consistent optional-dependency ImportError."""
    return ImportError(
        f"Missing optional dependency '{dependency}'. "
        f"Install '{extra}' extras with: pip install \"stacksats[{extra}]\" "
        f"to use {feature}."
    )
