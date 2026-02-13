"""Utilities for loading user-defined strategy classes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import inspect
import sys
from pathlib import Path

from .strategy_types import BaseStrategy


def _load_module(module_or_path: str):
    """Load module from dotted path or `.py` file path."""
    if module_or_path.endswith(".py"):
        file_path = Path(module_or_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")

        module_hash = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:10]
        module_name = f"stacksats_user_strategy_{module_hash}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        # Register before exec so decorators/type introspection can resolve
        # module globals via sys.modules during import-time execution.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    return importlib.import_module(module_or_path)


def load_strategy(
    spec: str,
    *,
    config: dict | None = None,
    config_path: str | None = None,
) -> BaseStrategy:
    """Load and instantiate a strategy from `module_or_path:ClassName`.

    Examples:
    - `my_strategy.py:MyStrategy`
    - `my_package.strategies:MyStrategy`
    """
    module_or_path, sep, class_name = spec.rpartition(":")
    if not sep or not module_or_path or not class_name:
        raise ValueError(
            "Invalid strategy spec. Use format 'module_or_path:ClassName'."
        )

    module = _load_module(module_or_path)
    try:
        strategy_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Class '{class_name}' not found in '{module_or_path}'."
        ) from exc

    merged_config = dict(config or {})
    if config_path is not None:
        cfg_path = Path(config_path).expanduser().resolve()
        merged_config.update(json.loads(cfg_path.read_text(encoding="utf-8")))

    strategy = strategy_cls(**merged_config)
    if not isinstance(strategy, BaseStrategy):
        base_name = f"{BaseStrategy.__module__}.{BaseStrategy.__name__}"
        raise TypeError(
            f"Strategy '{class_name}' must subclass {base_name}."
        )
    if "compute_weights" in strategy_cls.__dict__:
        raise TypeError(
            "Custom compute_weights overrides are not allowed. "
            "Implement propose_weight(state) or "
            "build_target_profile(ctx, features_df, signals) instead."
        )
    has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
    has_profile_hook = (
        strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
    )
    if not (has_propose_hook or has_profile_hook):
        raise TypeError(
            "Strategy must implement propose_weight(state) or "
            "build_target_profile(ctx, features_df, signals)."
        )
    if has_propose_hook:
        propose_weight = getattr(strategy, "propose_weight", None)
        if not callable(propose_weight):
            raise TypeError(f"Strategy '{class_name}' must define callable propose_weight.")
        signature = inspect.signature(propose_weight)
        if list(signature.parameters.keys()) != ["state"]:
            raise TypeError(
                "propose_weight must use signature: "
                "propose_weight(self, state)"
            )
    if has_profile_hook:
        build_target_profile = getattr(strategy, "build_target_profile", None)
        if not callable(build_target_profile):
            raise TypeError(
                f"Strategy '{class_name}' must define callable build_target_profile."
            )
        signature = inspect.signature(build_target_profile)
        if list(signature.parameters.keys()) != ["ctx", "features_df", "signals"]:
            raise TypeError(
                "build_target_profile must use signature: "
                "build_target_profile(self, ctx, features_df, signals)"
            )
    if not getattr(strategy, "strategy_id", None):
        raise ValueError(
            f"Strategy '{class_name}' must define non-empty strategy_id metadata."
        )
    return strategy
