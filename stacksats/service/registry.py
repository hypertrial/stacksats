"""Server-side strategy registry for the agent HTTP service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RegisteredStrategy:
    """Resolved server-side strategy registry entry."""

    strategy_id: str
    strategy_spec: str
    strategy_config_path: str | None
    enabled: bool
    btc_price_col: str | None


def load_strategy_registry(path: str | Path) -> dict[str, RegisteredStrategy]:
    """Load and validate the stable JSON service registry."""
    registry_path = Path(path).expanduser()
    if not registry_path.is_absolute():
        registry_path = Path.cwd() / registry_path
    if not registry_path.exists():
        raise FileNotFoundError(f"Strategy registry file not found: {registry_path}")

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Strategy registry JSON must be an object keyed by strategy_id.")

    registry_dir = registry_path.parent
    resolved: dict[str, RegisteredStrategy] = {}
    for strategy_id, entry in payload.items():
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("Strategy registry keys must be non-empty strings.")
        if not isinstance(entry, dict):
            raise ValueError(f"Registry entry '{strategy_id}' must be a JSON object.")
        strategy_spec = entry.get("strategy_spec")
        if not isinstance(strategy_spec, str) or not strategy_spec:
            raise ValueError(
                f"Registry entry '{strategy_id}' must define non-empty 'strategy_spec'."
            )
        strategy_config_path = entry.get("strategy_config_path")
        if strategy_config_path is not None and not isinstance(strategy_config_path, str):
            raise ValueError(
                f"Registry entry '{strategy_id}' has invalid 'strategy_config_path'."
            )
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Registry entry '{strategy_id}' has invalid 'enabled'.")
        btc_price_col = entry.get("btc_price_col")
        if btc_price_col is not None and (
            not isinstance(btc_price_col, str) or not btc_price_col
        ):
            raise ValueError(
                f"Registry entry '{strategy_id}' has invalid 'btc_price_col'."
            )

        resolved[strategy_id] = RegisteredStrategy(
            strategy_id=strategy_id,
            strategy_spec=_resolve_strategy_spec(strategy_spec, registry_dir),
            strategy_config_path=_resolve_optional_path(strategy_config_path, registry_dir),
            enabled=enabled,
            btc_price_col=btc_price_col,
        )
    return resolved


def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_strategy_spec(strategy_spec: str, base_dir: Path) -> str:
    if ":" not in strategy_spec:
        return strategy_spec
    module_or_path, class_name = strategy_spec.split(":", 1)
    candidate = Path(module_or_path).expanduser()
    if candidate.suffix == ".py" and not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
        return f"{candidate}:{class_name}"
    return strategy_spec


__all__ = ["RegisteredStrategy", "load_strategy_registry"]
