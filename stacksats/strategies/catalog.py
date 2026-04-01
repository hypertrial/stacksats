"""Built-in strategy catalog.

Support tier, public exports, audit inclusion, and docs grouping are defined here
rather than inferred from implementation module paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Literal

from ..strategy_types import BaseStrategy


StrategyTier = Literal["stable", "experimental", "private"]


@dataclass(frozen=True, slots=True)
class StrategyCatalogEntry:
    """Library-management metadata for a built-in strategy."""

    strategy_id: str
    strategy_spec: str
    class_name: str
    module_path: str
    tier: StrategyTier
    public_export: bool
    audit_enabled: bool
    family: str
    description: str
    docs_slug: str
    tags: tuple[str, ...]
    default_validation_config: dict[str, object]
    default_backtest_config: dict[str, object]


_CATALOG: tuple[StrategyCatalogEntry, ...] = (
    StrategyCatalogEntry(
        strategy_id="uniform",
        strategy_spec="stacksats.strategies.stable.baselines.uniform:UniformStrategy",
        class_name="UniformStrategy",
        module_path="stacksats.strategies.stable.baselines.uniform",
        tier="stable",
        public_export=True,
        audit_enabled=True,
        family="baseline",
        description="Uniform baseline allocation across each window.",
        docs_slug="uniform",
        tags=("baseline", "sanity-check"),
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
    StrategyCatalogEntry(
        strategy_id="run-daily-paper",
        strategy_spec=(
            "stacksats.strategies.stable.baselines.run_daily_paper:"
            "RunDailyPaperStrategy"
        ),
        class_name="RunDailyPaperStrategy",
        module_path="stacksats.strategies.stable.baselines.run_daily_paper",
        tier="stable",
        public_export=True,
        audit_enabled=False,
        family="baseline",
        description=(
            "Canonical agent-facing daily decision example with relaxed daily"
            " validation defaults."
        ),
        docs_slug="run-daily-paper",
        tags=("agent", "daily", "paper"),
        default_validation_config={"min_win_rate": 0.0, "strict": False},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
    StrategyCatalogEntry(
        strategy_id="simple-zscore",
        strategy_spec=(
            "stacksats.strategies.stable.signals.simple_zscore:"
            "SimpleZScoreStrategy"
        ),
        class_name="SimpleZScoreStrategy",
        module_path="stacksats.strategies.stable.signals.simple_zscore",
        tier="stable",
        public_export=True,
        audit_enabled=True,
        family="signal",
        description="Preference tilts toward lower mvrv_zscore values.",
        docs_slug="simple-zscore",
        tags=("toy", "profile", "value"),
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
    StrategyCatalogEntry(
        strategy_id="momentum",
        strategy_spec="stacksats.strategies.stable.signals.momentum:MomentumStrategy",
        class_name="MomentumStrategy",
        module_path="stacksats.strategies.stable.signals.momentum",
        tier="stable",
        public_export=True,
        audit_enabled=True,
        family="signal",
        description="Contrarian 30-day momentum preference model.",
        docs_slug="momentum",
        tags=("toy", "profile", "momentum"),
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
    StrategyCatalogEntry(
        strategy_id="mvrv",
        strategy_spec="stacksats.strategies.stable.mvrv.core:MVRVStrategy",
        class_name="MVRVStrategy",
        module_path="stacksats.strategies.stable.mvrv.core",
        tier="stable",
        public_export=True,
        audit_enabled=True,
        family="mvrv",
        description="Core package MVRV and MA preference model.",
        docs_slug="mvrv",
        tags=("baseline", "mvrv", "production"),
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
    StrategyCatalogEntry(
        strategy_id="example-mvrv",
        strategy_spec=(
            "stacksats.strategies.experimental.overlays.example_mvrv:"
            "ExampleMVRVStrategy"
        ),
        class_name="ExampleMVRVStrategy",
        module_path="stacksats.strategies.experimental.overlays.example_mvrv",
        tier="experimental",
        public_export=False,
        audit_enabled=True,
        family="overlay",
        description="Experimental MVRV model with multi-horizon BRK overlays.",
        docs_slug="example-mvrv",
        tags=("experimental", "overlay", "brk"),
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
    StrategyCatalogEntry(
        strategy_id="mvrv-plus",
        strategy_spec=(
            "stacksats.strategies.experimental.overlays.mvrv_plus:"
            "MVRVPlusStrategy"
        ),
        class_name="MVRVPlusStrategy",
        module_path="stacksats.strategies.experimental.overlays.mvrv_plus",
        tier="experimental",
        public_export=False,
        audit_enabled=True,
        family="overlay",
        description="Experimental MVRV baseline with BRK-aware regime gating.",
        docs_slug="mvrv-plus",
        tags=("experimental", "overlay", "risk"),
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        },
    ),
)

_CATALOG_BY_ID = {entry.strategy_id: entry for entry in _CATALOG}


def list_strategies(
    *,
    tier: StrategyTier | None = None,
    public_only: bool = True,
) -> tuple[StrategyCatalogEntry, ...]:
    """Return built-in strategy catalog entries in canonical order."""
    return tuple(
        entry
        for entry in _CATALOG
        if (tier is None or entry.tier == tier)
        and (not public_only or entry.public_export)
    )


def get_strategy_catalog_entry(strategy_id: str) -> StrategyCatalogEntry:
    """Return the catalog entry for a built-in strategy ID."""
    try:
        return _CATALOG_BY_ID[strategy_id]
    except KeyError as exc:
        raise KeyError(f"Unknown built-in strategy_id: {strategy_id}") from exc


def resolve_catalog_strategy_spec(strategy_id: str) -> str:
    """Return the canonical `module:Class` spec for a built-in strategy ID."""
    return get_strategy_catalog_entry(strategy_id).strategy_spec


def load_catalog_strategy_class(strategy_id: str) -> type[BaseStrategy]:
    """Resolve a built-in strategy ID to its class object."""
    entry = get_strategy_catalog_entry(strategy_id)
    module = importlib.import_module(entry.module_path)
    strategy_cls = getattr(module, entry.class_name)
    if not isinstance(strategy_cls, type) or not issubclass(strategy_cls, BaseStrategy):
        raise TypeError(
            f"Catalog entry '{strategy_id}' does not resolve to a BaseStrategy subclass."
        )
    return strategy_cls


def iter_catalog_strategy_classes(
    *,
    tier: StrategyTier | None = None,
    public_only: bool = True,
) -> tuple[type[BaseStrategy], ...]:
    """Return catalog strategy classes in canonical order."""
    return tuple(
        load_catalog_strategy_class(entry.strategy_id)
        for entry in list_strategies(tier=tier, public_only=public_only)
    )


__all__ = [
    "StrategyCatalogEntry",
    "get_strategy_catalog_entry",
    "iter_catalog_strategy_classes",
    "list_strategies",
    "load_catalog_strategy_class",
    "resolve_catalog_strategy_spec",
]
