"""Strategy interfaces, catalog helpers, and stable built-ins."""

from .base import BaseStrategy, DayState, StrategyContext
from .catalog import (
    StrategyCatalogEntry,
    get_strategy_catalog_entry,
    iter_catalog_strategy_classes,
    list_strategies,
)

_STABLE_PUBLIC_CLASSES = iter_catalog_strategy_classes(tier="stable", public_only=True)
for _strategy_cls in _STABLE_PUBLIC_CLASSES:
    globals()[_strategy_cls.__name__] = _strategy_cls

__all__ = [
    "BaseStrategy",
    "DayState",
    "StrategyCatalogEntry",
    "StrategyContext",
    "get_strategy_catalog_entry",
    "list_strategies",
]
__all__.extend(_strategy_cls.__name__ for _strategy_cls in _STABLE_PUBLIC_CLASSES)
