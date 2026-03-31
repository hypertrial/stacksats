"""BTC data loading, packaged assets, and prelude backtest helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = ["btc_price_fetcher", "data_btc", "data_setup"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
