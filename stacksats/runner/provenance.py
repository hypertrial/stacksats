"""Runner provenance helpers."""

from .core import StrategyRunner


def config_hash(config) -> str:
    """Return the stable config hash used by runner artifacts."""
    return StrategyRunner._config_hash(config)


__all__ = ["config_hash"]
