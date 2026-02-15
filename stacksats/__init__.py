"""StackSats package public API."""

from .api import BacktestResult, ValidationResult
from .loader import load_strategy
from .model_development import precompute_features
from .prelude import load_data
from .strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    StrategyTimeSeries,
    StrategyTimeSeriesBatch,
)
from .strategies.mvrv import MVRVStrategy
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    DayState,
    ExportConfig,
    StrategyArtifactSet,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)

__all__ = [
    "BacktestResult",
    "BacktestConfig",
    "BaseStrategy",
    "DayState",
    "ExportConfig",
    "MVRVStrategy",
    "ColumnSpec",
    "StrategySeriesMetadata",
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
    "StrategyArtifactSet",
    "StrategyContext",
    "TargetProfile",
    "ValidationResult",
    "ValidationConfig",
    "load_strategy",
    "load_data",
    "precompute_features",
]
