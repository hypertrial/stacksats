"""StackSats package public API."""

from .api import (
    BacktestResult,
    DailyOrderReceipt,
    DailyOrderRequest,
    DailyRunResult,
    ValidationResult,
)
from .column_map_provider import ColumnMapDataProvider, ColumnMapError
from .loader import load_strategy
from .model_development import precompute_features
from .prelude import load_data
from .runner import StrategyRunner
from .strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    TimeSeries,
    TimeSeriesBatch,
)
from .strategies.model_example import ExampleMVRVStrategy
from .strategies.model_mvrv_plus import MVRVPlusStrategy
from .strategies.mvrv import MVRVStrategy
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    DayState,
    ExportConfig,
    RunDailyConfig,
    StrategyArtifactSet,
    StrategyContractWarning,
    StrategyContext,
    StrategyMetadata,
    StrategyRunResult,
    StrategySpec,
    TargetProfile,
    ValidationConfig,
)

# Deprecated aliases — remove in 0.9.0
StrategyTimeSeries = TimeSeries
StrategyTimeSeriesBatch = TimeSeriesBatch

__all__ = [
    # Primary objects
    "TimeSeries",
    "TimeSeriesBatch",
    "BaseStrategy",
    # Results
    "BacktestResult",
    "ValidationResult",
    "DailyRunResult",
    # Configs
    "BacktestConfig",
    "ExportConfig",
    "RunDailyConfig",
    "ValidationConfig",
    # Supporting types
    "ColumnMapDataProvider",
    "ColumnMapError",
    "ColumnSpec",
    "DailyOrderReceipt",
    "DailyOrderRequest",
    "DayState",
    "ExampleMVRVStrategy",
    "MVRVStrategy",
    "MVRVPlusStrategy",
    "StrategyRunner",
    "StrategySeriesMetadata",
    "StrategyArtifactSet",
    "StrategyContractWarning",
    "StrategyContext",
    "StrategyMetadata",
    "StrategyRunResult",
    "StrategySpec",
    "TargetProfile",
    "load_strategy",
    "load_data",
    "precompute_features",
    # Deprecated aliases — remove in 0.9.0
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
]
