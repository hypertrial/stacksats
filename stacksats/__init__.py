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
    StrategyTimeSeries,
    StrategyTimeSeriesBatch,
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

__all__ = [
    "BacktestResult",
    "BacktestConfig",
    "BaseStrategy",
    "ColumnMapDataProvider",
    "ColumnMapError",
    "DailyOrderReceipt",
    "DailyOrderRequest",
    "DailyRunResult",
    "DayState",
    "ExportConfig",
    "ExampleMVRVStrategy",
    "MVRVStrategy",
    "MVRVPlusStrategy",
    "RunDailyConfig",
    "ColumnSpec",
    "StrategyRunner",
    "StrategySeriesMetadata",
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
    "StrategyArtifactSet",
    "StrategyContractWarning",
    "StrategyContext",
    "StrategyMetadata",
    "StrategyRunResult",
    "StrategySpec",
    "TargetProfile",
    "ValidationResult",
    "ValidationConfig",
    "load_strategy",
    "load_data",
    "precompute_features",
]
