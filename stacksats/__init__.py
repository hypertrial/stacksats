"""StackSats package public API."""

from .api import (
    BacktestResult,
    DailyOrderReceipt,
    DailyOrderRequest,
    DailyRunResult,
    ValidationResult,
)
from .column_map_provider import ColumnMapDataProvider, ColumnMapError
from .eda import MergedMetricsDataset, MetricCatalog, load_metric_catalog, open_merged_metrics
from .feature_time_series import FeatureTimeSeries
from .loader import load_strategy
from .model_development import precompute_features
from .prelude import load_data
from .runner import StrategyRunner
from .strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    WeightTimeSeries,
    WeightTimeSeriesBatch,
)
from .strategies.examples import MomentumStrategy, SimpleZScoreStrategy, UniformStrategy
from .strategies.mvrv import MVRVStrategy
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    DayState,
    ExportConfig,
    RunDailyConfig,
    StrategyArtifactSet,
    StrategyContext,
    StrategyLazyContext,
    StrategyMetadata,
    StrategyRunResult,
    StrategySpec,
    TargetProfile,
    ValidationConfig,
    strategy_context_from_features_df,
)

__all__ = [
    # Primary objects
    "FeatureTimeSeries",
    "MergedMetricsDataset",
    "WeightTimeSeries",
    "WeightTimeSeriesBatch",
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
    "MetricCatalog",
    "DailyOrderReceipt",
    "DailyOrderRequest",
    "DayState",
    "MomentumStrategy",
    "MVRVStrategy",
    "SimpleZScoreStrategy",
    "StrategyRunner",
    "StrategySeriesMetadata",
    "StrategyArtifactSet",
    "StrategyContext",
    "StrategyLazyContext",
    "StrategyMetadata",
    "StrategyRunResult",
    "StrategySpec",
    "TargetProfile",
    "UniformStrategy",
    "load_strategy",
    "load_metric_catalog",
    "load_data",
    "open_merged_metrics",
    "precompute_features",
    "strategy_context_from_features_df",
]
