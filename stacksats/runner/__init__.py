"""Strategy execution orchestration services."""

import datetime as dt

from .adapter import _FrameworkBacktestAdapter
from .core import StrategyRunner, backtest_dynamic_dca
from .daily import _DailyDecisionComputation, _DailyDecisionPreparationError
from ..strategy_time_series import WeightTimeSeriesBatch
from .validation import (
    STRICT_MUTATION_MESSAGE,
    STRICT_PROFILE_MUTATION_MESSAGE,
    WIN_RATE_TOLERANCE,
    WeightValidationError,
)

__all__ = [
    "STRICT_MUTATION_MESSAGE",
    "STRICT_PROFILE_MUTATION_MESSAGE",
    "WIN_RATE_TOLERANCE",
    "StrategyRunner",
    "WeightTimeSeriesBatch",
    "WeightValidationError",
    "_DailyDecisionComputation",
    "_DailyDecisionPreparationError",
    "_FrameworkBacktestAdapter",
    "backtest_dynamic_dca",
    "dt",
]
