from __future__ import annotations

import importlib.util
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from stacksats import BacktestConfig, StrategyRunner, ValidationConfig

STRATEGY_PATH = Path("my_strategy.py")
STRATEGY_CLASS_NAME = "MyStrategy"


def _load_strategy_class():
    module_path = STRATEGY_PATH.expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(
            f"Expected strategy file at {module_path}. Update STRATEGY_PATH if your strategy "
            "lives elsewhere."
        )

    spec = importlib.util.spec_from_file_location("my_strategy", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, STRATEGY_CLASS_NAME)


def _btc_df(days: int = 1461) -> pl.DataFrame:
    start = datetime(2022, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(days)]
    return pl.DataFrame(
        {
            "date": dates,
            "price_usd": [20000.0 + (offset * 40.0) for offset in range(days)],
            "mvrv": [0.8 + (offset / max(days, 1)) for offset in range(days)],
        }
    )


def test_my_strategy_smoke() -> None:
    strategy_cls = _load_strategy_class()
    strategy = strategy_cls()
    runner = StrategyRunner.from_dataframe(_btc_df())

    validation = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
            min_win_rate=0.0,
            strict=True,
        ),
    )
    backtest = runner.backtest(
        strategy,
        BacktestConfig(start_date="2024-01-01", end_date="2024-12-31"),
    )

    assert validation.summary()
    assert backtest.spd_table.height > 0
    assert math.isfinite(backtest.score)
