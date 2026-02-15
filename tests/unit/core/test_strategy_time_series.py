from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.strategy_time_series import (
    StrategySeriesMetadata,
    StrategyTimeSeries,
    StrategyTimeSeriesBatch,
)


def _metadata() -> StrategySeriesMetadata:
    return StrategySeriesMetadata(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        window_start=pd.Timestamp("2024-01-01"),
        window_end=pd.Timestamp("2024-01-03"),
    )


def test_strategy_time_series_valid_payload() -> None:
    data = pd.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, np.nan],
            "locked": [True, True, False],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    out = series.to_dataframe()
    assert list(out.columns) == ["day_index", "date", "weight", "price_usd", "locked"]
    assert np.isclose(float(out["weight"].sum()), 1.0)


def test_strategy_time_series_rejects_unknown_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "weight": [0.4, 0.6],
            "price_usd": [42000.0, 43000.0],
            "mystery": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Schema coverage missing"):
        StrategyTimeSeries(metadata=_metadata(), data=data)


def test_strategy_time_series_rejects_weight_sum_mismatch() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "weight": [0.4, 0.4],
            "price_usd": [42000.0, 43000.0],
        }
    )

    with pytest.raises(ValueError, match="must sum to 1.0"):
        StrategyTimeSeries(
            metadata=StrategySeriesMetadata(
                strategy_id="test-strategy",
                strategy_version="1.2.3",
                run_id="run-1",
                config_hash="abc123",
                window_start=pd.Timestamp("2024-01-01"),
                window_end=pd.Timestamp("2024-01-02"),
            ),
            data=data,
        )


def test_strategy_time_series_batch_from_flat_dataframe() -> None:
    flat = pd.DataFrame(
        {
            "start_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "end_date": ["2024-01-02", "2024-01-02", "2024-02-02", "2024-02-02"],
            "date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
            "weight": [0.45, 0.55, 0.4, 0.6],
            "price_usd": [40000.0, 41000.0, 50000.0, np.nan],
        }
    )
    batch = StrategyTimeSeriesBatch.from_flat_dataframe(
        flat,
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
    )

    assert batch.window_count == 2
    assert batch.row_count == 4
    flattened = batch.to_dataframe()
    assert set(["start_date", "end_date", "date", "weight", "price_usd"]).issubset(
        flattened.columns
    )
    first = batch.for_window("2024-01-01", "2024-01-02")
    assert len(first.data) == 2


def test_strategy_time_series_batch_rejects_duplicate_windows() -> None:
    md = _metadata()
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [1.0, 2.0, 3.0],
        }
    )
    window = StrategyTimeSeries(metadata=md, data=data)
    with pytest.raises(ValueError, match="Duplicate window key"):
        StrategyTimeSeriesBatch(
            strategy_id=md.strategy_id,
            strategy_version=md.strategy_version,
            run_id=md.run_id,
            config_hash=md.config_hash,
            windows=(window, window),
        )
