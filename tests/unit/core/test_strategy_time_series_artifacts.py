from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stacksats.runner import StrategyRunner
from stacksats.strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    StrategyTimeSeries,
    StrategyTimeSeriesBatch,
)
from stacksats.strategy_types import BaseStrategy, ExportConfig, StrategyContext, TargetProfile


class UniformExportStrategy(BaseStrategy):
    strategy_id = "uniform-export"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        return TargetProfile(
            values=pd.Series(np.ones(len(features_df.index)), index=features_df.index),
            mode="absolute",
        )


def _btc_df() -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=1500, freq="D")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000, 60000, len(idx)),
            "CapMVRVCur": np.linspace(0.9, 2.1, len(idx)),
        },
        index=idx,
    )


def _metadata(**overrides: object) -> StrategySeriesMetadata:
    payload = {
        "strategy_id": "test-strategy",
        "strategy_version": "1.2.3",
        "run_id": "run-1",
        "config_hash": "abc123",
        "schema_version": "1.0.0",
        "generated_at": pd.Timestamp("2024-01-03 12:34:56"),
        "window_start": pd.Timestamp("2024-01-01 15:00:00"),
        "window_end": pd.Timestamp("2024-01-03 23:59:59"),
    }
    payload.update(overrides)
    return StrategySeriesMetadata(**payload)


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, np.nan],
        }
    )


def test_metadata_normalizes_and_validates_fields() -> None:
    metadata = _metadata()
    assert metadata.generated_at.tzinfo is not None
    assert str(metadata.generated_at.tzinfo) == "UTC"
    assert metadata.window_start == pd.Timestamp("2024-01-01")
    assert metadata.window_end == pd.Timestamp("2024-01-03")

    with pytest.raises(ValueError, match="strategy_id must be a non-empty string"):
        _metadata(strategy_id="")
    with pytest.raises(ValueError, match="window_start must be <= window_end"):
        _metadata(window_start=pd.Timestamp("2024-01-04"), window_end=pd.Timestamp("2024-01-03"))


def test_strategy_time_series_extra_schema_supports_declared_columns() -> None:
    extra_schema = (
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="Custom strategy score.",
            constraints=("finite when present",),
            source="strategy",
        ),
    )
    data = _data()
    data["custom_signal"] = [1.0, 2.0, 3.0]
    series = StrategyTimeSeries(metadata=_metadata(), data=data, extra_schema=extra_schema)

    assert "custom_signal" in series.schema()
    assert "custom_signal" in series.schema_markdown()


def test_strategy_time_series_extra_schema_rejects_duplicate_and_core_collision() -> None:
    duplicate = (
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="One.",
            source="strategy",
        ),
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="Two.",
            source="strategy",
        ),
    )
    with pytest.raises(ValueError, match="duplicate column names"):
        StrategyTimeSeries(metadata=_metadata(), data=_data(), extra_schema=duplicate)

    collision = (
        ColumnSpec(
            name="price_usd",
            dtype="float64",
            required=False,
            description="Bad collision.",
            source="strategy",
        ),
    )
    with pytest.raises(ValueError, match="collide with core StrategyTimeSeries schema"):
        StrategyTimeSeries(metadata=_metadata(), data=_data(), extra_schema=collision)


def test_strategy_time_series_csv_roundtrip(tmp_path) -> None:
    extra_schema = (
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="Custom strategy score.",
            source="strategy",
        ),
    )
    data = _data()
    data["custom_signal"] = [1.0, 2.0, 3.0]
    series = StrategyTimeSeries(metadata=_metadata(), data=data, extra_schema=extra_schema)
    csv_path = tmp_path / "series.csv"

    series.to_csv(csv_path)
    loaded = StrategyTimeSeries.from_csv(csv_path, metadata=_metadata(), extra_schema=extra_schema)

    assert loaded.to_dataframe().equals(series.to_dataframe())
    assert loaded.columns == series.columns
    assert loaded.window_key() == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))


def test_strategy_time_series_batch_roundtrips_and_propagates_generated_at(tmp_path) -> None:
    extra_schema = (
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="Custom strategy score.",
            source="strategy",
        ),
    )
    flat = pd.DataFrame(
        {
            "start_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "end_date": ["2024-01-02", "2024-01-02", "2024-02-02", "2024-02-02"],
            "date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
            "weight": [0.45, 0.55, 0.4, 0.6],
            "price_usd": [40000.0, 41000.0, 50000.0, np.nan],
            "custom_signal": [1.0, 2.0, 3.0, 4.0],
        }
    )
    generated_at = pd.Timestamp("2024-02-15 09:30:00Z")
    batch = StrategyTimeSeriesBatch.from_flat_dataframe(
        flat,
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        generated_at=generated_at,
        extra_schema=extra_schema,
    )

    assert batch.generated_at == generated_at
    assert all(window.metadata.generated_at == generated_at for window in batch.windows)
    assert batch.window_keys() == (
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-02")),
    )
    assert batch.date_span() == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-02"))

    csv_path = tmp_path / "batch.csv"
    batch.to_csv(csv_path)
    loaded = StrategyTimeSeriesBatch.from_csv(
        csv_path,
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        generated_at=generated_at,
        extra_schema=extra_schema,
    )

    assert loaded.to_dataframe().equals(batch.to_dataframe())
    assert loaded.for_window("2024-01-01", "2024-01-02").row_count == 2


def test_strategy_time_series_batch_defaults_extra_schema_from_windows() -> None:
    extra_schema = (
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="Custom strategy score.",
            source="strategy",
        ),
    )
    data = _data()
    data["custom_signal"] = [1.0, 2.0, 3.0]
    window = StrategyTimeSeries(metadata=_metadata(), data=data, extra_schema=extra_schema)

    batch = StrategyTimeSeriesBatch(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        windows=(window,),
    )

    assert batch.extra_schema == extra_schema


def test_strategy_time_series_batch_from_artifact_dir_roundtrip(tmp_path) -> None:
    runner = StrategyRunner()
    batch = runner.export(
        UniformExportStrategy(),
        ExportConfig(
            range_start="2023-01-01",
            range_end="2024-12-31",
            output_dir=str(tmp_path),
        ),
        btc_df=_btc_df(),
        current_date=pd.Timestamp("2024-01-15"),
    )

    artifact_dir = next(tmp_path.glob("uniform-export/1.0.0/*"))
    payload = (artifact_dir / "artifacts.json").read_text(encoding="utf-8")
    assert '"weights_csv": "weights.csv"' in payload
    loaded = StrategyTimeSeriesBatch.from_artifact_dir(artifact_dir)

    assert loaded.strategy_id == "uniform-export"
    assert loaded.strategy_version == "1.0.0"
    assert loaded.run_id == batch.run_id
    assert loaded.row_count == batch.row_count
    assert_frame_equal(loaded.to_dataframe(), batch.to_dataframe(), check_dtype=False)
