from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    WeightTimeSeries,
    WeightTimeSeriesBatch,
)
from stacksats.strategy_types import BaseStrategy, ExportConfig, StrategyContext, TargetProfile
from tests.test_helpers import btc_frame


class UniformExportStrategy(BaseStrategy):
    strategy_id = "uniform-export"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx, signals
        return TargetProfile(
            values=pl.DataFrame(
                {
                    "date": features_df["date"],
                    "value": [1.0] * features_df.height,
                }
            ),
            mode="absolute",
        )


def _metadata(**overrides: object) -> StrategySeriesMetadata:
    payload = {
        "strategy_id": "test-strategy",
        "strategy_version": "1.2.3",
        "run_id": "run-1",
        "config_hash": "abc123",
        "schema_version": "1.0.0",
        "generated_at": dt.datetime(2024, 1, 3, 12, 34, 56, tzinfo=dt.timezone.utc),
        "window_start": dt.datetime(2024, 1, 1, 15, 0, 0),
        "window_end": dt.datetime(2024, 1, 3, 23, 59, 59),
    }
    payload.update(overrides)
    return StrategySeriesMetadata(**payload)


def _data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
            ],
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, None],
        }
    )


def test_metadata_normalizes_and_validates_fields() -> None:
    metadata = _metadata()

    assert metadata.generated_at.tzinfo is not None
    assert metadata.generated_at.tzinfo == dt.timezone.utc
    assert metadata.window_start == dt.datetime(2024, 1, 1)
    assert metadata.window_end == dt.datetime(2024, 1, 3)

    with pytest.raises(ValueError, match="strategy_id must be a non-empty string"):
        _metadata(strategy_id="")
    with pytest.raises(ValueError, match="window_start must be <= window_end"):
        _metadata(window_start=dt.datetime(2024, 1, 4), window_end=dt.datetime(2024, 1, 3))


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
    data = _data().with_columns(pl.Series("custom_signal", [1.0, 2.0, 3.0]))
    series = WeightTimeSeries(metadata=_metadata(), data=data, extra_schema=extra_schema)

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
        WeightTimeSeries(metadata=_metadata(), data=_data(), extra_schema=duplicate)

    collision = (
        ColumnSpec(
            name="price_usd",
            dtype="float64",
            required=False,
            description="Bad collision.",
            source="strategy",
        ),
    )
    with pytest.raises(ValueError, match="collide with core WeightTimeSeries schema"):
        WeightTimeSeries(metadata=_metadata(), data=_data(), extra_schema=collision)


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
    data = _data().with_columns(pl.Series("custom_signal", [1.0, 2.0, 3.0]))
    series = WeightTimeSeries(metadata=_metadata(), data=data, extra_schema=extra_schema)
    csv_path = tmp_path / "series.csv"

    series.to_csv(csv_path)
    loaded = WeightTimeSeries.from_csv(csv_path, metadata=_metadata(), extra_schema=extra_schema)

    assert loaded.to_dataframe().equals(series.to_dataframe())
    assert loaded.columns == series.columns
    assert loaded.window_key() == (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3))


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
    flat = pl.DataFrame(
        {
            "start_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "end_date": ["2024-01-02", "2024-01-02", "2024-02-02", "2024-02-02"],
            "date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
            "weight": [0.45, 0.55, 0.4, 0.6],
            "price_usd": [40000.0, 41000.0, 50000.0, None],
            "custom_signal": [1.0, 2.0, 3.0, 4.0],
        }
    )
    generated_at = dt.datetime(2024, 2, 15, 9, 30, 0, tzinfo=dt.timezone.utc)
    batch = WeightTimeSeriesBatch.from_flat_dataframe(
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
        (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)),
        (dt.datetime(2024, 2, 1), dt.datetime(2024, 2, 2)),
    )
    assert batch.date_span() == (dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 2))

    csv_path = tmp_path / "batch.csv"
    batch.to_csv(csv_path)
    loaded = WeightTimeSeriesBatch.from_csv(
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
    data = _data().with_columns(pl.Series("custom_signal", [1.0, 2.0, 3.0]))
    window = WeightTimeSeries(metadata=_metadata(), data=data, extra_schema=extra_schema)

    batch = WeightTimeSeriesBatch(
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
        btc_df=btc_frame(start="2021-01-01", days=1500),
        current_date=dt.datetime(2024, 1, 15, tzinfo=dt.timezone.utc),
    )

    artifact_dir = next(tmp_path.glob("uniform-export/1.0.0/*"))
    payload = (artifact_dir / "artifacts.json").read_text(encoding="utf-8")
    assert '"weights_csv": "weights.csv"' in payload

    loaded = WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    assert loaded.strategy_id == "uniform-export"
    assert loaded.strategy_version == "1.0.0"
    assert loaded.run_id == batch.run_id
    assert loaded.row_count == batch.row_count


def test_metadata_datetime_fallback_and_batch_edge_paths(tmp_path) -> None:
    fallback = StrategySeriesMetadata(
        strategy_id="s",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg",
        generated_at="2024-01-03 trailing",
    )
    assert fallback.generated_at == dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc)

    with pytest.raises(ValueError, match="must not be empty"):
        WeightTimeSeriesBatch(
            strategy_id="s",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg",
            windows=(),
        )

    dt_flat = pl.DataFrame(
        {
            "start_date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "end_date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "weight": [1.0, 1.0],
            "price_usd": [100.0, 101.0],
        }
    )
    batch = WeightTimeSeriesBatch.from_flat_dataframe(
        dt_flat,
        strategy_id="s",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg",
    )
    assert batch.window_count == 2

    bad_flat = dt_flat.with_columns(pl.lit(None).alias("date"))
    with pytest.raises(ValueError, match="valid datetimes"):
        WeightTimeSeriesBatch.from_flat_dataframe(
            bad_flat,
            strategy_id="s",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg",
        )


def test_batch_artifact_dir_resolution_and_validation_edges(tmp_path) -> None:
    window = WeightTimeSeries(metadata=_metadata(), data=_data())
    batch = WeightTimeSeriesBatch(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        windows=(window,),
    )

    mismatch_run = WeightTimeSeriesBatch(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        windows=(window,),
    )
    object.__setattr__(mismatch_run, "run_id", "batch-run")
    with pytest.raises(ValueError, match="run_id does not match"):
        mismatch_run.validate()

    mismatch_schema = WeightTimeSeriesBatch(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        windows=(window,),
        schema_version="1.0.0",
    )
    object.__setattr__(mismatch_schema, "schema_version", "9.9.9")
    with pytest.raises(ValueError, match="schema_version does not match"):
        mismatch_schema.validate()

    mismatch_config = WeightTimeSeriesBatch(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        windows=(window,),
    )
    object.__setattr__(mismatch_config, "config_hash", "other")
    with pytest.raises(ValueError, match="config_hash does not match"):
        mismatch_config.validate()

    extra_schema = (
        ColumnSpec(
            name="custom_signal",
            dtype="float64",
            required=False,
            description="Custom strategy score.",
            source="strategy",
        ),
    )
    window_with_schema = WeightTimeSeries(
        metadata=_metadata(),
        data=_data().with_columns(pl.Series("custom_signal", [1.0, 2.0, 3.0])),
        extra_schema=extra_schema,
    )
    batch_mismatch_extra = WeightTimeSeriesBatch(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        windows=(window_with_schema,),
    )
    object.__setattr__(batch_mismatch_extra, "extra_schema", ())
    with pytest.raises(ValueError, match="extra_schema"):
        batch_mismatch_extra.validate()

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    csv_path = artifact_dir / "weights.csv"
    batch.to_csv(csv_path)

    (artifact_dir / "artifacts.json").write_text(
        '{"strategy_id":"test-strategy","version":"1.2.3","run_id":"run-1","config_hash":"abc123","files":{"weights_csv":"missing/subdir.csv"}}',
        encoding="utf-8",
    )
    renamed = artifact_dir / "subdir.csv"
    csv_path.replace(renamed)
    loaded = WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)
    assert loaded.row_count == batch.row_count

    direct_csv = artifact_dir / "direct.csv"
    batch.to_csv(direct_csv)
    (artifact_dir / "artifacts.json").write_text(
        f'{{"strategy_id":"test-strategy","version":"1.2.3","run_id":"run-1","config_hash":"abc123","files":{{"weights_csv":"{direct_csv}"}}}}',
        encoding="utf-8",
    )
    loaded_direct = WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)
    assert loaded_direct.row_count == batch.row_count
