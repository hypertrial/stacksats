from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from stacksats.strategy_time_series import StrategySeriesMetadata, WeightTimeSeries
from stacksats.strategy_time_series_schema import ColumnSpec


def _metadata() -> StrategySeriesMetadata:
    return StrategySeriesMetadata(
        strategy_id="s",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg",
        schema_version="1.0.0",
        window_start=dt.datetime(2024, 1, 1),
        window_end=dt.datetime(2024, 1, 3),
    )


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
            ],
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [10.0, 11.0, 12.0],
            "day_index": [0, 1, 2],
        }
    )


def test_strategy_time_series_from_dataframe_and_date_index() -> None:
    series = WeightTimeSeries.from_dataframe(_frame(), metadata=_metadata())
    assert series.date_index() == [
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 2),
        dt.datetime(2024, 1, 3),
    ]


def test_strategy_time_series_normalize_core_columns_without_date_passthrough() -> None:
    data = pl.DataFrame({"weight": [1.0], "price_usd": [10.0]})
    assert WeightTimeSeries._normalize_core_columns(data).equals(data)


def test_strategy_time_series_optional_numeric_and_datetime_validation_edges() -> None:
    numeric_schema = (
        ColumnSpec(
            name="signal",
            dtype="float64",
            required=False,
            description="signal",
            source="strategy",
        ),
        ColumnSpec(
            name="decision_time",
            dtype="datetime64[ns]",
            required=False,
            description="decision",
            source="strategy",
        ),
    )

    bad_numeric = _frame().with_columns(
        pl.Series("signal", ["bad", "1.0", "2.0"]),
        pl.Series("decision_time", [dt.datetime(2024, 1, 1)] * 3),
    )
    with pytest.raises(ValueError, match="signal"):
        WeightTimeSeries(metadata=_metadata(), data=bad_numeric, extra_schema=numeric_schema)

    bad_numeric_inf = _frame().with_columns(
        pl.Series("signal", [1.0, float("inf"), 2.0]),
        pl.Series("decision_time", [dt.datetime(2024, 1, 1)] * 3),
    )
    with pytest.raises(ValueError, match="signal"):
        WeightTimeSeries(metadata=_metadata(), data=bad_numeric_inf, extra_schema=numeric_schema)

    bad_datetime_utf8 = _frame().with_columns(
        pl.Series("signal", [1.0, 2.0, 3.0]),
        pl.Series("decision_time", ["2024-01-01", "bad", "2024-01-03"]),
    )
    with pytest.raises(ValueError, match="decision_time"):
        WeightTimeSeries(metadata=_metadata(), data=bad_datetime_utf8, extra_schema=numeric_schema)

    bad_datetime_other = _frame().with_columns(
        pl.Series("signal", [1.0, 2.0, 3.0]),
        pl.Series("decision_time", [object(), object(), object()], dtype=pl.Object),
    )
    with pytest.raises(Exception):
        WeightTimeSeries(metadata=_metadata(), data=bad_datetime_other, extra_schema=numeric_schema)


def test_strategy_time_series_native_timestamp_and_empty_summary_edges() -> None:
    class _NullLike:
        def is_null(self) -> bool:
            return True

    class _BadValue:
        def __str__(self) -> str:
            return "bad"

    assert WeightTimeSeries._native_timestamp(None) is None
    assert WeightTimeSeries._native_timestamp("NaT") is None
    assert WeightTimeSeries._native_timestamp(_NullLike()) is None
    assert WeightTimeSeries._native_timestamp(_BadValue()) is None

    summary = WeightTimeSeries._series_numeric_summary(pl.Series("x", [None, None], dtype=pl.Float64))
    assert summary == {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "p25": None,
        "median": None,
        "p75": None,
        "max": None,
    }


def test_strategy_time_series_date_contract_direct_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    series = WeightTimeSeries.from_dataframe(_frame(), metadata=_metadata())

    with pytest.raises(ValueError, match="valid datetimes"):
        series._validate_date_contract(
            pl.Series("date", [dt.datetime(2024, 1, 1), None, dt.datetime(2024, 1, 3)], dtype=pl.Datetime)
        )

    with pytest.raises(ValueError, match="sorted ascending"):
        series._validate_date_contract(
            pl.Series(
                "date",
                [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)],
                dtype=pl.Datetime,
            )
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            "stacksats.prelude.date_range_list",
            lambda start, end: [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)],
        )
        with pytest.raises(ValueError, match="unexpected dates detected"):
            series._validate_date_contract(
                pl.Series(
                    "date",
                    [
                        dt.datetime(2024, 1, 1),
                        dt.datetime(2024, 1, 2),
                        dt.datetime(2024, 1, 3),
                    ],
                    dtype=pl.Datetime,
                )
            )

    monkeypatch.setattr(
        "stacksats.strategy_time_series._to_naive_dt",
        lambda value: value if isinstance(value, dt.datetime) else value,
    )
    object.__setattr__(series.metadata, "window_end", dt.datetime(2024, 1, 4))

    with monkeypatch.context() as patcher:
        patcher.setattr(
            "stacksats.prelude.date_range_list",
            lambda start, end: [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 4),
            ],
        )
        with pytest.raises(ValueError, match="must exactly match the daily range"):
            series._validate_date_contract(
                pl.Series(
                    "date",
                    [
                        dt.datetime(2024, 1, 1),
                        dt.datetime(2024, 1, 3),
                        dt.datetime(2024, 1, 4),
                    ],
                    dtype=pl.Datetime,
                )
            )


def test_strategy_time_series_validate_handles_utf8_dates() -> None:
    utf8 = _frame().with_columns(
        pl.Series("date", ["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    series = WeightTimeSeries(metadata=_metadata(), data=utf8)
    assert series.to_dataframe()["date"].dtype == pl.Datetime
