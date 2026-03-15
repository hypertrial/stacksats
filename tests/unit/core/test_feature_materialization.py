from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from stacksats.feature_materialization import (
    build_observed_frame,
    hash_dataframe,
    materialize_versioned_observations,
)
from stacksats.prelude import date_range_list


def test_build_observed_frame_restricts_to_observed_prefix() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 5),
        interval="1d", eager=True
    )
    frame = pl.DataFrame({"date": dates, "x": range(5)})

    observed = build_observed_frame(
        frame,
        start_date="2024-01-02",
        current_date="2024-01-04",
    )

    expected_dates = date_range_list("2024-01-02", "2024-01-04")
    assert observed["date"].to_list() == expected_dates


def test_build_observed_frame_handles_empty_frame() -> None:
    observed = build_observed_frame(
        pl.DataFrame(),
        start_date="2024-01-01",
        current_date="2024-01-03",
    )

    expected_dates = date_range_list("2024-01-01", "2024-01-03")
    assert observed["date"].to_list() == expected_dates


def test_build_observed_frame_accepts_date_dtype() -> None:
    frame = pl.DataFrame({
        "date": [dt.date(2024, 1, day) for day in range(1, 6)],
        "x": range(5),
    })

    observed = build_observed_frame(
        frame,
        start_date="2024-01-02",
        current_date="2024-01-04",
    )

    expected_dates = date_range_list("2024-01-02", "2024-01-04")
    assert observed["date"].to_list() == expected_dates


def test_materialize_versioned_observations_selects_latest_available_revision() -> None:
    observations = pl.DataFrame({
        "effective_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "available_at": ["2024-01-01", "2024-01-03", "2024-01-02"],
        "revision_id": [1, 2, 1],
        "signal": [1.0, 2.0, 3.0],
    })

    as_of_old = materialize_versioned_observations(observations, as_of_date="2024-01-02")
    as_of_new = materialize_versioned_observations(observations, as_of_date="2024-01-03")

    jan1 = dt.datetime(2024, 1, 1)
    jan1 = dt.datetime(2024, 1, 1)
    row_old = as_of_old.filter(pl.col("effective_date").dt.date() == jan1.date())
    row_new = as_of_new.filter(pl.col("effective_date").dt.date() == jan1.date())
    assert float(row_old["signal"][0]) == 1.0
    assert float(row_new["signal"][0]) == 2.0


def test_materialize_versioned_observations_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        materialize_versioned_observations(
            pl.DataFrame({"effective_date": ["2024-01-01"]}),
            as_of_date="2024-01-01",
        )


def test_hash_dataframe_is_stable_for_equal_content() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2),
        interval="1d", eager=True
    )
    frame_a = pl.DataFrame({"date": dates, "x": [1.0, None]})
    frame_b = pl.DataFrame({"date": dates, "x": [1.0, None]})

    assert hash_dataframe(frame_a) == hash_dataframe(frame_b)


def test_hash_dataframe_accepts_date_dtype() -> None:
    frame = pl.DataFrame({
        "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
        "x": [1.0, None],
    })

    assert hash_dataframe(frame)
