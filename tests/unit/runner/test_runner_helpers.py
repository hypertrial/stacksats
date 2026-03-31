"""Tests for stacksats.runner.helpers.slice_window_or_filter dict path and edge branches."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from stacksats.runner.helpers import (
    build_fold_ranges,
    perturb_future_features,
    perturb_future_source_data,
    slice_window_or_filter,
)


def test_slice_window_or_filter_dict_path_exact_match() -> None:
    """Dict date_index with start_idx, expected_days, exact slice returns window."""
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    date_index = {d: i for i, d in enumerate(dates.to_list())}
    start = dates[2]
    end = dates[6]
    result = slice_window_or_filter(
        frame,
        date_index,
        start,
        end,
        expected_days=5,
    )
    assert result.height == 5
    assert result["date"][0] == start
    assert result["date"][-1] == end


def test_slice_window_or_filter_dict_path_mismatch_falls_to_filter() -> None:
    """Dict path when slice height or end mismatch falls through to filter."""
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    date_index = {d: i for i, d in enumerate(dates.to_list())}
    start = dates[2]
    end = dates[4]
    result = slice_window_or_filter(
        frame,
        date_index,
        start,
        end,
        expected_days=5,
    )
    assert result.height == 3
    assert result["date"][0] == start
    assert result["date"][-1] == end


def test_slice_window_or_filter_dict_path_no_expected_days_uses_end_idx() -> None:
    """Dict path with expected_days=None uses end_idx slice."""
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    date_index = {d: i for i, d in enumerate(dates.to_list())}
    start = dates[1]
    end = dates[5]
    result = slice_window_or_filter(
        frame,
        date_index,
        start,
        end,
        expected_days=None,
    )
    assert result.height == 5
    assert result["date"][0] == start
    assert result["date"][-1] == end


def test_slice_window_or_filter_dict_path_start_missing_falls_to_filter() -> None:
    """Dict path when start not in index falls through to filter."""
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    date_index = {dates[i]: i for i in [1, 3, 5]}
    start = dates[2]
    end = dates[6]
    result = slice_window_or_filter(
        frame,
        date_index,
        start,
        end,
        expected_days=5,
    )
    assert result.height == 5
    assert result["date"][0] == start
    assert result["date"][-1] == end


def test_slice_window_or_filter_dict_path_end_idx_lt_start_falls_to_filter() -> None:
    """Dict path when end_idx < start_idx falls through to filter."""
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    date_index = {d: i for i, d in enumerate(dates.to_list())}
    date_index[dates[5]] = 1
    start = dates[2]
    end = dates[5]
    result = slice_window_or_filter(
        frame,
        date_index,
        start,
        end,
        expected_days=None,
    )
    assert result.height == 4
    assert result["date"][0] == start
    assert result["date"][-1] == end


def test_slice_window_or_filter_dict_path_end_missing_falls_to_filter() -> None:
    """Dict path when end not in index (expected_days=None) falls through to filter."""
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    date_index = {dates[i]: i for i in range(5)}
    start = dates[1]
    end = dates[7]
    result = slice_window_or_filter(
        frame,
        date_index,
        start,
        end,
        expected_days=None,
    )
    assert result.height == 7
    assert result["date"][0] == start
    assert result["date"][-1] == end


def test_runner_helper_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pl.datetime_range(
        dt.datetime(2020, 1, 1),
        dt.datetime(2025, 12, 31),
        interval="1d",
        eager=True,
    ).to_list()
    probe = dates[1]

    labels = pl.DataFrame({"date": dates[:4], "label": ["a", "b", "c", "d"]})
    perturbed = perturb_future_features(labels, probe)
    assert perturbed["label"].to_list() == ["a", "b", "d", "c"]

    single_future = perturb_future_source_data(
        pl.DataFrame({"date": dates[:3], "label": ["a", "b", "c"]}),
        probe,
    )
    assert single_future["label"].to_list() == ["a", "b", "c"]

    numeric_only = perturb_future_features(
        pl.DataFrame({"date": dates[:4], "value": [1.0, 2.0, 3.0, 4.0]}),
        probe,
    )
    assert numeric_only["value"].to_list()[:2] == [1.0, 2.0]

    assert build_fold_ranges(dates[0], dates[-1])

    monkeypatch.setattr(
        "stacksats.runner.helpers.np.linspace",
        lambda *args, **kwargs: [0, 1, 366, 732, 1096],
    )
    assert build_fold_ranges(dates[0], dates[-1])

    monkeypatch.setattr(
        "stacksats.runner.helpers.np.linspace",
        lambda *args, **kwargs: [0, 100, 500, 800, 1096],
    )
    sparse = build_fold_ranges(dates[0], dates[-1])
    assert sparse

    monkeypatch.setattr(
        "stacksats.runner.helpers.np.linspace",
        lambda *args, **kwargs: [0, 1, 2, 3, 1096],
    )
    assert build_fold_ranges(dates[0], dates[-1])

    shorter_end = dt.datetime(2022, 12, 31)
    monkeypatch.setattr(
        "stacksats.runner.helpers.np.linspace",
        lambda *args, **kwargs: [0, 1, 101, 1096],
    )
    assert build_fold_ranges(dates[0], shorter_end)
