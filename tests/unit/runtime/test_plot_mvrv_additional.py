from __future__ import annotations

import datetime as dt
import runpy
import sys
import warnings

import numpy as np
import polars as pl
import pytest

from stacksats import plot_mvrv
from stacksats.plot_mvrv import plot_mvrv_metrics


def test_plot_mvrv_metrics_autocomputes_missing_zscore_and_uses_long_range_locators(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    dates = pl.datetime_range(dt.datetime(2022, 1, 1), dt.datetime(2023, 5, 15), interval="1d", eager=True)
    df = pl.DataFrame({"date": dates, "mvrv": np.linspace(0.8, 2.5, len(dates))})
    monkeypatch.setattr("stacksats.plot_mvrv.plt.savefig", lambda *_args, **_kwargs: None)

    calls = {"year": 0, "month_args": []}
    original_year_locator = plot_mvrv.mdates.YearLocator
    original_month_locator = plot_mvrv.mdates.MonthLocator

    def _year_locator_spy(*args, **kwargs):
        calls["year"] += 1
        return original_year_locator(*args, **kwargs)

    def _month_locator_spy(*args, **kwargs):
        calls["month_args"].append(args)
        return original_month_locator(*args, **kwargs)

    monkeypatch.setattr("stacksats.plot_mvrv.mdates.YearLocator", _year_locator_spy)
    monkeypatch.setattr("stacksats.plot_mvrv.mdates.MonthLocator", _month_locator_spy)

    plot_mvrv_metrics(df, output_path=str(tmp_path / "mvrv.svg"))

    assert calls["year"] >= 1
    assert any(args and args[0] == (1, 7) for args in calls["month_args"])


def test_plot_mvrv_metrics_uses_medium_range_date_locators(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    dates = pl.datetime_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 4, 29), interval="1d", eager=True)
    df = pl.DataFrame(
        {
            "date": dates,
            "mvrv": np.linspace(1.0, 2.0, len(dates)),
            "CapMVRVZ": np.linspace(-1.0, 1.0, len(dates)),
        },
    )
    monkeypatch.setattr("stacksats.plot_mvrv.plt.savefig", lambda *_args, **_kwargs: None)

    calls = {"month": 0, "weekday": 0}
    original_month_locator = plot_mvrv.mdates.MonthLocator
    original_weekday_locator = plot_mvrv.mdates.WeekdayLocator

    def _month_locator_spy(*args, **kwargs):
        calls["month"] += 1
        return original_month_locator(*args, **kwargs)

    def _weekday_locator_spy(*args, **kwargs):
        calls["weekday"] += 1
        return original_weekday_locator(*args, **kwargs)

    monkeypatch.setattr("stacksats.plot_mvrv.mdates.MonthLocator", _month_locator_spy)
    monkeypatch.setattr("stacksats.plot_mvrv.mdates.WeekdayLocator", _weekday_locator_spy)

    plot_mvrv_metrics(df, output_path=str(tmp_path / "mvrv_medium.svg"))

    assert calls["month"] >= 1
    assert calls["weekday"] >= 1


def test_plot_mvrv_metrics_raises_when_cleaned_dataset_is_empty(tmp_path) -> None:
    dates = pl.datetime_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 10), interval="1d", eager=True)
    df = pl.DataFrame(
        {
            "date": dates,
            "mvrv": np.full(len(dates), np.nan),
            "CapMVRVZ": np.full(len(dates), np.nan),
        },
    )

    with pytest.raises(ValueError, match="No valid MVRV data available after removing missing values"):
        plot_mvrv_metrics(df, output_path=str(tmp_path / "unused.svg"))


def test_plot_mvrv_module_dunder_main_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    dates = pl.datetime_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 4, 29), interval="1d", eager=True)
    df = pl.DataFrame(
        {
            "date": dates,
            "mvrv": np.linspace(1.0, 2.0, len(dates)),
            "CapMVRVZ": np.linspace(-1.0, 1.0, len(dates)),
        },
    )
    monkeypatch.setattr(
        "stacksats.plot_mvrv.BTCDataProvider.load",
        lambda *_args, **_kwargs: df,
    )
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_mvrv.py", "--output", str(tmp_path / "mvrv.svg")],
    )

    with pytest.raises(SystemExit) as excinfo:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'.*' found in sys.modules after import of package '.*'",
                category=RuntimeWarning,
            )
            runpy.run_module("stacksats.plot_mvrv", run_name="__main__")

    assert excinfo.value.code == 0
