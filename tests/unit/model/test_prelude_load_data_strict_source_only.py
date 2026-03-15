"""Tests that prelude.load_data no longer applies MVRV fallback/synthetic-row logic."""

from __future__ import annotations

import datetime as dt
import polars as pl

from stacksats.prelude import load_data


def test_load_data_preserves_missing_today_mvrv_without_fallback(mocker) -> None:
    today = dt.datetime(2024, 1, 3)
    source_df = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), today],
            "PriceUSD": [40000.0, 41000.0, 42000.0],
            "price_usd": [40000.0, 41000.0, 42000.0],
            "mvrv": [2.0, 2.1, None],
        },
    )
    provider_instance = mocker.Mock(load=mocker.Mock(return_value=source_df))
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    result_df = load_data(end_date=today.strftime("%Y-%m-%d"))

    row = result_df.filter(pl.col("date") == today).row(0, named=True)
    assert row["mvrv"] is None
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date="2024-01-03",
    )


def test_load_data_does_not_synthesize_today_row(mocker) -> None:
    source_df = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "PriceUSD": [40000.0, 41000.0],
            "price_usd": [40000.0, 41000.0],
            "mvrv": [2.0, 2.1],
        },
    )
    provider_instance = mocker.Mock(load=mocker.Mock(return_value=source_df))
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    result_df = load_data(end_date="2024-01-02")

    assert result_df["date"].to_list() == source_df["date"].to_list()
    assert result_df.equals(source_df)
