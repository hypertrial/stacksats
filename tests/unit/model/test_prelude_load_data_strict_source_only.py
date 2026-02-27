"""Tests that prelude.load_data no longer applies MVRV fallback/synthetic-row logic."""

from __future__ import annotations

import pandas as pd

from stacksats.prelude import load_data


def test_load_data_preserves_missing_today_mvrv_without_fallback(mocker) -> None:
    today = pd.Timestamp("2024-01-03")
    source_df = pd.DataFrame(
        {
            "PriceUSD": [40000.0, 41000.0, 42000.0],
            "PriceUSD_coinmetrics": [40000.0, 41000.0, 42000.0],
            "CapMVRVCur": [2.0, 2.1, None],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    provider_instance = mocker.Mock(load=mocker.Mock(return_value=source_df))
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    result_df = load_data(end_date=today.strftime("%Y-%m-%d"))

    assert pd.isna(result_df.loc[today, "CapMVRVCur"])
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date="2024-01-03",
    )


def test_load_data_does_not_synthesize_today_row(mocker) -> None:
    source_df = pd.DataFrame(
        {
            "PriceUSD": [40000.0, 41000.0],
            "PriceUSD_coinmetrics": [40000.0, 41000.0],
            "CapMVRVCur": [2.0, 2.1],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    provider_instance = mocker.Mock(load=mocker.Mock(return_value=source_df))
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    result_df = load_data(end_date="2024-01-02")

    assert list(result_df.index) == list(source_df.index)
    assert result_df.equals(source_df)
