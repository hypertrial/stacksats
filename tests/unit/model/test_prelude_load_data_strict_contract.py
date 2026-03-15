"""Tests for strict prelude.load_data delegation contract."""

from __future__ import annotations

import pandas as pd
import pytest

from stacksats.data_btc import DataLoadError
from stacksats.prelude import load_data


def test_load_data_delegates_to_btc_provider_with_defaults(mocker) -> None:
    expected_df = pd.DataFrame(
        {
            "PriceUSD": [40000.0, 41000.0],
            "price_usd": [40000.0, 41000.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    load_mock = mocker.Mock(return_value=expected_df)
    provider_instance = mocker.Mock(load=load_mock)
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    df = load_data()

    provider_cls.assert_called_once_with(parquet_path=None, max_staleness_days=3)
    load_mock.assert_called_once_with(backtest_start="2018-01-01", end_date=None)
    assert df is expected_df


def test_load_data_propagates_missing_price_failure(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(side_effect=DataLoadError("missing price_usd values"))
    )
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    with pytest.raises(DataLoadError, match="missing price_usd values"):
        load_data(parquet_path=None)


def test_load_data_propagates_missing_dates_failure(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(side_effect=DataLoadError("missing dates"))
    )
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    with pytest.raises(DataLoadError, match="missing dates"):
        load_data(parquet_path=None)


def test_load_data_propagates_past_only_cache_failure(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(
            side_effect=DataLoadError(
                "BRK data does not cover requested end_date."
            )
        )
    )
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    with pytest.raises(DataLoadError, match="does not cover requested end_date"):
        load_data(parquet_path="~/analytics.parquet", end_date="2020-12-31")

    provider_cls.assert_called_once_with(
        parquet_path="~/analytics.parquet",
        max_staleness_days=3,
    )
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date="2020-12-31",
    )
