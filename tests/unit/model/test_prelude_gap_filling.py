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
            "PriceUSD_coinmetrics": [40000.0, 41000.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    load_mock = mocker.Mock(return_value=expected_df)
    provider_instance = mocker.Mock(load=load_mock)
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    df = load_data()

    provider_cls.assert_called_once_with(cache_dir="~/.stacksats/cache", max_age_hours=24)
    load_mock.assert_called_once_with(backtest_start="2018-01-01", end_date=None)
    assert df is expected_df


def test_load_data_propagates_missing_price_failure(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(side_effect=DataLoadError("missing PriceUSD values"))
    )
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    with pytest.raises(DataLoadError, match="missing PriceUSD values"):
        load_data(cache_dir=None)


def test_load_data_propagates_missing_dates_failure(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(side_effect=DataLoadError("missing dates"))
    )
    mocker.patch("stacksats.prelude.BTCDataProvider", return_value=provider_instance)

    with pytest.raises(DataLoadError, match="missing dates"):
        load_data(cache_dir=None)


def test_load_data_propagates_past_only_cache_failure(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(
            side_effect=DataLoadError(
                "Past-only backtest requested but no local CoinMetrics cache file was found."
            )
        )
    )
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    with pytest.raises(DataLoadError, match="Past-only backtest requested"):
        load_data(cache_dir="~/.stacksats/cache", end_date="2020-12-31")

    provider_cls.assert_called_once_with(cache_dir="~/.stacksats/cache", max_age_hours=24)
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date="2020-12-31",
    )
