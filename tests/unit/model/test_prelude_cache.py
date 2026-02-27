"""Tests for prelude.load_data provider configuration passthrough."""

from __future__ import annotations

import pandas as pd

from stacksats.prelude import load_data


def test_load_data_passes_cache_config_to_provider(mocker, tmp_path) -> None:
    expected_df = pd.DataFrame(
        {"PriceUSD_coinmetrics": [40000.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    provider_instance = mocker.Mock(load=mocker.Mock(return_value=expected_df))
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    load_data(cache_dir=str(tmp_path), max_age_hours=6)

    provider_cls.assert_called_once_with(cache_dir=str(tmp_path), max_age_hours=6)
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date=None,
    )


def test_load_data_passes_end_date_to_provider(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(
            return_value=pd.DataFrame(
                {"PriceUSD_coinmetrics": [50000.0]},
                index=pd.to_datetime(["2024-01-02"]),
            )
        )
    )
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    load_data(cache_dir=None, end_date="2024-01-02")

    provider_cls.assert_called_once_with(cache_dir=None, max_age_hours=24)
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date="2024-01-02",
    )
