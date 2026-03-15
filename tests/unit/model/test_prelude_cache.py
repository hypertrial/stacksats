"""Tests for prelude.load_data provider configuration passthrough."""

from __future__ import annotations

import datetime as dt
import polars as pl

from stacksats.prelude import load_data


def test_load_data_passes_parquet_config_to_provider(mocker, tmp_path) -> None:
    expected_df = pl.DataFrame(
        {"date": [dt.datetime(2024, 1, 1)], "price_usd": [40000.0]}
    )
    provider_instance = mocker.Mock(load=mocker.Mock(return_value=expected_df))
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    load_data(parquet_path=str(tmp_path / "analytics.parquet"), max_staleness_days=6)

    provider_cls.assert_called_once_with(
        parquet_path=str(tmp_path / "analytics.parquet"),
        max_staleness_days=6,
    )
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date=None,
    )


def test_load_data_passes_end_date_to_provider(mocker) -> None:
    provider_instance = mocker.Mock(
        load=mocker.Mock(
            return_value=pl.DataFrame(
                {"date": [dt.datetime(2024, 1, 2)], "price_usd": [50000.0]}
            )
        )
    )
    provider_cls = mocker.Mock(return_value=provider_instance)
    mocker.patch("stacksats.prelude.BTCDataProvider", provider_cls)

    load_data(parquet_path=None, end_date="2024-01-02")

    provider_cls.assert_called_once_with(parquet_path=None, max_staleness_days=3)
    provider_instance.load.assert_called_once_with(
        backtest_start="2018-01-01",
        end_date="2024-01-02",
    )
