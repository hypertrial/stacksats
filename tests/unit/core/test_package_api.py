"""Tests for the installable strategy-first StackSats package API."""

import numpy as np
import pandas as pd

import stacksats.export_weights as export_weights
import stacksats.export_weights as pkg_export_weights
from stacksats import BacktestConfig, MVRVStrategy
from stacksats.strategies.examples import UniformStrategy


def _sample_btc_df() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=520, freq="D")
    price = np.linspace(20000.0, 45000.0, len(dates))
    mvrv = np.linspace(0.8, 2.2, len(dates))
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": price,
            "CapMVRVCur": mvrv,
        },
        index=dates,
    )


def test_export_module_identity():
    """Top-level export module should alias packaged module object."""
    assert export_weights is pkg_export_weights


def test_backtest_uniform_strategy():
    """Users can backtest a custom strategy through strategy methods."""
    btc_df = _sample_btc_df()
    result = UniformStrategy().backtest(
        BacktestConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            strategy_label="uniform-test",
        ),
        btc_df=btc_df,
    )

    assert len(result.spd_table) > 0
    assert np.isfinite(result.win_rate)
    assert np.isfinite(result.score)


def test_backtest_default_strategy():
    """Built-in MVRV strategy is compatible with strategy methods."""
    btc_df = _sample_btc_df()

    result = MVRVStrategy().backtest(
        BacktestConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            strategy_label="mvrv-test",
        ),
        btc_df=btc_df,
    )

    assert len(result.spd_table) > 0
    assert np.isfinite(result.exp_decay_percentile)
