"""Tests for the installable strategy-first StackSats package API."""

from __future__ import annotations

import numpy as np

import stacksats.export_weights as export_weights
import stacksats.export_weights as pkg_export_weights
from stacksats import BacktestConfig, MVRVStrategy
from stacksats.strategies.examples import UniformStrategy
from tests.test_helpers import btc_frame


def _sample_btc_df():
    return btc_frame(start="2022-01-01", days=520, price_start=20000.0, price_step=50.0).with_columns(
        mvrv=np.linspace(0.8, 2.2, 520)
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

    assert result.spd_table.height > 0
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

    assert result.spd_table.height > 0
    assert np.isfinite(result.exp_decay_percentile)
