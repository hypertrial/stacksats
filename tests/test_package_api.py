"""Tests for the installable StackSats package API."""

import numpy as np
import pandas as pd

import stacksats.export_weights as export_weights
import stacksats.export_weights as pkg_export_weights
from stacksats import CallableWindowStrategy, MVRVStrategy, run_backtest


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


def test_compat_alias_module_identity():
    """Top-level compatibility module should alias packaged module object."""
    assert export_weights is pkg_export_weights


def test_run_backtest_callable_strategy():
    """Users can backtest a custom callable strategy through package API."""
    btc_df = _sample_btc_df()

    def uniform_strategy(
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        idx = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.Series(np.full(len(idx), 1.0 / len(idx)), index=idx)

    result = run_backtest(
        CallableWindowStrategy(uniform_strategy),
        btc_df=btc_df,
        start_date="2022-01-01",
        end_date="2023-05-01",
        strategy_label="uniform-test",
    )

    assert len(result.spd_table) > 0
    assert np.isfinite(result.win_rate)
    assert np.isfinite(result.score)


def test_run_backtest_default_strategy():
    """Built-in MVRV strategy is compatible with package API."""
    btc_df = _sample_btc_df()

    result = run_backtest(
        MVRVStrategy(),
        btc_df=btc_df,
        start_date="2022-01-01",
        end_date="2023-05-01",
        strategy_label="mvrv-test",
    )

    assert len(result.spd_table) > 0
    assert np.isfinite(result.exp_decay_percentile)
