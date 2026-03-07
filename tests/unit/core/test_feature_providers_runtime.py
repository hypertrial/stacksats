from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stacksats.feature_providers import (
    CoinMetricsOverlayFeatureProvider,
    CoreModelFeatureProvider,
)


def _btc_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000.0, 20000.0, n),
            "CapMVRVCur": np.linspace(1.0, 2.0, n),
            "PriceUSD": np.linspace(10000.0, 20000.0, n),
            "FlowInExUSD": np.linspace(1_000_000.0, 2_000_000.0, n),
            "FlowOutExUSD": np.linspace(900_000.0, 1_900_000.0, n),
            "CapMrktCurUSD": np.linspace(5e11, 6e11, n),
            "AdrActCnt": np.linspace(-10.0, 1000.0, n),
            "TxCnt": np.linspace(-5.0, 2000.0, n),
            "TxTfrCnt": np.linspace(-3.0, 3000.0, n),
            "FeeTotNtv": np.linspace(-1.0, 50.0, n),
            "volume_reported_spot_usd_1d": np.linspace(-1000.0, 1e9, n),
            "SplyExNtv": np.linspace(1e6, 2e6, n),
            "SplyCur": np.linspace(19e6, 20e6, n),
            "IssTotUSD": np.linspace(1e7, 2e7, n),
            "HashRate": np.linspace(-1.0, 4e8, n),
            "ROI30d": np.linspace(-0.2, 0.2, n),
            "ROI1yr": np.linspace(-0.5, 0.5, n),
        },
        index=index,
    )


def test_overlay_provider_handles_non_positive_values_without_runtime_warnings() -> None:
    provider = CoinMetricsOverlayFeatureProvider()
    idx = pd.date_range("2024-01-01", periods=420, freq="D")
    btc_df = _btc_frame(idx)
    btc_df.loc[idx[0], "PriceUSD"] = -10.0
    btc_df.loc[idx[1], "PriceUSD_coinmetrics"] = -5.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        features = provider.materialize(
            btc_df,
            start_date=idx[20],
            end_date=idx[-1],
            as_of_date=idx[-1],
        )

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_core_provider_cache_invalidates_after_price_change() -> None:
    provider = CoreModelFeatureProvider()
    idx = pd.date_range("2024-01-01", periods=420, freq="D")
    btc_df = _btc_frame(idx)
    first = provider.materialize(
        btc_df,
        start_date=idx[20],
        end_date=idx[-1],
        as_of_date=idx[-1],
    )

    btc_df.loc[idx[-1], "PriceUSD_coinmetrics"] = 2_000_000.0
    second = provider.materialize(
        btc_df,
        start_date=idx[20],
        end_date=idx[-1],
        as_of_date=idx[-1],
    )

    assert not first.equals(second)
