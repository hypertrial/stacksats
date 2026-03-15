from __future__ import annotations

import warnings

import numpy as np

from stacksats.feature_providers import BRKOverlayFeatureProvider, CoreModelFeatureProvider
from tests.test_helpers import btc_frame


def _btc_frame(days: int = 420):
    frame = btc_frame(start="2024-01-01", days=days)
    return frame.with_columns(
        PriceUSD=frame["price_usd"],
        FlowInExUSD=np.linspace(1_000_000.0, 2_000_000.0, days),
        FlowOutExUSD=np.linspace(900_000.0, 1_900_000.0, days),
        CapMrktCurUSD=np.linspace(5e11, 6e11, days),
        AdrActCnt=np.linspace(-10.0, 1000.0, days),
        TxCnt=np.linspace(-5.0, 2000.0, days),
        TxTfrCnt=np.linspace(-3.0, 3000.0, days),
        FeeTotNtv=np.linspace(-1.0, 50.0, days),
        volume_reported_spot_usd_1d=np.linspace(-1000.0, 1e9, days),
        SplyExNtv=np.linspace(1e6, 2e6, days),
        SplyCur=np.linspace(19e6, 20e6, days),
        IssTotUSD=np.linspace(1e7, 2e7, days),
        HashRate=np.linspace(-1.0, 4e8, days),
        ROI30d=np.linspace(-0.2, 0.2, days),
        ROI1yr=np.linspace(-0.5, 0.5, days),
    )


def test_overlay_provider_handles_non_positive_values_without_runtime_warnings() -> None:
    provider = BRKOverlayFeatureProvider()
    btc_df = _btc_frame()
    dates = btc_df["date"].to_list()
    btc_df = btc_df.with_columns(
        price_usd=np.where(np.arange(btc_df.height) == 1, -5.0, btc_df["price_usd"].to_numpy()),
        PriceUSD=np.where(np.arange(btc_df.height) == 0, -10.0, btc_df["PriceUSD"].to_numpy()),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        features = provider.materialize(
            btc_df,
            start_date=dates[20],
            end_date=dates[-1],
            as_of_date=dates[-1],
        )

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
    assert features.height > 0
    assert "brk_roi_context" in features.columns
    finite_roi = features["brk_roi_context"].drop_nulls()
    assert finite_roi.len() > 0
    assert np.isfinite(finite_roi.to_numpy()).all()


def test_core_provider_cache_invalidates_after_price_change() -> None:
    provider = CoreModelFeatureProvider()
    btc_df = _btc_frame()
    dates = btc_df["date"].to_list()
    first = provider.materialize(
        btc_df,
        start_date=dates[20],
        end_date=dates[-1],
        as_of_date=dates[-1],
    )

    btc_df = btc_df.with_columns(
        price_usd=np.where(np.arange(btc_df.height) == (btc_df.height - 1), 2_000_000.0, btc_df["price_usd"].to_numpy())
    )
    second = provider.materialize(
        btc_df,
        start_date=dates[20],
        end_date=dates[-1],
        as_of_date=dates[-1],
    )

    assert not first.equals(second)
