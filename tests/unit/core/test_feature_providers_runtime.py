from __future__ import annotations

import warnings

import duckdb
import numpy as np
import pandas as pd
import pytest

from stacksats.feature_providers import (
    BRKOverlayFeatureProvider,
    CoreModelFeatureProvider,
)


def _btc_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "price_usd": np.linspace(10000.0, 20000.0, n),
            "mvrv": np.linspace(1.0, 2.0, n),
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


def _create_brk_overlay_duckdb_fixture(path: str) -> None:
    con = duckdb.connect(path)
    try:
        con.execute(
            "CREATE TABLE metrics_distribution (date_day DATE, metric VARCHAR, value DOUBLE)"
        )
        con.execute(
            "CREATE TABLE metrics_supply (date_day DATE, metric VARCHAR, value DOUBLE)"
        )
        con.execute(
            "CREATE TABLE metrics_transactions (date_day DATE, metric VARCHAR, value DOUBLE)"
        )
        con.execute(
            "CREATE TABLE metrics_blocks (date_day DATE, metric VARCHAR, value DOUBLE)"
        )

        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        distribution_metrics = (
            "adjusted_sopr",
            "adjusted_sopr_7d_ema",
            "net_sentiment",
            "greed_index",
            "pain_index",
        )
        supply_metrics = ("realized_cap_growth_rate", "market_cap_growth_rate")
        tx_metrics = ("tx_count_pct10", "annualized_volume_usd")
        block_metrics = ("hash_rate_1y_sma", "subsidy_usd_average")

        rows_distribution: list[tuple[object, str, float]] = []
        rows_supply: list[tuple[object, str, float]] = []
        rows_tx: list[tuple[object, str, float]] = []
        rows_blocks: list[tuple[object, str, float]] = []
        for day_idx, day in enumerate(dates):
            day_factor = float(day_idx + 1)
            for metric_idx, metric in enumerate(distribution_metrics):
                rows_distribution.append(
                    (day.date(), metric, day_factor * float(metric_idx + 1))
                )
            for metric_idx, metric in enumerate(supply_metrics):
                rows_supply.append(
                    (day.date(), metric, day_factor * float(metric_idx + 2))
                )
            for metric_idx, metric in enumerate(tx_metrics):
                rows_tx.append((day.date(), metric, day_factor * float(metric_idx + 3)))
            for metric_idx, metric in enumerate(block_metrics):
                rows_blocks.append(
                    (day.date(), metric, day_factor * float(metric_idx + 4))
                )

        con.executemany("INSERT INTO metrics_distribution VALUES (?, ?, ?)", rows_distribution)
        con.executemany("INSERT INTO metrics_supply VALUES (?, ?, ?)", rows_supply)
        con.executemany("INSERT INTO metrics_transactions VALUES (?, ?, ?)", rows_tx)
        con.executemany("INSERT INTO metrics_blocks VALUES (?, ?, ?)", rows_blocks)
    finally:
        con.close()


@pytest.fixture(autouse=True)
def _overlay_duckdb_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "overlay.duckdb"
    _create_brk_overlay_duckdb_fixture(str(db_path))
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))


def test_overlay_provider_handles_non_positive_values_without_runtime_warnings() -> None:
    provider = BRKOverlayFeatureProvider()
    idx = pd.date_range("2024-01-01", periods=420, freq="D")
    btc_df = _btc_frame(idx)
    btc_df.loc[idx[0], "PriceUSD"] = -10.0
    btc_df.loc[idx[1], "price_usd"] = -5.0

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

    btc_df.loc[idx[-1], "price_usd"] = 2_000_000.0
    second = provider.materialize(
        btc_df,
        start_date=idx[20],
        end_date=idx[-1],
        as_of_date=idx[-1],
    )

    assert not first.equals(second)
