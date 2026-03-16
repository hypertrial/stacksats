from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from stacksats.feature_providers import BRKOverlayFeatureProvider, _rolling_zscore_pl
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.feature_registry import DEFAULT_FEATURE_REGISTRY
from stacksats.loader import load_strategy
from stacksats.model_development import precompute_features
from stacksats.model_development_allocation import (
    _compute_stable_signal,
    allocate_sequential_stable,
)
from stacksats.framework_contract import apply_clipped_weight
from stacksats.model_development_helpers import (
    classify_mvrv_zone,
    compute_mvrv_volatility,
    compute_signal_confidence,
    rolling_percentile,
    zscore,
)
from stacksats.prelude import DATE_COL, compute_cycle_spd
from stacksats.runner import StrategyRunner
from stacksats.runner_validation import WIN_RATE_TOLERANCE
from stacksats.strategy_types import BaseStrategy, StrategyContext, ValidationConfig


class _NoCacheRunner(StrategyRunner):
    def _materialize_strategy_features(self, *args, **kwargs):
        kwargs["cache_namespace"] = None
        return super()._materialize_strategy_features(*args, **kwargs)

    def _compute_strategy_weights(self, *args, **kwargs):
        kwargs["cache_namespace"] = None
        return super()._compute_strategy_weights(*args, **kwargs)


class _TiltedStrategy(BaseStrategy):
    strategy_id = "tilted"
    version = "0.0.1"

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("price_usd",)

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del ctx, signals
        values = np.linspace(-0.5, 0.5, features_df.height)
        return pl.DataFrame({"date": features_df["date"], "value": values})


def _precompute_features_reference(df: pl.DataFrame) -> pl.DataFrame:
    price_col = "price_usd"
    mvrv_col = "mvrv"
    ma_window = 200
    mvrv_rolling_window = 365
    mvrv_cycle_window = 1461
    mvrv_gradient_window = 30
    mvrv_accel_window = 14
    mvrv_volatility_window = 90
    cutoff = dt.datetime(2010, 7, 18)
    frame = df.filter(pl.col("date") >= cutoff).sort("date")
    if frame.is_empty():
        return pl.DataFrame(schema={"date": pl.Datetime("us")})

    price = frame[price_col].cast(pl.Float64, strict=False).to_numpy()
    n = len(price)
    ma_arr = np.full(n, np.nan)
    for i in range(ma_window // 2, n):
        start = max(0, i - ma_window + 1)
        ma_arr[i] = np.nanmean(price[start : i + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = np.clip((price / ma_arr) - 1, -1, 1)
    price_vs_ma = np.where(np.isfinite(price_vs_ma), price_vs_ma, 0)

    if mvrv_col in frame.columns:
        mvrv_s = pl.Series("mvrv", frame[mvrv_col].to_numpy())
        mvrv_z = zscore(mvrv_s, mvrv_rolling_window)
        mvrv_z_arr = np.clip(mvrv_z.to_numpy(), -4, 4)
        mvrv_z_arr = np.where(np.isfinite(mvrv_z_arr), mvrv_z_arr, 0)
        mvrv_pct_s = rolling_percentile(mvrv_s, mvrv_cycle_window)
        mvrv_pct = mvrv_pct_s.to_numpy()
        mvrv_pct = np.where(np.isfinite(mvrv_pct), mvrv_pct, 0.5)

        gradient_raw = np.zeros(n)
        gradient_raw[mvrv_gradient_window:] = (
            mvrv_z_arr[mvrv_gradient_window:] - mvrv_z_arr[:-mvrv_gradient_window]
        )
        alpha = 2 / (mvrv_gradient_window + 1)
        gradient_smooth = np.zeros(n)
        for i in range(1, n):
            gradient_smooth[i] = (
                alpha * gradient_raw[i] + (1 - alpha) * gradient_smooth[i - 1]
            )
        mvrv_gradient = np.tanh(gradient_smooth * 2)
        mvrv_gradient = np.where(np.isfinite(mvrv_gradient), mvrv_gradient, 0)

        accel_raw = np.zeros(n)
        accel_raw[mvrv_accel_window:] = (
            mvrv_gradient[mvrv_accel_window:] - mvrv_gradient[:-mvrv_accel_window]
        )
        accel_alpha = 2 / (mvrv_accel_window + 1)
        mvrv_acceleration = np.zeros(n)
        for i in range(1, n):
            mvrv_acceleration[i] = (
                accel_alpha * accel_raw[i]
                + (1 - accel_alpha) * mvrv_acceleration[i - 1]
            )
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3)
        mvrv_acceleration = np.where(np.isfinite(mvrv_acceleration), mvrv_acceleration, 0)

        mvrv_zone = classify_mvrv_zone(mvrv_z_arr)
        mvrv_volatility = compute_mvrv_volatility(
            pl.Series("z", mvrv_z_arr), mvrv_volatility_window
        ).to_numpy()
        mvrv_volatility = np.where(np.isfinite(mvrv_volatility), mvrv_volatility, 0.5)
        signal_confidence = np.full(n, 0.5)
    else:
        mvrv_z_arr = np.zeros(n)
        mvrv_pct = np.full(n, 0.5)
        mvrv_gradient = np.zeros(n)
        mvrv_acceleration = np.zeros(n)
        mvrv_zone = np.zeros(n, dtype=int)
        mvrv_volatility = np.full(n, 0.5)
        signal_confidence = np.full(n, 0.5)

    features = pl.DataFrame({
        "date": frame["date"].to_list(),
        price_col: price,
        "price_ma": ma_arr,
        "price_vs_ma": price_vs_ma,
        "mvrv_zscore": mvrv_z_arr,
        "mvrv_gradient": mvrv_gradient,
        "mvrv_percentile": mvrv_pct,
        "mvrv_acceleration": mvrv_acceleration,
        "mvrv_zone": mvrv_zone,
        "mvrv_volatility": mvrv_volatility,
        "signal_confidence": signal_confidence,
    })
    signal_cols = [
        "price_vs_ma",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_percentile",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
    ]
    for col in signal_cols:
        features = features.with_columns(pl.col(col).shift(1).alias(col))
    features = features.with_columns(
        pl.col("mvrv_percentile").fill_null(0.5),
        pl.col("mvrv_zone").fill_null(0),
        pl.col("mvrv_volatility").fill_null(0.5),
    ).fill_null(0)
    sc = compute_signal_confidence(
        features["mvrv_zscore"].to_numpy(),
        features["mvrv_percentile"].to_numpy(),
        features["mvrv_gradient"].to_numpy(),
        features["price_vs_ma"].to_numpy(),
    )
    return features.with_columns(pl.Series("signal_confidence", sc))


def _overlay_reference(btc_df: pl.DataFrame) -> pl.DataFrame:
    price_arr = btc_df["price_usd"].cast(pl.Float64, strict=False).to_numpy()
    price_safe = np.where(price_arr > 0, price_arr, 1.0)
    price_log = np.log(price_safe)
    diff_30 = np.zeros_like(price_log)
    diff_30[30:] = price_log[30:] - price_log[:-30]
    diff_90 = np.zeros_like(price_log)
    diff_90[90:] = price_log[90:] - price_log[:-90]
    mom_30 = _rolling_zscore_pl(pl.Series("d30", diff_30), 365)
    mom_90 = _rolling_zscore_pl(pl.Series("d90", diff_90), 365)

    n = btc_df.height
    features = pl.DataFrame({
        DATE_COL: btc_df[DATE_COL].to_list(),
        "brk_flow": [0.0] * n,
        "brk_supply_pressure": [0.0] * n,
        "brk_activity_div": [0.0] * n,
        "brk_roi_context": [0.0] * n,
        "brk_liquidity_impulse": [0.0] * n,
        "brk_miner_pressure": [0.0] * n,
        "brk_hash_momentum": [0.0] * n,
        "brk_sentiment": [0.0] * n,
    })
    if "adjusted_sopr" in btc_df.columns and "adjusted_sopr_7d_ema" in btc_df.columns:
        sopr = btc_df["adjusted_sopr"].cast(pl.Float64, strict=False).fill_null(0)
        sopr_ema = btc_df["adjusted_sopr_7d_ema"].cast(pl.Float64, strict=False).fill_null(0)
        diff = sopr - sopr_ema
        features = features.with_columns(
            _rolling_zscore_pl(diff, 240).alias("brk_flow"),
            _rolling_zscore_pl((-0.65 * sopr) + (-0.35 * sopr_ema), 365).alias("brk_roi_context"),
        )
    else:
        features = features.with_columns(((-0.65 * mom_30) + (-0.35 * mom_90)).alias("brk_roi_context"))
    if "realized_cap_growth_rate" in btc_df.columns and "market_cap_growth_rate" in btc_df.columns:
        mkt = btc_df["market_cap_growth_rate"].cast(pl.Float64, strict=False).fill_null(0)
        real = btc_df["realized_cap_growth_rate"].cast(pl.Float64, strict=False).fill_null(0)
        features = features.with_columns(
            _rolling_zscore_pl(mkt - real, 365).alias("brk_supply_pressure"),
        )
    flow_fast = features["brk_flow"].rolling_mean(window_size=7, min_samples=3)
    flow_slow = features["brk_flow"].rolling_mean(window_size=30, min_samples=10)
    features = features.with_columns(
        _rolling_zscore_pl(flow_fast, 120).alias("brk_netflow_fast"),
        _rolling_zscore_pl(flow_slow, 240).alias("brk_netflow_slow"),
        _rolling_zscore_pl(flow_fast - flow_slow, 180).alias("brk_netflow_slope"),
        _rolling_zscore_pl(flow_fast, 180).alias("brk_netflow"),
    )
    activity_level = _rolling_zscore_pl(features["brk_activity_div"] + mom_30, 365)
    features = features.with_columns(
        activity_level.alias("brk_activity_level"),
        features["brk_activity_div"].alias("brk_activity_div_fast"),
        (activity_level - mom_90).alias("brk_activity_div_slow"),
        _rolling_zscore_pl(features["brk_liquidity_impulse"], 180).alias("brk_liquidity_level"),
        _rolling_zscore_pl(features["brk_supply_pressure"], 240).alias("brk_exchange_share_level"),
    )
    exchange_level = features["brk_exchange_share_level"]
    features = features.with_columns(
        _rolling_zscore_pl(exchange_level.diff(30), 240).alias("brk_exchange_share_delta"),
        _rolling_zscore_pl(exchange_level, 365).alias("brk_exchange_share"),
        _rolling_zscore_pl(mom_30, 365).alias("brk_roi30"),
        _rolling_zscore_pl(mom_90, 365).alias("brk_roi1y"),
    )
    lagged_cols = [c for c in features.columns if c != DATE_COL]
    features = features.with_columns(
        [pl.col(c).shift(1).fill_null(0.0).alias(c) for c in lagged_cols]
    ).fill_null(0)
    float_cols = [
        c
        for c in features.columns
        if c != DATE_COL and features.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    return features.with_columns([pl.col(c).clip(-6.0, 6.0) for c in float_cols])


def _allocate_reference(raw: np.ndarray, n_past: int) -> np.ndarray:
    n = len(raw)
    if n == 0:
        return np.array([], dtype=float)
    if n_past <= 0:
        return np.full(n, 1.0 / n, dtype=float)

    n_past = min(n_past, n)
    raw_arr = np.asarray(raw, dtype=float)
    out = np.zeros(n, dtype=float)
    base_weight = 1.0 / n
    remaining_budget = 1.0
    for i in range(n_past):
        signal = float(_compute_stable_signal(raw_arr[: i + 1])[-1])
        proposed = signal * base_weight
        clipped, remaining_budget = apply_clipped_weight(
            proposed,
            remaining_budget,
            n - i,
            enforce_contract_bounds=n == ALLOCATION_SPAN_DAYS,
        )
        out[i] = clipped
    n_future = n - n_past
    if n_future > 0:
        out[n_past:] = max(remaining_budget, 0.0) / n_future
    elif n > 0:
        out[-1] += max(remaining_budget, 0.0)
    return out


def _compute_cycle_spd_reference(
    dataframe: pl.DataFrame,
    strategy_function,
    *,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    start_ts = dt.datetime.strptime(start_date[:10], "%Y-%m-%d")
    end_ts = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
    max_start_date = end_ts - dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)
    start_dates = pl.datetime_range(start_ts, max_start_date, interval="1d", eager=True)
    rows: list[dict[str, float | str]] = []
    for window_start in start_dates.to_list():
        window_end = window_start + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)
        price_slice = dataframe.filter(
            (pl.col(DATE_COL) >= window_start) & (pl.col(DATE_COL) <= window_end)
        )
        if price_slice.height != ALLOCATION_SPAN_DAYS:
            continue
        weight_df = strategy_function(price_slice)
        merged = price_slice.select(DATE_COL).join(
            weight_df.select([DATE_COL, "weight"]),
            on=DATE_COL,
            how="left",
        )
        weights = merged["weight"].fill_null(0.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            weights = pl.Series("weight", [1.0 / price_slice.height] * price_slice.height)
        else:
            weights = weights / total
        inv_price = 1e8 / price_slice["price_usd"].to_numpy()
        min_spd = float(np.nanmin(inv_price))
        max_spd = float(np.nanmax(inv_price))
        uniform_spd = float(np.nanmean(inv_price))
        dynamic_spd = float(np.sum(weights.to_numpy() * inv_price))
        span = max_spd - min_spd
        if span > 0.0:
            uniform_pct = (uniform_spd - min_spd) / span * 100
            dynamic_pct = (dynamic_spd - min_spd) / span * 100
        else:
            uniform_pct = float("nan")
            dynamic_pct = float("nan")
        rows.append(
            {
                "window": f"{window_start:%Y-%m-%d} → {window_end:%Y-%m-%d}",
                "uniform_percentile": uniform_pct,
                "dynamic_percentile": dynamic_pct,
                "excess_percentile": dynamic_pct - uniform_pct,
            }
        )
    return pl.DataFrame(rows)


def test_allocate_sequential_stable_matches_reference_kernel() -> None:
    raw = np.linspace(0.75, 1.25, ALLOCATION_SPAN_DAYS, dtype=float)
    for n_past in [0, 1, 90, ALLOCATION_SPAN_DAYS // 2, ALLOCATION_SPAN_DAYS]:
        current = allocate_sequential_stable(raw, n_past=n_past)
        reference = _allocate_reference(raw, n_past=n_past)
        np.testing.assert_allclose(current, reference, atol=1e-12, rtol=0.0)


def test_compute_cycle_spd_matches_reference_window_metrics(sample_btc_df) -> None:
    btc_df = sample_btc_df.select(["date", "price_usd"]).slice(0, 420)
    start_date = btc_df["date"][0].strftime("%Y-%m-%d")
    end_date = btc_df["date"][-1].strftime("%Y-%m-%d")

    def _strategy(window: pl.DataFrame) -> pl.DataFrame:
        weights = np.linspace(1.0, 2.0, window.height, dtype=float)
        weights /= weights.sum()
        return pl.DataFrame({"date": window["date"], "weight": weights})

    current = compute_cycle_spd(
        btc_df,
        _strategy,
        features_df=btc_df,
        start_date=start_date,
        end_date=end_date,
    ).select(["window", "uniform_percentile", "dynamic_percentile", "excess_percentile"])
    reference = _compute_cycle_spd_reference(
        btc_df,
        _strategy,
        start_date=start_date,
        end_date=end_date,
    )

    assert current["window"].to_list() == reference["window"].to_list()
    np.testing.assert_allclose(
        current["uniform_percentile"].to_numpy(),
        reference["uniform_percentile"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        current["dynamic_percentile"].to_numpy(),
        reference["dynamic_percentile"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        current["excess_percentile"].to_numpy(),
        reference["excess_percentile"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


def test_precompute_features_matches_reference_parity(sample_btc_df) -> None:
    btc_df = sample_btc_df.select(["date", "price_usd", "mvrv"]).slice(0, 520)

    current = precompute_features(btc_df)
    reference = _precompute_features_reference(btc_df)

    assert current.columns == reference.columns
    assert current.height == reference.height
    assert_frame_equal(current, reference, check_dtypes=False, atol=1e-9, rtol=0.0)


def test_precompute_features_matches_reference_without_mvrv(sample_btc_df) -> None:
    btc_df = sample_btc_df.select(["date", "price_usd"]).slice(0, 520)

    current = precompute_features(btc_df)
    reference = _precompute_features_reference(btc_df)

    assert current.columns == reference.columns
    assert current.height == reference.height
    assert_frame_equal(current, reference, check_dtypes=False, atol=1e-9, rtol=0.0)


def test_brk_overlay_provider_matches_reference_with_optional_columns() -> None:
    provider = BRKOverlayFeatureProvider()
    days = 420
    btc_df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 1) + dt.timedelta(days=days - 1),
                interval="1d",
                eager=True,
            ),
            "price_usd": np.linspace(40_000.0, 60_000.0, days),
            "mvrv": np.linspace(0.8, 2.2, days),
            "adjusted_sopr": np.linspace(0.9, 1.1, days),
            "adjusted_sopr_7d_ema": np.linspace(0.92, 1.08, days),
            "realized_cap_growth_rate": np.linspace(-0.1, 0.15, days),
            "market_cap_growth_rate": np.linspace(-0.08, 0.2, days),
        }
    )

    current = provider.materialize(
        btc_df,
        start_date=btc_df["date"][0],
        end_date=btc_df["date"][-1],
        as_of_date=btc_df["date"][-1],
    )
    reference = _overlay_reference(btc_df)

    assert current.columns == reference.columns
    assert current.height == reference.height
    assert_frame_equal(current, reference, check_dtypes=False, atol=1e-12, rtol=0.0)


def test_brk_overlay_provider_matches_reference_without_optional_columns() -> None:
    provider = BRKOverlayFeatureProvider()
    days = 420
    btc_df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 1) + dt.timedelta(days=days - 1),
                interval="1d",
                eager=True,
            ),
            "price_usd": np.linspace(40_000.0, 60_000.0, days),
            "mvrv": np.linspace(0.8, 2.2, days),
        }
    )

    current = provider.materialize(
        btc_df,
        start_date=btc_df["date"][0],
        end_date=btc_df["date"][-1],
        as_of_date=btc_df["date"][-1],
    )
    reference = _overlay_reference(btc_df)

    assert current.columns == reference.columns
    assert current.height == reference.height
    assert_frame_equal(current, reference, check_dtypes=False, atol=1e-12, rtol=0.0)


def test_brk_overlay_materialization_preserves_columns_for_single_row_prefix() -> None:
    strategy = load_strategy("stacksats.strategies.model_example:ExampleMVRVStrategy")
    btc_df = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1)],
            "price_usd": [42000.0],
            "mvrv": [1.2],
        }
    )

    frame = DEFAULT_FEATURE_REGISTRY.materialize_for_strategy(
        strategy,
        btc_df,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 1),
        current_date=dt.datetime(2024, 1, 1),
    )

    assert frame.height == 1
    assert "brk_netflow_fast" in frame.columns
    assert "brk_activity_level" in frame.columns


def test_validate_uses_runtime_cache_without_changing_result(sample_btc_df) -> None:
    btc_df = sample_btc_df.select(["date", "price_usd", "mvrv"])
    config = ValidationConfig(
        start_date="2021-01-01",
        end_date="2023-12-31",
        min_win_rate=0.0,
        strict=False,
    )

    cached_runner = StrategyRunner.from_dataframe(btc_df)
    uncached_runner = _NoCacheRunner(data_provider=cached_runner._data_provider)

    cached = cached_runner.validate(_TiltedStrategy(), config, btc_df=btc_df)
    uncached = uncached_runner.validate(_TiltedStrategy(), config, btc_df=btc_df)

    assert cached.passed == uncached.passed
    assert cached.forward_leakage_ok == uncached.forward_leakage_ok
    assert cached.weight_constraints_ok == uncached.weight_constraints_ok
    assert cached.win_rate == pytest.approx(uncached.win_rate, abs=WIN_RATE_TOLERANCE)

    hot_path = cached.diagnostics["hot_path_profile"]
    assert hot_path["materialize_cache_hits"] > 0
    assert hot_path["compute_weights_cache_hits"] > 0


def test_validate_empty_range_does_not_leave_runtime_cache(sample_btc_df) -> None:
    runner = StrategyRunner.from_dataframe(sample_btc_df.select(["date", "price_usd", "mvrv"]))
    result = runner.validate(
        _TiltedStrategy(),
        ValidationConfig(
            start_date="1999-01-01",
            end_date="1999-12-31",
            min_win_rate=0.0,
            strict=False,
        ),
        btc_df=sample_btc_df.select(["date", "price_usd", "mvrv"]),
    )

    assert result.passed is False
    assert runner._active_runtime_cache() is None
