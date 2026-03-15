from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.feature_registry import DEFAULT_FEATURE_REGISTRY
from stacksats.loader import load_strategy
from stacksats.model_development_allocation import (
    _compute_stable_signal,
    allocate_sequential_stable,
)
from stacksats.framework_contract import apply_clipped_weight
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
