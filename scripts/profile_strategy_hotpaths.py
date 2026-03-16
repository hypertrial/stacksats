#!/usr/bin/env python3
# ruff: noqa: E402
"""Profile strategy validation, backtest windowing, and allocation hot paths."""

from __future__ import annotations

import cProfile
import datetime as dt
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_all_strategies import ROOT as AUDIT_ROOT
from run_all_strategies import STRATEGY_SPECS, _load_audit_dataset
from stacksats.feature_providers import BRKOverlayFeatureProvider, CoreModelFeatureProvider
from stacksats.feature_registry import DEFAULT_FEATURE_REGISTRY
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS, ALLOCATION_WINDOW_OFFSET
from stacksats.loader import load_strategy
from stacksats.model_development import precompute_features
from stacksats.model_development_allocation import (
    _compute_stable_signal,
    allocate_sequential_stable,
)
from stacksats.prelude import DATE_COL, WINDOW_DAYS, compute_cycle_spd
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import ValidationConfig
from stacksats.framework_contract import apply_clipped_weight


class NoCacheRunner(StrategyRunner):
    """Runner variant that disables runtime caching for profiling comparisons."""

    def _materialize_strategy_features(self, *args, **kwargs):
        kwargs["cache_namespace"] = None
        return super()._materialize_strategy_features(*args, **kwargs)

    def _compute_strategy_weights(self, *args, **kwargs):
        kwargs["cache_namespace"] = None
        return super()._compute_strategy_weights(*args, **kwargs)


def _profile_call(label: str, fn, *args, **kwargs) -> dict[str, object]:
    """Run deterministic wall-time and cProfile measurement for one call."""
    profiler = cProfile.Profile()
    started_at = time.perf_counter()
    result = profiler.runcall(fn, *args, **kwargs)
    wall_seconds = time.perf_counter() - started_at
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative")
    stats.print_stats(10)
    top_entries: list[dict[str, object]] = []
    sorted_stats = sorted(
        stats.stats.items(),
        key=lambda item: float(item[1][3]),
        reverse=True,
    )
    for (filename, lineno, funcname), stat in sorted_stats[:10]:
        ccalls, ncalls, tottime, cumtime, _ = stat
        top_entries.append(
            {
                "function": f"{Path(filename).name}:{lineno}:{funcname}",
                "primitive_calls": int(ccalls),
                "total_calls": int(ncalls),
                "total_seconds": round(float(tottime), 6),
                "cumulative_seconds": round(float(cumtime), 6),
            }
        )
    return {
        "label": label,
        "wall_seconds": wall_seconds,
        "top_cumulative": top_entries,
        "stats_text": stats_stream.getvalue(),
        "result": result,
    }


def _allocate_reference(
    raw: np.ndarray,
    n_past: int,
) -> np.ndarray:
    """Reference allocation kernel before the incremental signal optimization."""
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
    """Reference implementation before positional window slicing optimization."""
    start_ts = dt.datetime.strptime(start_date[:10], "%Y-%m-%d")
    end_ts = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
    dataframe = dataframe.sort(DATE_COL)
    start_dates = pl.datetime_range(
        start_ts,
        end_ts - ALLOCATION_WINDOW_OFFSET,
        interval="1d",
        eager=True,
    )
    rows: list[dict[str, float | str]] = []
    for window_start in start_dates.to_list():
        window_end = window_start + ALLOCATION_WINDOW_OFFSET
        price_slice = dataframe.filter(
            (pl.col(DATE_COL) >= window_start) & (pl.col(DATE_COL) <= window_end)
        )
        if price_slice.height != WINDOW_DAYS:
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
        if span > 0:
            uniform_pct = (uniform_spd - min_spd) / span * 100
            dynamic_pct = (dynamic_spd - min_spd) / span * 100
        else:
            uniform_pct = float("nan")
            dynamic_pct = float("nan")
        rows.append(
            {
                "window": f"{window_start:%Y-%m-%d} → {window_end:%Y-%m-%d}",
                "dynamic_percentile": dynamic_pct,
                "uniform_percentile": uniform_pct,
            }
        )
    return pl.DataFrame(rows)


def _uniform_strategy_fn(window: pl.DataFrame) -> pl.DataFrame:
    n = window.height
    return pl.DataFrame({DATE_COL: window[DATE_COL], "weight": [1.0 / n] * n})


def main() -> int:
    pq_path, btc_df, start_date, end_date = _load_audit_dataset(AUDIT_ROOT)
    strategy = load_strategy(STRATEGY_SPECS[1])
    start_ts = dt.datetime.strptime(start_date[:10], "%Y-%m-%d")
    end_ts = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
    config = ValidationConfig(
        start_date=start_date,
        end_date=end_date,
        min_win_rate=0.0,
        strict=False,
    )

    validate_uncached = _profile_call(
        "validate_uncached",
        NoCacheRunner().validate,
        strategy,
        config,
        btc_df=btc_df,
    )
    validate_cached = _profile_call(
        "validate_cached",
        StrategyRunner().validate,
        strategy,
        config,
        btc_df=btc_df,
    )
    precompute_current = _profile_call(
        "precompute_features_current",
        precompute_features,
        btc_df,
    )
    core_provider = CoreModelFeatureProvider()
    core_provider_current = _profile_call(
        "core_provider_materialize_current",
        core_provider.materialize,
        btc_df,
        start_date=start_ts,
        end_date=end_ts,
        as_of_date=end_ts,
    )
    overlay_provider = BRKOverlayFeatureProvider()
    overlay_provider_current = _profile_call(
        "overlay_provider_materialize_current",
        overlay_provider.materialize,
        btc_df,
        start_date=start_ts,
        end_date=end_ts,
        as_of_date=end_ts,
    )
    registry_current = _profile_call(
        "registry_materialize_current",
        DEFAULT_FEATURE_REGISTRY.materialize_for_strategy,
        strategy,
        btc_df,
        start_date=start_ts,
        end_date=end_ts,
        current_date=end_ts,
    )
    backtest_reference = _profile_call(
        "compute_cycle_spd_reference",
        _compute_cycle_spd_reference,
        btc_df.select(["date", "price_usd"]).sort("date"),
        _uniform_strategy_fn,
        start_date=start_date,
        end_date=end_date,
    )
    backtest_current = _profile_call(
        "compute_cycle_spd_current",
        compute_cycle_spd,
        btc_df.select(["date", "price_usd"]).sort("date"),
        _uniform_strategy_fn,
        start_date=start_date,
        end_date=end_date,
    )

    raw = np.linspace(0.75, 1.25, ALLOCATION_SPAN_DAYS, dtype=float)
    allocation_reference = _profile_call(
        "allocate_reference",
        _allocate_reference,
        raw,
        ALLOCATION_SPAN_DAYS,
    )
    allocation_current = _profile_call(
        "allocate_current",
        allocate_sequential_stable,
        raw,
        ALLOCATION_SPAN_DAYS,
    )

    payload = {
        "parquet_path": str(pq_path),
        "strategy": strategy.metadata().strategy_id,
        "date_range": {"start_date": start_date, "end_date": end_date},
        "validate": {
            "uncached_wall_seconds": round(validate_uncached["wall_seconds"], 3),
            "cached_wall_seconds": round(validate_cached["wall_seconds"], 3),
            "speedup_multiple": round(
                validate_uncached["wall_seconds"] / validate_cached["wall_seconds"],
                3,
            ),
            "uncached_top_cumulative": validate_uncached["top_cumulative"],
            "cached_top_cumulative": validate_cached["top_cumulative"],
        },
        "precompute_features": {
            "current_wall_seconds": round(precompute_current["wall_seconds"], 3),
            "current_top_cumulative": precompute_current["top_cumulative"],
        },
        "providers": {
            "core_model_features_v1": {
                "current_wall_seconds": round(core_provider_current["wall_seconds"], 3),
                "current_top_cumulative": core_provider_current["top_cumulative"],
            },
            "brk_overlay_v1": {
                "current_wall_seconds": round(overlay_provider_current["wall_seconds"], 3),
                "current_top_cumulative": overlay_provider_current["top_cumulative"],
            },
        },
        "registry_materialization": {
            "current_wall_seconds": round(registry_current["wall_seconds"], 3),
            "current_top_cumulative": registry_current["top_cumulative"],
        },
        "compute_cycle_spd": {
            "reference_wall_seconds": round(backtest_reference["wall_seconds"], 3),
            "current_wall_seconds": round(backtest_current["wall_seconds"], 3),
            "speedup_multiple": round(
                backtest_reference["wall_seconds"] / backtest_current["wall_seconds"],
                3,
            ),
            "reference_top_cumulative": backtest_reference["top_cumulative"],
            "current_top_cumulative": backtest_current["top_cumulative"],
        },
        "allocate_sequential_stable": {
            "reference_wall_seconds": round(allocation_reference["wall_seconds"], 6),
            "current_wall_seconds": round(allocation_current["wall_seconds"], 6),
            "speedup_multiple": round(
                allocation_reference["wall_seconds"] / allocation_current["wall_seconds"],
                3,
            ),
            "reference_top_cumulative": allocation_reference["top_cumulative"],
            "current_top_cumulative": allocation_current["top_cumulative"],
        },
    }

    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "strategy_hotpath_profile.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Strategy: {strategy.metadata().strategy_id}")
    print(
        "validate wall seconds: "
        f"uncached={payload['validate']['uncached_wall_seconds']:.3f}, "
        f"cached={payload['validate']['cached_wall_seconds']:.3f}, "
        f"speedup={payload['validate']['speedup_multiple']:.3f}x"
    )
    print(
        "precompute_features wall seconds: "
        f"current={payload['precompute_features']['current_wall_seconds']:.3f}"
    )
    print(
        "registry materialization wall seconds: "
        f"current={payload['registry_materialization']['current_wall_seconds']:.3f}"
    )
    print(
        "compute_cycle_spd wall seconds: "
        f"reference={payload['compute_cycle_spd']['reference_wall_seconds']:.3f}, "
        f"current={payload['compute_cycle_spd']['current_wall_seconds']:.3f}, "
        f"speedup={payload['compute_cycle_spd']['speedup_multiple']:.3f}x"
    )
    print(
        "allocate_sequential_stable wall seconds: "
        f"reference={payload['allocate_sequential_stable']['reference_wall_seconds']:.6f}, "
        f"current={payload['allocate_sequential_stable']['current_wall_seconds']:.6f}, "
        f"speedup={payload['allocate_sequential_stable']['speedup_multiple']:.3f}x"
    )
    print(f"Wrote profile summary to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
