#!/usr/bin/env python3
# ruff: noqa: E402
"""Run validate + backtest for all built-in strategies with local workspace code."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stacksats.framework_contract import ALLOCATION_WINDOW_OFFSET
from stacksats.loader import load_strategy
from stacksats.prelude import BACKTEST_END
from stacksats.runner import StrategyRunner
from stacksats.runner_helpers import weights_match
from stacksats.strategy_types import BacktestConfig, ValidationConfig

STRATEGY_SPECS = [
    "stacksats.strategies.examples:UniformStrategy",
    "stacksats.strategies.examples:SimpleZScoreStrategy",
    "stacksats.strategies.examples:MomentumStrategy",
    "stacksats.strategies.mvrv:MVRVStrategy",
    "stacksats.strategies.experimental.model_example:ExampleMVRVStrategy",
    "stacksats.strategies.experimental.model_mvrv_plus:MVRVPlusStrategy",
]

MERGED_METRICS = (
    "market_cap",
    "supply_btc",
    "mvrv",
    "adjusted_sopr",
    "adjusted_sopr_7d_ema",
    "realized_cap_growth_rate",
    "market_cap_growth_rate",
)


def _resolve_parquet_path(root_dir: Path) -> Path:
    """Use STACKSATS_ANALYTICS_PARQUET or merged_metrics*.parquet in repo root."""
    env_path = os.environ.get("STACKSATS_ANALYTICS_PARQUET")
    if env_path:
        parquet_path = Path(env_path).expanduser()
        if parquet_path.exists():
            return parquet_path.resolve()
        raise SystemExit(f"STACKSATS_ANALYTICS_PARQUET={env_path} not found.")
    matches = sorted(root_dir.glob("merged_metrics*.parquet"))
    if not matches:
        raise SystemExit(
            "No merged_metrics*.parquet in repo root. "
            "Set STACKSATS_ANALYTICS_PARQUET or place the file there."
        )
    return matches[-1].resolve()


def _convert_merged_metrics_to_brk(pq_path: Path, out_path: Path) -> Path:
    """Convert long-format merged_metrics (day_utc, metric, value) to BRK-wide format."""
    schema = pl.scan_parquet(pq_path).collect_schema()
    if set(schema.names()) != {"day_utc", "metric", "value"}:
        raise SystemExit(
            "merged_metrics parquet must have columns "
            f"(day_utc, metric, value), got {schema.names()}"
        )
    projected = (
        pl.scan_parquet(pq_path)
        .filter(pl.col("metric").is_in(MERGED_METRICS))
        .select(["day_utc", "metric", "value"])
        .collect()
    )
    wide = (
        projected
        .pivot(values="value", index="day_utc", on="metric")
        .with_columns((pl.col("market_cap") / pl.col("supply_btc")).alias("price_usd"))
        .rename({"day_utc": "date"})
        .select(
            "date",
            "price_usd",
            "mvrv",
            pl.col("adjusted_sopr").cast(pl.Float64, strict=False),
            pl.col("adjusted_sopr_7d_ema").cast(pl.Float64, strict=False),
            pl.col("realized_cap_growth_rate").cast(pl.Float64, strict=False),
            pl.col("market_cap_growth_rate").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("price_usd").is_finite() & (pl.col("price_usd") > 0))
    )
    wide.write_parquet(out_path)
    return out_path


def _load_audit_dataset(root_dir: Path) -> tuple[Path, pl.DataFrame, str, str]:
    """Resolve parquet input and derive a safe audit date range."""
    raw_pq = _resolve_parquet_path(root_dir)
    tmp_path = Path(tempfile.mkdtemp(prefix="stacksats-run-all-"))
    df_sample = pl.read_parquet(raw_pq, n_rows=1)
    if set(df_sample.columns) == {"day_utc", "metric", "value"}:
        pq_path = _convert_merged_metrics_to_brk(
            raw_pq,
            tmp_path / "bitcoin_analytics.parquet",
        )
    else:
        pq_path = raw_pq
    brk_df = pl.read_parquet(pq_path)
    last_date = brk_df["date"].max()
    latest_date = last_date.isoformat() if hasattr(last_date, "isoformat") else str(last_date)[:10]
    latest_dt = dt.datetime.strptime(latest_date[:10], "%Y-%m-%d")
    fixed_end_dt = dt.datetime.strptime(BACKTEST_END, "%Y-%m-%d")
    end_date = min(latest_dt, fixed_end_dt).strftime("%Y-%m-%d")
    start_date = "2018-01-01"
    try:
        end_dt = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
        if start_dt > end_dt or (end_dt - start_dt).days < 365:
            start_date = (end_dt - dt.timedelta(days=400)).strftime("%Y-%m-%d")
    except ValueError:
        pass
    return pq_path, brk_df, start_date, end_date


def _weight_frame_for_first_window(
    runner: StrategyRunner,
    strategy,
    btc_df: pl.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Return a representative first-window weight frame for sanity checks."""
    start_ts = dt.datetime.strptime(start_date[:10], "%Y-%m-%d")
    end_ts = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
    window_end = start_ts + ALLOCATION_WINDOW_OFFSET
    if window_end > end_ts:
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})

    _, created_cache = runner._ensure_runtime_cache(strategy, btc_df)
    try:
        features_df = runner._materialize_strategy_features(
            strategy,
            btc_df,
            start_date=start_ts,
            end_date=window_end,
            current_date=window_end,
            cache_namespace="audit-window",
        )
        weights, _ = runner._compute_strategy_weights(
            strategy,
            features_df=features_df,
            start_date=start_ts,
            end_date=window_end,
            current_date=window_end,
            strict_mode=False,
            cache_namespace="audit-window",
        )
        return weights.sort("date")
    finally:
        runner._release_runtime_cache(clear=created_cache)


def _multiple_str(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def _build_judgment(
    name: str,
    *,
    result: dict[str, object],
    uniform_result: dict[str, object],
    uniform_weights: pl.DataFrame,
) -> str:
    """Return a short reasonableness judgment for a strategy audit row."""
    if not result["validation_passed"] or not result["backtest_passed"]:
        return "reject: failed validate/backtest"
    if not result["metrics_finite"]:
        return "reject: non-finite metrics"
    if result["windows"] <= 0:
        return "reject: empty SPD output"

    multiple = result["exp_decay_multiple_vs_uniform"]
    weight_frame = result["representative_weights"]
    weight_matches_uniform = weights_match(weight_frame, uniform_weights, atol=1e-12)

    if name == "UniformStrategy":
        if multiple is None or abs(multiple - 1.0) > 1e-9:
            return "reject: baseline multiple drifted from 1.0"
        if not weight_matches_uniform:
            return "reject: baseline weights differ from uniform reference"
        return "baseline OK"

    if name in {"SimpleZScoreStrategy", "MomentumStrategy"}:
        if weight_matches_uniform:
            return "acceptable toy model: finite but window weights collapse to baseline"
        if multiple is None:
            return "acceptable toy model: finite results, undefined multiple"
        if 0.75 <= multiple <= 1.25:
            return "acceptable toy model: finite, distinct, near-baseline"
        return "acceptable toy model: finite, distinct, outside baseline band"

    if weight_matches_uniform:
        return "reject: profile strategy collapsed to uniform weights"
    if multiple is None:
        return "investigate: finite results but undefined baseline multiple"
    uniform_score = float(uniform_result["score"])
    score = float(result["score"])
    if abs(score - uniform_score) < 0.05 and abs(multiple - 1.0) < 0.01:
        return "investigate: profile path behaves too close to uniform"
    return "distinct profile behavior looks plausible"


def _row_payload(
    name: str,
    validation,
    backtest,
    *,
    validate_seconds: float,
    backtest_seconds: float,
    representative_weights: pl.DataFrame,
) -> dict[str, object]:
    """Normalize a strategy run into a serializable audit row."""
    diagnostics = dict(validation.diagnostics or {})
    hot_path_profile = dict(diagnostics.get("hot_path_profile") or {})
    metrics = [
        float(backtest.win_rate),
        float(backtest.score),
        float(backtest.exp_decay_percentile),
        float(backtest.uniform_exp_decay_percentile),
    ]
    return {
        "strategy": name,
        "validation_passed": bool(validation.passed),
        "backtest_passed": not backtest.spd_table.is_empty(),
        "win_rate": float(backtest.win_rate),
        "score": float(backtest.score),
        "exp_decay_percentile": float(backtest.exp_decay_percentile),
        "uniform_exp_decay_percentile": float(backtest.uniform_exp_decay_percentile),
        "exp_decay_multiple_vs_uniform": backtest.exp_decay_multiple_vs_uniform,
        "windows": int(backtest.spd_table.height),
        "validate_seconds": validate_seconds,
        "backtest_seconds": backtest_seconds,
        "metrics_finite": all(map(lambda value: value == value and abs(value) != float("inf"), metrics)),
        "hot_path_profile": hot_path_profile,
        "representative_weights": representative_weights,
    }


def _failure_row(
    name: str,
    *,
    validate_seconds: float = 0.0,
    backtest_seconds: float = 0.0,
    validation_passed: bool = False,
    backtest_passed: bool = False,
    error: str | None = None,
    hot_path_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a normalized audit row for failed strategy runs."""
    return {
        "strategy": name,
        "validation_passed": validation_passed,
        "backtest_passed": backtest_passed,
        "win_rate": 0.0,
        "score": 0.0,
        "exp_decay_percentile": 0.0,
        "uniform_exp_decay_percentile": 0.0,
        "exp_decay_multiple_vs_uniform": None,
        "windows": 0,
        "validate_seconds": validate_seconds,
        "backtest_seconds": backtest_seconds,
        "metrics_finite": False,
        "hot_path_profile": dict(hot_path_profile or {}),
        "representative_weights": pl.DataFrame(
            schema={"date": pl.Datetime("us"), "weight": pl.Float64}
        ),
        "error": error,
    }


def main() -> int:
    root_dir = ROOT
    pq_path, btc_df, start_date, end_date = _load_audit_dataset(root_dir)
    os.environ["STACKSATS_ANALYTICS_PARQUET"] = str(pq_path)
    print(f"Using parquet: {pq_path}")
    print(f"Audit range: {start_date} -> {end_date}")

    runner = StrategyRunner()
    rows: list[dict[str, object]] = []
    failed: list[tuple[str, str]] = []

    for spec in STRATEGY_SPECS:
        name = spec.split(":")[-1]
        print(f"\n--- {name} ---")
        try:
            strategy = load_strategy(spec)
            validate_started = time.perf_counter()
            validation = runner.validate(
                strategy,
                ValidationConfig(
                    start_date=start_date,
                    end_date=end_date,
                    min_win_rate=0.0,
                    strict=False,
                ),
                btc_df=btc_df,
            )
            validate_seconds = time.perf_counter() - validate_started
            print(validation.summary())
            if not validation.passed:
                failed.append((name, "validate"))
                rows.append(
                    _failure_row(
                        name,
                        validate_seconds=validate_seconds,
                        validation_passed=False,
                        backtest_passed=False,
                        error="validation_failed",
                        hot_path_profile=(validation.diagnostics or {}).get(
                            "hot_path_profile"
                        ),
                    )
                )
                continue

            backtest_started = time.perf_counter()
            backtest = runner.backtest(
                strategy,
                BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=str(root_dir / "output"),
                    strategy_label=strategy.strategy_id,
                ),
                btc_df=btc_df,
            )
            backtest_seconds = time.perf_counter() - backtest_started
            print(backtest.summary())
            if backtest.spd_table.is_empty():
                failed.append((name, "backtest"))
                rows.append(
                    _failure_row(
                        name,
                        validate_seconds=validate_seconds,
                        backtest_seconds=backtest_seconds,
                        validation_passed=True,
                        backtest_passed=False,
                        error="backtest_empty",
                        hot_path_profile=(validation.diagnostics or {}).get(
                            "hot_path_profile"
                        ),
                    )
                )
                continue

            representative_weights = _weight_frame_for_first_window(
                runner,
                strategy,
                btc_df,
                start_date=start_date,
                end_date=end_date,
            )
            rows.append(
                _row_payload(
                    name,
                    validation,
                    backtest,
                    validate_seconds=validate_seconds,
                    backtest_seconds=backtest_seconds,
                    representative_weights=representative_weights,
                )
            )
            print("  OK: validate + backtest")
        except Exception as exc:
            failed.append((name, "exception"))
            rows.append(
                _failure_row(
                    name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    if not rows:
        print("\nNo strategy results collected.")
        return 1

    uniform_row = next((row for row in rows if row["strategy"] == "UniformStrategy"), None)
    if uniform_row is None:
        raise SystemExit("UniformStrategy audit row missing; cannot build comparisons.")
    uniform_weights = uniform_row["representative_weights"]
    printable_rows: list[dict[str, object]] = []
    for row in rows:
        judgment = _build_judgment(
            str(row["strategy"]),
            result=row,
            uniform_result=uniform_row,
            uniform_weights=uniform_weights,
        )
        printable_rows.append(
            {
                "strategy": row["strategy"],
                "validation_passed": row["validation_passed"],
                "backtest_passed": row["backtest_passed"],
                "win_rate": round(float(row["win_rate"]), 2),
                "score": round(float(row["score"]), 2),
                "exp_decay_percentile": round(float(row["exp_decay_percentile"]), 2),
                "uniform_exp_decay_percentile": round(
                    float(row["uniform_exp_decay_percentile"]),
                    2,
                ),
                "exp_decay_multiple_vs_uniform": (
                    None
                    if row["exp_decay_multiple_vs_uniform"] is None
                    else round(float(row["exp_decay_multiple_vs_uniform"]), 3)
                ),
                "windows": row["windows"],
                "validate_seconds": round(float(row["validate_seconds"]), 3),
                "backtest_seconds": round(float(row["backtest_seconds"]), 3),
                "judgment": judgment,
                "hot_path_profile": row["hot_path_profile"],
                "error": row.get("error"),
            }
        )

    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "strategy_audit.json"
    output_path.write_text(json.dumps(printable_rows, indent=2), encoding="utf-8")

    print("\nStrategy Audit")
    header = (
        "strategy | valid | backtest | win_rate | score | exp_pct | uniform_pct | "
        "multiple | windows | validate_s | backtest_s | judgment"
    )
    print(header)
    print("-" * len(header))
    for row in printable_rows:
        print(
            f"{row['strategy']} | "
            f"{row['validation_passed']} | "
            f"{row['backtest_passed']} | "
            f"{row['win_rate']:.2f} | "
            f"{row['score']:.2f} | "
            f"{row['exp_decay_percentile']:.2f} | "
            f"{row['uniform_exp_decay_percentile']:.2f} | "
            f"{_multiple_str(row['exp_decay_multiple_vs_uniform'])} | "
            f"{row['windows']} | "
            f"{row['validate_seconds']:.3f} | "
            f"{row['backtest_seconds']:.3f} | "
            f"{row['judgment']}"
        )

    print(f"\nWrote JSON audit summary to {output_path}")
    if failed:
        print(f"\nFailed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
