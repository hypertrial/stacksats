#!/usr/bin/env python3
"""Run validate + backtest for all built-in strategies with merged_metrics*.parquet."""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

import polars as pl

from stacksats.loader import load_strategy
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, ValidationConfig

STRATEGY_SPECS = [
    "stacksats.strategies.examples:UniformStrategy",
    "stacksats.strategies.examples:SimpleZScoreStrategy",
    "stacksats.strategies.examples:MomentumStrategy",
    "stacksats.strategies.mvrv:MVRVStrategy",
    "stacksats.strategies.model_example:ExampleMVRVStrategy",
    "stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy",
]


def _resolve_parquet_path(root_dir: Path) -> Path:
    """Use STACKSATS_ANALYTICS_PARQUET or merged_metrics*.parquet in repo root."""
    env_path = os.environ.get("STACKSATS_ANALYTICS_PARQUET")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p.resolve()
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
    df = pl.read_parquet(pq_path)
    if set(df.columns) != {"day_utc", "metric", "value"}:
        raise SystemExit(
            f"merged_metrics parquet must have columns (day_utc, metric, value), got {df.columns}"
        )
    needed = ["market_cap", "supply_btc", "mvrv"]
    sub = df.filter(pl.col("metric").is_in(needed))
    wide = sub.pivot(values="value", index="day_utc", on="metric")
    wide = wide.with_columns(
        (pl.col("market_cap") / pl.col("supply_btc")).alias("price_usd")
    )
    wide = wide.rename({"day_utc": "date"}).select(["date", "price_usd", "mvrv"])
    wide = wide.filter(pl.col("price_usd").is_finite() & (pl.col("price_usd") > 0))
    wide.write_parquet(out_path)
    return out_path


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    raw_pq = _resolve_parquet_path(root_dir)
    with tempfile.TemporaryDirectory(prefix="stacksats-run-all-") as tmp:
        tmp_path = Path(tmp)
        # Convert merged_metrics (long) to BRK-wide if needed
        df_sample = pl.read_parquet(raw_pq, n_rows=1)
        if set(df_sample.columns) == {"day_utc", "metric", "value"}:
            brk_pq = tmp_path / "bitcoin_analytics.parquet"
            _convert_merged_metrics_to_brk(raw_pq, brk_pq)
            pq_path = brk_pq
            print(f"Converted merged_metrics -> {pq_path}")
        else:
            pq_path = raw_pq
            print(f"Using parquet: {pq_path}")
        os.environ["STACKSATS_ANALYTICS_PARQUET"] = str(pq_path)
        # Use data's last date to avoid "does not cover requested end_date"
        brk_df = pl.read_parquet(pq_path)
        last_date = brk_df["date"].max()
        end_date = (
            last_date.isoformat()
            if hasattr(last_date, "isoformat")
            else str(last_date)[:10]
        )
        # Need >= 365 days for backtest windows (ALLOCATION_SPAN_DAYS)
        start_date = "2018-01-01"
        try:
            end_dt = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
            start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
            if start_dt > end_dt or (end_dt - start_dt).days < 365:
                start_date = (end_dt - dt.timedelta(days=400)).strftime("%Y-%m-%d")
        except Exception:
            pass
        runner = StrategyRunner()
        failed = []
        for spec in STRATEGY_SPECS:
            name = spec.split(":")[-1]
            print(f"\n--- {name} ---")
            strategy = load_strategy(spec)
            validation = runner.validate(
                strategy,
                ValidationConfig(
                    start_date=start_date,
                    end_date=end_date,
                    min_win_rate=0.0,
                    strict=False,
                ),
            )
            print(validation.summary())
            if not validation.passed:
                failed.append((name, "validate"))
                continue
            backtest = runner.backtest(
                strategy,
                BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=str(root_dir / "output"),
                    strategy_label=strategy.strategy_id,
                ),
            )
            print(backtest.summary())
            if backtest.spd_table.empty:
                failed.append((name, "backtest"))
                continue
            print("  OK: validate + backtest")

        if failed:
            print("\nFailed:", failed)
            return 1
        print("\nAll strategies passed validate + backtest.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
