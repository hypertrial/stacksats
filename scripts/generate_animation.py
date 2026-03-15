#!/usr/bin/env python3
"""Generate the strategy-vs-uniform animation GIF from a backtest.

Uses merged_metrics*.parquet in repo root (or STACKSATS_ANALYTICS_PARQUET).

Run from repo root:
  python scripts/generate_animation.py

Output: output/animation/strategy_vs_uniform_hd.gif
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import polars as pl

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def _load_btc_from_merged_metrics(root_dir: Path) -> pl.DataFrame:
    """Load BTC data from merged_metrics parquet (long or wide format)."""
    raw_pq = _resolve_parquet_path(root_dir)
    df_sample = pl.read_parquet(raw_pq, n_rows=1)
    if set(df_sample.columns) == {"day_utc", "metric", "value"}:
        df = pl.read_parquet(raw_pq)
        needed = ["market_cap", "supply_btc", "mvrv"]
        sub = df.filter(pl.col("metric").is_in(needed))
        wide = sub.pivot(values="value", index="day_utc", on="metric")
        wide = wide.with_columns(
            (pl.col("market_cap") / pl.col("supply_btc")).alias("price_usd")
        )
        wide = wide.rename({"day_utc": "date"}).select(["date", "price_usd", "mvrv"])
    else:
        wide = pl.read_parquet(raw_pq)
        if "day_utc" in wide.columns and "date" not in wide.columns:
            wide = wide.rename({"day_utc": "date"})
    wide = wide.filter(pl.col("price_usd").is_finite() & (pl.col("price_usd") > 0))
    # Ensure date is Datetime (not Date) for feature materialization
    if wide["date"].dtype == pl.Date:
        wide = wide.with_columns(
            pl.col("date").cast(pl.Datetime("us")).dt.truncate("1d")
        )
    return wide


def main() -> int:
    from stacksats.loader import load_strategy
    from stacksats.runner import StrategyRunner
    from stacksats.strategy_types import BacktestConfig

    output_dir = ROOT / "output" / "animation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data from merged_metrics parquet...")
    btc_df = _load_btc_from_merged_metrics(ROOT)
    last_date = btc_df["date"].max()
    end_date = (
        last_date.isoformat()[:10]
        if hasattr(last_date, "isoformat")
        else str(last_date)[:10]
    )
    start_date = "2024-01-01"
    try:
        end_dt = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
        if start_dt > end_dt or (end_dt - start_dt).days < 365:
            start_date = (end_dt - dt.timedelta(days=400)).strftime("%Y-%m-%d")
    except Exception:
        pass
    print(f"Date range: {start_date} to {end_date} ({btc_df.height} rows)")

    strategy = load_strategy("stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy")
    runner = StrategyRunner()

    print("Running backtest...")
    result = runner.backtest(
        strategy,
        BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            output_dir=str(ROOT / "output"),
            strategy_label=strategy.strategy_id,
        ),
        btc_df=btc_df,
    )
    print(result.summary())

    if result.spd_table.is_empty():
        print("ERROR: No backtest windows generated.")
        return 1

    backtest_json = output_dir / "backtest_result.json"
    result.to_json(backtest_json)
    print(f"Saved backtest: {backtest_json}")

    print("Generating animation GIF...")
    paths = result.animate(
        output_dir=str(output_dir),
        fps=12,
        width=1280,
        height=720,
        max_frames=60,
        filename="strategy_vs_uniform_hd.gif",
        window_mode="rolling",
    )
    gif_path = paths["gif"]
    print(f"Animation saved: {gif_path}")
    print(f"Manifest: {paths['manifest_json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
