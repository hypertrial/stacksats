"""Baseline regression tests for the Polars strategy runtime."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stacksats.api import BacktestResult
from stacksats.export_weights import process_start_date_batch
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import compute_window_weights, precompute_features
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import (
    BacktestConfig,
    BaseStrategy,
    ExportConfig,
    StrategyContext,
    TargetProfile,
)
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df,
    normalize_weight_frame,
)

pytestmark = pytest.mark.integration

PRICE_COL = "price_usd"


class _UniformBaseStrategy(BaseStrategy):
    strategy_id = "uniform-regression"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.is_empty():
            return TargetProfile(
                values=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
                mode="absolute",
            )
        return TargetProfile(
            values=features_df.select("date").with_columns(pl.lit(1.0).alias("value")),
            mode="absolute",
        )


def test_empty_backtest_range_returns_clear_error() -> None:
    runner = StrategyRunner()
    strategy = _UniformBaseStrategy()

    with pytest.raises(Exception):
        runner.backtest(
            strategy,
            BacktestConfig(start_date="2050-01-01", end_date="2050-12-31"),
            btc_df=make_btc_df(),
        )


def test_export_backtest_weight_alignment_regression() -> None:
    btc_df = make_btc_df(start="2020-01-01", days=2500)
    features_df = precompute_features(btc_df)

    start_date = dt_at("2024-01-01")
    end_date = start_date + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)
    current_date = dt_at("2024-07-01")

    backtest_weights = normalize_weight_frame(
        compute_window_weights(
            features_df=features_df,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        )
    )
    export_weights = normalize_weight_frame(
        process_start_date_batch(
            start_date,
            [end_date],
            features_df,
            btc_df,
            current_date,
            PRICE_COL,
        ).select(["date", "weight"])
    )

    np.testing.assert_allclose(
        backtest_weights["weight"].to_numpy(),
        export_weights["weight"].to_numpy(),
        atol=1e-12,
    )


def test_runner_export_backtest_parity_with_base_strategy(tmp_path: Path) -> None:
    btc_df = make_btc_df(start="2020-01-01", days=2500)
    runner = StrategyRunner()
    strategy = _UniformBaseStrategy()

    backtest_result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2023-01-01", end_date="2024-12-31"),
        btc_df=btc_df,
    )
    exported_batch = runner.export(
        strategy,
        ExportConfig(
            range_start="2023-01-01",
            range_end="2024-12-31",
            output_dir=str(tmp_path),
        ),
        btc_df=btc_df,
        current_date=dt_at("2024-07-01"),
    )

    assert not backtest_result.spd_table.is_empty()
    assert not exported_batch.to_dataframe().is_empty()


def test_assert_bypass_not_allowed_under_python_optimize(tmp_path: Path) -> None:
    script = tmp_path / "optimize_guard_check.py"
    payload = tmp_path / "result.json"
    script.write_text(
        """
import json
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, BaseStrategy

class BadStrategy(BaseStrategy):
    strategy_id = "bad-opt"
    version = "1.0.0"

    def build_target_profile(self, ctx, features_df, signals):
        del ctx, signals
        return pl.DataFrame({"date": features_df["date"], "bad": np.nan})

start = datetime(2022, 1, 1)
dates = [start + timedelta(days=i) for i in range(500)]
btc_df = pl.DataFrame(
    {
        "date": dates,
        "price_usd": np.linspace(20000, 40000, 500),
        "PriceUSD": np.linspace(20000, 40000, 500),
        "mvrv": np.linspace(1.0, 2.0, 500),
    }
)

ok = False
try:
    StrategyRunner().backtest(
        BadStrategy(),
        BacktestConfig(start_date="2022-01-01", end_date="2023-01-01"),
        btc_df=btc_df,
    )
except Exception:
    ok = True

with open(r\"\"\"%s\"\"\", "w", encoding="utf-8") as f:
    json.dump({"raised": ok}, f)
"""
        % str(payload),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-O", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        },
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(payload.read_text(encoding="utf-8"))
    assert result["raised"] is True


def test_backtest_json_schema_snapshot_contract() -> None:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "snapshots"
        / "backtest_result_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    spd = pl.DataFrame({
        "window": ["2024-01-01 → 2025-01-01"],
        "uniform_percentile": [50.0],
        "dynamic_percentile": [55.0],
        "dynamic_sats_per_dollar": [5000.0],
        "uniform_sats_per_dollar": [4800.0],
        "min_sats_per_dollar": [4000.0],
        "max_sats_per_dollar": [6000.0],
        "excess_percentile": [5.0],
    })
    result = BacktestResult(
        spd_table=spd,
        exp_decay_percentile=55.0,
        win_rate=100.0,
        score=77.5,
    )
    payload = result.to_json()

    for required in schema["required"]:
        assert required in payload
    for required in schema["properties"]["provenance"]["required"]:
        assert required in payload["provenance"]
    for required in schema["properties"]["summary_metrics"]["required"]:
        assert required in payload["summary_metrics"]
