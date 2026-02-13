"""Baseline regression tests for strategy runtime behavior.

These tests pin down known fragile areas so refactors can be validated against
a stable contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stacksats.api import BacktestResult
from stacksats.export_weights import process_start_date_batch
from stacksats.model_development import compute_window_weights, precompute_features
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import (
    BacktestConfig,
    BaseStrategy,
    ExportConfig,
    StrategyContext,
    TargetProfile,
)

PRICE_COL = "PriceUSD_coinmetrics"


def _sample_btc_df(start: str = "2021-01-01", days: int = 900) -> pd.DataFrame:
    idx = pd.date_range(start, periods=days, freq="D")
    trend = np.linspace(10000, 60000, num=days)
    return pd.DataFrame(
        {
            PRICE_COL: trend,
            "CapMVRVCur": np.linspace(0.8, 2.2, num=days),
        },
        index=idx,
    )


class _UniformBaseStrategy(BaseStrategy):
    strategy_id = "uniform-regression"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="absolute")
        return TargetProfile(
            values=pd.Series(np.ones(len(features_df.index)), index=features_df.index),
            mode="absolute",
        )


def test_empty_backtest_range_returns_clear_error() -> None:
    btc_df = _sample_btc_df()
    strategy = _UniformBaseStrategy()
    runner = StrategyRunner()

    with pytest.raises(Exception):
        runner.backtest(
            strategy,
            BacktestConfig(
                start_date="2050-01-01",
                end_date="2050-12-31",
            ),
            btc_df=btc_df,
        )


def test_export_backtest_weight_alignment_regression() -> None:
    btc_df = _sample_btc_df(start="2020-01-01", days=2500)
    features_df = precompute_features(btc_df)

    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    current_date = pd.Timestamp("2024-07-01")

    backtest_weights = compute_window_weights(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        current_date=current_date,
    )
    export_df = process_start_date_batch(
        start_date,
        [end_date],
        features_df,
        btc_df,
        current_date,
        PRICE_COL,
    )
    export_weights = export_df.set_index("DCA_date")["weight"]
    export_weights.index = pd.to_datetime(export_weights.index)

    np.testing.assert_allclose(backtest_weights.values, export_weights.values, atol=1e-12)


def test_runner_export_backtest_parity_with_base_strategy(tmp_path: Path) -> None:
    btc_df = _sample_btc_df(start="2020-01-01", days=2500)
    runner = StrategyRunner()
    strategy = _UniformBaseStrategy()
    backtest_result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2023-01-01", end_date="2024-12-31"),
        btc_df=btc_df,
    )
    exported = runner.export(
        strategy,
        ExportConfig(
            range_start="2023-01-01",
            range_end="2024-12-31",
            output_dir=str(tmp_path),
        ),
        btc_df=btc_df,
        current_date=pd.Timestamp("2024-07-01"),
    )
    assert not backtest_result.spd_table.empty
    assert not exported.empty


def test_assert_bypass_not_allowed_under_python_optimize(tmp_path: Path) -> None:
    script = tmp_path / "optimize_guard_check.py"
    payload = tmp_path / "result.json"
    script.write_text(
        """
import json
import pandas as pd
import numpy as np
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, BaseStrategy, StrategyContext

class BadStrategy(BaseStrategy):
    strategy_id = "bad-opt"
    version = "1.0.0"

    def build_target_profile(self, ctx, features_df, signals):
        del ctx, signals
        return pd.Series(np.nan, index=features_df.index)

idx = pd.date_range("2022-01-01", periods=500, freq="D")
btc_df = pd.DataFrame(
    {
        "PriceUSD_coinmetrics": np.linspace(20000, 40000, 500),
        "CapMVRVCur": np.linspace(1.0, 2.0, 500),
    },
    index=idx,
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

    spd = pd.DataFrame(
        {
            "window": ["2024-01-01 → 2025-01-01"],
            "uniform_percentile": [50.0],
            "dynamic_percentile": [55.0],
            "dynamic_sats_per_dollar": [5000.0],
            "uniform_sats_per_dollar": [4800.0],
            "min_sats_per_dollar": [4000.0],
            "max_sats_per_dollar": [6000.0],
            "excess_percentile": [5.0],
        }
    ).set_index("window")
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
