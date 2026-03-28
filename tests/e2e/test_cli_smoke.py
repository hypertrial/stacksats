from __future__ import annotations

import json
import os
import subprocess
import sys
from json import JSONDecoder
from pathlib import Path

from stacksats.data_setup import packaged_demo_parquet_path
from stacksats.strategy_time_series_batch import WeightTimeSeriesBatch


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_MODULE = [sys.executable, "-m", "stacksats.cli"]


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    return env


def _run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI_MODULE, *args],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _decode_leading_json(stdout: str) -> dict[str, object]:
    payload, _ = JSONDecoder().raw_decode(stdout)
    return payload


def test_cli_smoke_happy_paths(tmp_path: Path) -> None:
    env = _base_env(tmp_path)

    demo_backtest = _run_cli(
        "demo",
        "backtest",
        "--output-dir",
        str(tmp_path / "demo"),
        env=env,
    )
    assert demo_backtest.returncode == 0, demo_backtest.stderr
    assert "Score:" in demo_backtest.stdout
    assert "Saved:" in demo_backtest.stdout
    backtest_json = next((tmp_path / "demo").glob("**/backtest_result.json"))
    assert backtest_json.exists()

    with packaged_demo_parquet_path() as demo_parquet:
        env["STACKSATS_ANALYTICS_PARQUET"] = str(demo_parquet)

        export_run = _run_cli(
            "strategy",
            "export",
            "--strategy",
            "stacksats.strategies.examples:SimpleZScoreStrategy",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-12-31",
            "--output-dir",
            str(tmp_path / "export"),
            env=env,
        )
        assert export_run.returncode == 0, export_run.stderr
        export_artifact_dir = next((tmp_path / "export").glob("*/*/*"))
        batch = WeightTimeSeriesBatch.from_artifact_dir(export_artifact_dir)
        assert batch.strategy_id == "simple-zscore"
        assert batch.row_count > 0
        assert batch.window_count > 0

        animate_run = _run_cli(
            "strategy",
            "animate",
            "--backtest-json",
            str(backtest_json),
            "--output-dir",
            str(backtest_json.parent),
            "--output-name",
            "smoke.gif",
            "--fps",
            "5",
            "--width",
            "640",
            "--height",
            "360",
            "--max-frames",
            "10",
            env=env,
        )
        assert animate_run.returncode == 0, animate_run.stderr
        assert (backtest_json.parent / "smoke.gif").exists()
        assert (backtest_json.parent / "animation_manifest.json").exists()

        runtime_parquet = tmp_path / "runtime.parquet"
        prepare_run = _run_cli(
            "data",
            "prepare",
            "--source",
            str(demo_parquet),
            "--output",
            str(runtime_parquet),
            "--overwrite",
            env=env,
        )
        assert prepare_run.returncode == 0, prepare_run.stderr

        doctor_run = _run_cli(
            "data",
            "doctor",
            "--parquet-path",
            str(runtime_parquet),
            env=env,
        )
        assert doctor_run.returncode == 0, doctor_run.stderr
        doctor_payload = json.loads(doctor_run.stdout)
        assert doctor_payload["status"] == "ok"
        assert doctor_payload["has_price_usd"] is True
        assert doctor_payload["has_mvrv"] is True
        assert doctor_payload["has_daily_gaps"] is False

        state_db = tmp_path / "state.sqlite3"
        output_dir = tmp_path / "daily"
        run_daily_args = [
            "strategy",
            "run-daily",
            "--strategy",
            "stacksats.strategies.examples:RunDailyPaperStrategy",
            "--run-date",
            "2024-12-31",
            "--total-window-budget-usd",
            "1000",
            "--mode",
            "paper",
            "--state-db-path",
            str(state_db),
            "--output-dir",
            str(output_dir),
        ]
        first_run = _run_cli(*run_daily_args, env=env)
        assert first_run.returncode == 0, first_run.stderr
        assert "Status: EXECUTED" in first_run.stdout

        second_run = _run_cli(*run_daily_args, env=env)
        assert second_run.returncode == 0, second_run.stderr
        assert "Status: NO-OP (idempotent)" in second_run.stdout

        reconcile_run = _run_cli(
            "strategy",
            "reconcile-daily",
            "--strategy",
            "stacksats.strategies.examples:RunDailyPaperStrategy",
            "--run-date",
            "2024-12-31",
            "--mode",
            "paper",
            "--state-db-path",
            str(state_db),
            env=env,
        )
        assert reconcile_run.returncode == 0, reconcile_run.stderr
        reconcile_payload = _decode_leading_json(reconcile_run.stdout)
        assert "status" in reconcile_payload
