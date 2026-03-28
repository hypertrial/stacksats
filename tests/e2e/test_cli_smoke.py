from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from json import JSONDecodeError
from pathlib import Path

import polars as pl

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
    decoder = json.JSONDecoder()
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("{"):
            continue
        try:
            payload, _ = decoder.raw_decode("\n".join(lines[index:]).lstrip())
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"No JSON object found in stdout: {stdout!r}")


def _copy_demo_parquet(tmp_path: Path) -> Path:
    with packaged_demo_parquet_path() as demo_parquet:
        source = Path(demo_parquet)
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        return target


def _demo_env(tmp_path: Path) -> dict[str, str]:
    env = _base_env(tmp_path)
    env["STACKSATS_ANALYTICS_PARQUET"] = str(_copy_demo_parquet(tmp_path))
    return env


def _demo_backtest(
    tmp_path: Path,
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output_dir = tmp_path / "demo"
    result = _run_cli(
        "demo",
        "backtest",
        "--output-dir",
        str(output_dir),
        env=env,
    )
    return result, output_dir


def _require_single_artifact(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    assert matches, f"Expected artifact matching {pattern!r} under {root}, found none."
    assert len(matches) == 1, f"Expected one artifact matching {pattern!r} under {root}, found {matches!r}."
    return matches[0]


def _write_missing_price_runtime_parquet(tmp_path: Path, *, run_date: str) -> Path:
    with packaged_demo_parquet_path() as demo_parquet:
        df = pl.read_parquet(demo_parquet)
    df = df.with_columns(
        pl.when(pl.col("date").dt.strftime("%Y-%m-%d") == run_date)
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(pl.col("price_usd"))
        .alias("price_usd")
    )
    output = tmp_path / "missing-price.parquet"
    df.write_parquet(output)
    return output


def _write_gapped_runtime_parquet(tmp_path: Path) -> Path:
    output = tmp_path / "gapped.parquet"
    pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)],
            "price_usd": [100.0, 101.0],
            "mvrv": [0.5, 0.6],
        }
    ).write_parquet(output)
    return output


def test_demo_backtest_smoke(tmp_path: Path) -> None:
    env = _base_env(tmp_path)

    demo_backtest, output_dir = _demo_backtest(tmp_path, env)

    assert demo_backtest.returncode == 0, demo_backtest.stderr
    assert "Score:" in demo_backtest.stdout
    assert "Saved:" in demo_backtest.stdout
    backtest_json = _require_single_artifact(output_dir, "**/backtest_result.json")
    payload = json.loads(backtest_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert payload["provenance"]["strategy_id"] == "simple-zscore"
    assert payload["summary_metrics"]["windows"] > 0
    assert str(backtest_json.parent).startswith(str(tmp_path / "demo"))


def test_strategy_export_round_trip_smoke(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)
    output_dir = tmp_path / "export"

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
        str(output_dir),
        env=env,
    )

    assert export_run.returncode == 0, export_run.stderr
    payload = _decode_leading_json(export_run.stdout)
    assert payload["strategy_id"] == "simple-zscore"
    assert payload["rows"] > 0
    assert payload["windows"] > 0
    artifact_dir = Path(str(payload["output_dir"]))
    assert artifact_dir.exists()
    assert str(artifact_dir).startswith(str(output_dir))

    batch = WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)
    assert batch.strategy_id == "simple-zscore"
    assert batch.row_count > 0
    assert batch.window_count > 0


def test_strategy_animate_smoke(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    demo_backtest, output_dir = _demo_backtest(tmp_path, env)
    assert demo_backtest.returncode == 0, demo_backtest.stderr
    backtest_json = _require_single_artifact(output_dir, "**/backtest_result.json")

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
    payload = _decode_leading_json(animate_run.stdout)
    assert payload["gif"].endswith("smoke.gif")
    assert payload["manifest_json"].endswith("animation_manifest.json")
    assert Path(payload["gif"]).exists()
    assert Path(payload["manifest_json"]).exists()
    assert str(Path(payload["gif"]).parent) == str(backtest_json.parent.resolve())


def test_data_prepare_and_doctor_smoke(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    runtime_parquet = tmp_path / "runtime.parquet"

    prepare_run = _run_cli(
        "data",
        "prepare",
        "--source",
        str(_copy_demo_parquet(tmp_path)),
        "--output",
        str(runtime_parquet),
        "--overwrite",
        env=env,
    )

    assert prepare_run.returncode == 0, prepare_run.stderr
    prepare_payload = json.loads(prepare_run.stdout)
    assert prepare_payload["runtime_parquet"] == str(runtime_parquet.resolve())
    assert runtime_parquet.exists()

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


def test_run_daily_and_reconcile_smoke(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)
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
    first_payload = _decode_leading_json(first_run.stdout)
    assert first_payload["status"] == "executed"
    assert first_payload["idempotency_hit"] is False
    assert first_payload["validation_passed"] is True
    assert first_payload["artifact_path"] is not None
    assert Path(str(first_payload["artifact_path"])).exists()
    assert str(first_payload["artifact_path"]).startswith(str(output_dir.resolve()))
    assert "Status: EXECUTED" in first_run.stdout

    second_run = _run_cli(*run_daily_args, env=env)
    assert second_run.returncode == 0, second_run.stderr
    second_payload = _decode_leading_json(second_run.stdout)
    assert second_payload["status"] == "noop"
    assert second_payload["idempotency_hit"] is True
    assert second_payload["run_key"] == first_payload["run_key"]
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
    assert reconcile_payload["status"] in {
        "stable",
        "data_revised_no_decision_change",
        "decision_changed_due_to_revision",
    }
    assert "previous_feature_snapshot_hash" in reconcile_payload
    assert "recomputed_feature_snapshot_hash" in reconcile_payload


def test_run_daily_missing_price_coverage_fails(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env["STACKSATS_ANALYTICS_PARQUET"] = str(
        _write_missing_price_runtime_parquet(tmp_path, run_date="2024-12-31")
    )

    run = _run_cli(
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
        str(tmp_path / "state.sqlite3"),
        "--output-dir",
        str(tmp_path / "daily"),
        env=env,
    )

    assert run.returncode == 1
    payload = _decode_leading_json(run.stdout)
    assert payload["status"] == "failed"
    assert "missing price_usd values" in payload["message"]
    assert "Status: FAILED" in run.stdout


def test_run_daily_live_mode_without_adapter_fails(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)

    run = _run_cli(
        "strategy",
        "run-daily",
        "--strategy",
        "stacksats.strategies.examples:RunDailyPaperStrategy",
        "--run-date",
        "2024-12-31",
        "--total-window-budget-usd",
        "1000",
        "--mode",
        "live",
        env=env,
    )

    assert run.returncode == 2
    assert "Error: Live mode requires --adapter." in run.stderr


def test_strategy_animate_with_malformed_backtest_json_fails(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    bad_json = tmp_path / "backtest_result.json"
    bad_json.write_text("{}", encoding="utf-8")

    run = _run_cli(
        "strategy",
        "animate",
        "--backtest-json",
        str(bad_json),
        env=env,
    )

    assert run.returncode == 2
    assert "Error: Backtest JSON is missing 'window_level_data' list." in run.stderr


def test_data_doctor_reports_warning_for_daily_gaps(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    gapped_parquet = _write_gapped_runtime_parquet(tmp_path)

    run = _run_cli(
        "data",
        "doctor",
        "--parquet-path",
        str(gapped_parquet),
        env=env,
    )

    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["status"] == "warning"
    assert payload["has_daily_gaps"] is True
    assert payload["gap_count"] == 1
