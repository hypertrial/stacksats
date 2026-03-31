from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from json import JSONDecodeError
from pathlib import Path
from uuid import UUID

import polars as pl

from stacksats.data.data_setup import packaged_demo_parquet_path
from stacksats.strategy_time_series.batch import WeightTimeSeriesBatch


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


def _write_live_adapter_module(tmp_path: Path) -> Path:
    adapter_path = tmp_path / "temp_live_adapter.py"
    adapter_path.write_text(
        "\n".join(
            [
                "from stacksats.api import DailyOrderReceipt",
                "",
                "class TempAdapter:",
                "    def submit_order(self, request, *, idempotency_key):",
                "        return DailyOrderReceipt(",
                "            status='filled',",
                "            external_order_id='live-' + idempotency_key,",
                "            filled_notional_usd=float(request.notional_usd),",
                "            filled_quantity_btc=float(request.quantity_btc),",
                "            fill_price_usd=float(request.price_usd),",
                "            metadata={'adapter': 'temp-live', 'mode': request.mode},",
                "        )",
            ]
        ),
        encoding="utf-8",
    )
    return adapter_path


def _write_broken_live_adapter_module(tmp_path: Path) -> Path:
    adapter_path = tmp_path / "broken_live_adapter.py"
    adapter_path.write_text(
        "\n".join(
            [
                "class BrokenAdapter:",
                "    def submit_order(self, request, *, idempotency_key):",
                "        del request, idempotency_key",
                "        return object()",
            ]
        ),
        encoding="utf-8",
    )
    return adapter_path


def _assert_required_keys(payload: dict[str, object], keys: set[str]) -> None:
    assert keys.issubset(set(payload)), payload


def _assert_uuid_like(value: object) -> None:
    UUID(str(value))


def _assert_path_under(path_value: object, root: Path) -> Path:
    path = Path(str(path_value))
    assert path.exists()
    assert str(path).startswith(str(root.resolve()))
    return path


def _assert_run_daily_payload_contract(payload: dict[str, object], *, expected_mode: str) -> None:
    _assert_required_keys(
        payload,
        {
            "status",
            "strategy_id",
            "strategy_version",
            "run_date",
            "run_key",
            "mode",
            "idempotency_hit",
            "forced_rerun",
            "state_db_path",
            "artifact_path",
            "message",
        },
    )
    assert payload["strategy_id"] == "run-daily-paper"
    assert payload["strategy_version"] == "1.0.0"
    assert payload["run_date"] == "2024-12-31"
    assert payload["mode"] == expected_mode
    _assert_uuid_like(payload["run_key"])


def _assert_decide_daily_payload_contract(payload: dict[str, object]) -> None:
    _assert_required_keys(
        payload,
        {
            "status",
            "strategy_id",
            "strategy_version",
            "run_date",
            "decision_key",
            "btc_price_col",
            "state_db_path",
            "artifact_path",
            "message",
        },
    )
    assert payload["strategy_id"] == "run-daily-paper"
    assert payload["strategy_version"] == "1.0.0"
    assert payload["run_date"] == "2024-12-31"
    _assert_uuid_like(payload["decision_key"])


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
    _assert_required_keys(
        payload,
        {
            "rows",
            "windows",
            "strategy_id",
            "version",
            "schema_version",
            "output_dir",
        },
    )
    assert payload["strategy_id"] == "simple-zscore"
    assert payload["version"] == "1.0.0"
    assert payload["schema_version"] == "1.0.0"
    assert payload["rows"] > 0
    assert payload["windows"] > 0
    artifact_dir = _assert_path_under(payload["output_dir"], output_dir)

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
    _assert_required_keys(payload, {"gif", "manifest_json"})
    assert payload["gif"].endswith("smoke.gif")
    assert payload["manifest_json"].endswith("animation_manifest.json")
    assert Path(str(payload["gif"])).exists()
    assert Path(str(payload["manifest_json"])).exists()
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
    _assert_required_keys(prepare_payload, {"runtime_parquet"})
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
    _assert_required_keys(
        doctor_payload,
        {"status", "has_price_usd", "has_mvrv", "has_daily_gaps"},
    )
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
    _assert_run_daily_payload_contract(first_payload, expected_mode="paper")
    assert first_payload["status"] == "executed"
    assert first_payload["idempotency_hit"] is False
    assert first_payload["forced_rerun"] is False
    assert first_payload["validation_passed"] is True
    assert first_payload["artifact_path"] is not None
    artifact_path = _assert_path_under(first_payload["artifact_path"], output_dir)
    assert Path(str(first_payload["state_db_path"])) == state_db.resolve()
    assert artifact_path.name.endswith(".json")
    assert "Status: EXECUTED" in first_run.stdout

    second_run = _run_cli(*run_daily_args, env=env)
    assert second_run.returncode == 0, second_run.stderr
    second_payload = _decode_leading_json(second_run.stdout)
    _assert_run_daily_payload_contract(second_payload, expected_mode="paper")
    assert second_payload["status"] == "noop"
    assert second_payload["idempotency_hit"] is True
    assert second_payload["forced_rerun"] is False
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
    _assert_required_keys(
        reconcile_payload,
        {
            "status",
            "previous_weight_today",
            "recomputed_weight_today",
            "previous_feature_snapshot_hash",
            "recomputed_feature_snapshot_hash",
        },
    )
    assert reconcile_payload["status"] in {
        "stable",
        "data_revised_no_decision_change",
        "decision_changed_due_to_revision",
    }
    assert "previous_feature_snapshot_hash" in reconcile_payload
    assert "recomputed_feature_snapshot_hash" in reconcile_payload


def test_decide_daily_smoke(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)
    state_db = tmp_path / "decision-state.sqlite3"
    output_dir = tmp_path / "decisions"
    decide_args = [
        "strategy",
        "decide-daily",
        "--strategy",
        "stacksats.strategies.examples:RunDailyPaperStrategy",
        "--run-date",
        "2024-12-31",
        "--total-window-budget-usd",
        "1000",
        "--state-db-path",
        str(state_db),
        "--output-dir",
        str(output_dir),
    ]

    first_run = _run_cli(*decide_args, env=env)
    assert first_run.returncode == 0, first_run.stderr
    first_payload = _decode_leading_json(first_run.stdout)
    _assert_decide_daily_payload_contract(first_payload)
    assert first_payload["status"] == "decided"
    assert first_payload["validation_passed"] is True
    assert "order_receipt" not in first_payload
    assert "adapter_name" not in first_payload
    assert first_payload["artifact_path"] is not None
    artifact_path = _assert_path_under(first_payload["artifact_path"], output_dir)
    assert Path(str(first_payload["state_db_path"])) == state_db.resolve()
    assert artifact_path.name == "decision_result.json"
    assert "Status: DECIDED" in first_run.stdout

    second_run = _run_cli(*decide_args, env=env)
    assert second_run.returncode == 0, second_run.stderr
    second_payload = _decode_leading_json(second_run.stdout)
    _assert_decide_daily_payload_contract(second_payload)
    assert second_payload["status"] == "noop"
    assert second_payload["idempotency_hit"] is True
    assert second_payload["decision_key"] == first_payload["decision_key"]
    assert "Status: NO-OP (idempotent)" in second_run.stdout


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
    _assert_run_daily_payload_contract(payload, expected_mode="paper")
    assert payload["status"] == "failed"
    assert "missing price_usd values" in payload["message"]
    assert "Status: FAILED" in run.stdout


def test_run_daily_force_rerun_executes_with_new_run_key(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)
    state_db = tmp_path / "state.sqlite3"
    output_dir = tmp_path / "daily"
    base_args = [
        "strategy",
        "run-daily",
        "--strategy",
        "stacksats.strategies.examples:RunDailyPaperStrategy",
        "--run-date",
        "2024-12-31",
        "--mode",
        "paper",
        "--state-db-path",
        str(state_db),
        "--output-dir",
        str(output_dir),
    ]

    first_run = _run_cli(*base_args, "--total-window-budget-usd", "1000", env=env)
    assert first_run.returncode == 0, first_run.stderr
    first_payload = _decode_leading_json(first_run.stdout)

    conflicting_run = _run_cli(*base_args, "--total-window-budget-usd", "2000", env=env)
    assert conflicting_run.returncode == 2
    assert "Use --force to rerun with new parameters." in conflicting_run.stderr

    forced_run = _run_cli(
        *base_args,
        "--total-window-budget-usd",
        "2000",
        "--force",
        env=env,
    )
    assert forced_run.returncode == 0, forced_run.stderr
    forced_payload = _decode_leading_json(forced_run.stdout)
    _assert_run_daily_payload_contract(forced_payload, expected_mode="paper")
    assert forced_payload["status"] == "executed"
    assert forced_payload["forced_rerun"] is True
    assert forced_payload["idempotency_hit"] is False
    assert forced_payload["run_key"] != first_payload["run_key"]
    _assert_path_under(forced_payload["artifact_path"], output_dir)


def test_run_daily_live_mode_with_temp_adapter_smoke(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)
    state_db = tmp_path / "state-live.sqlite3"
    output_dir = tmp_path / "daily-live"
    adapter_path = _write_live_adapter_module(tmp_path)

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
        "--adapter",
        f"{adapter_path}:TempAdapter",
        "--state-db-path",
        str(state_db),
        "--output-dir",
        str(output_dir),
        env=env,
    )

    assert run.returncode == 0, run.stderr
    payload = _decode_leading_json(run.stdout)
    _assert_run_daily_payload_contract(payload, expected_mode="live")
    assert payload["status"] == "executed"
    assert payload["adapter_name"] == "TempAdapter"
    assert payload["order_receipt"]["external_order_id"].startswith("live-")
    assert payload["order_receipt"]["metadata"]["adapter"] == "temp-live"
    assert payload["order_receipt"]["metadata"]["mode"] == "live"
    artifact_path = _assert_path_under(payload["artifact_path"], output_dir)
    saved_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved_payload["mode"] == "live"
    assert saved_payload["adapter_name"] == "TempAdapter"


def test_run_daily_live_mode_with_broken_adapter_fails(tmp_path: Path) -> None:
    env = _demo_env(tmp_path)
    state_db = tmp_path / "state-live-bad.sqlite3"
    output_dir = tmp_path / "daily-live-bad"
    adapter_path = _write_broken_live_adapter_module(tmp_path)

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
        "--adapter",
        f"{adapter_path}:BrokenAdapter",
        "--state-db-path",
        str(state_db),
        "--output-dir",
        str(output_dir),
        env=env,
    )

    assert run.returncode == 1
    payload = _decode_leading_json(run.stdout)
    _assert_run_daily_payload_contract(payload, expected_mode="live")
    assert payload["status"] == "failed"
    assert payload["artifact_path"] is None
    assert payload["message"] == (
        "Daily execution failed: Execution adapter must return DailyOrderReceipt."
    )
    assert Path(str(payload["state_db_path"])) == state_db.resolve()
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
    _assert_required_keys(
        payload,
        {"status", "has_price_usd", "has_mvrv", "has_daily_gaps", "gap_count"},
    )
    assert payload["status"] == "warning"
    assert payload["has_daily_gaps"] is True
    assert payload["gap_count"] == 1
