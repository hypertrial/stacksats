from __future__ import annotations

import json
import runpy
import subprocess
import sys
import warnings
from types import SimpleNamespace
from pathlib import Path

import pytest

from stacksats import cli
from stacksats.strategy_types import RunDailyConfig


def test_cli_help() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "stacksats.cli", "--help"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "strategy" in proc.stdout


def test_cli_strategy_validate_uses_runner(monkeypatch, capsys) -> None:
    class FakeResult:
        passed = True
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class FakeRunner:
        def validate(self, strategy, config):
            del strategy
            assert config.min_win_rate == 50.0
            return FakeResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "validate",
            "--strategy",
            "dummy.py:Dummy",
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert "Validation PASSED" in out


def test_cli_strategy_validate_strict_flag(monkeypatch, capsys) -> None:
    class FakeResult:
        passed = True
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class FakeRunner:
        def validate(self, strategy, config):
            del strategy
            assert config.strict is True
            return FakeResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "validate",
            "--strategy",
            "dummy.py:Dummy",
            "--strict",
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert "Validation PASSED" in out


def test_cli_strategy_backtest_writes_strategy_addressable_output(monkeypatch, tmp_path) -> None:
    class FakeBacktestResult:
        strategy_id = "fake-strategy"
        strategy_version = "9.9.9"
        run_id = "run-123"

        def summary(self) -> str:
            return "Score: 50.00%"

        def plot(self, output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return {"metrics_json": str(Path(output_dir) / "metrics.json")}

        def to_json(self, path):
            Path(path).write_text(json.dumps({"ok": True}), encoding="utf-8")

    class FakeRunner:
        def backtest(self, strategy, config):
            del strategy
            assert config.strategy_label == "fake-strategy"
            return FakeBacktestResult()

    class FakeStrategy:
        strategy_id = "fake-strategy"

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: FakeStrategy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "backtest",
            "--strategy",
            "dummy.py:Dummy",
            "--output-dir",
            str(tmp_path),
        ],
    )

    cli.main()
    expected = tmp_path / "fake-strategy" / "9.9.9" / "run-123" / "backtest_result.json"
    assert expected.exists()


def test_cli_strategy_export_emits_json_summary(monkeypatch, capsys) -> None:
    class FakeBatch:
        row_count = 2
        window_count = 1
        schema_version = "1.0.0"
        run_id = "run-123"

    class FakeRunner:
        def export(self, strategy, config):
            del config
            return FakeBatch()

    class FakeStrategy:
        strategy_id = "fake-strategy"
        version = "1.0.0"

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: FakeStrategy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "export",
            "--strategy",
            "dummy.py:Dummy",
            "--start-date",
            "2025-12-01",
            "--end-date",
            "2027-12-31",
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert '"rows": 2' in out
    assert '"windows": 1' in out


def test_cli_strategy_run_daily_maps_config(monkeypatch, capsys, tmp_path) -> None:
    class FakeRunResult:
        status = "executed"

        def to_json(self):
            return {"status": "executed", "run_key": "rk-1"}

    class FakeRunner:
        def run_daily(self, strategy, config):
            del strategy
            assert isinstance(config, RunDailyConfig)
            assert config.total_window_budget_usd == 1000.0
            assert config.mode == "paper"
            return FakeRunResult()

    class FakeStrategy:
        strategy_id = "fake-strategy"
        version = "1.0.0"

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: FakeStrategy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "run-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--total-window-budget-usd",
            "1000",
            "--state-db-path",
            str(tmp_path / "state.sqlite3"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert '"status": "executed"' in out
    assert "Status: EXECUTED" in out


def test_cli_strategy_run_daily_failure_exits_nonzero(monkeypatch) -> None:
    class FakeRunResult:
        status = "failed"

        def to_json(self):
            return {"status": "failed"}

    class FakeRunner:
        def run_daily(self, strategy, config):
            del strategy, config
            return FakeRunResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "run-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--total-window-budget-usd",
            "1000",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_cli_strategy_run_daily_live_without_adapter_exits_user_error(
    monkeypatch, capsys
) -> None:
    class FakeRunner:
        def run_daily(self, strategy, config):
            del strategy
            if config.mode == "live" and config.adapter_spec is None:
                raise ValueError("Live mode requires --adapter.")
            raise AssertionError("Expected invalid config to fail before execution.")

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "run-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--total-window-budget-usd",
            "1000",
            "--mode",
            "live",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Live mode requires --adapter." in err


def test_cli_unsupported_command_routes_to_parser_error(monkeypatch) -> None:
    class FakeParser:
        def parse_args(self):
            return SimpleNamespace(
                strategy="dummy.py:Dummy",
                strategy_config=None,
                strategy_command="unknown",
            )

        def error(self, message):
            raise RuntimeError(message)

    monkeypatch.setattr(cli, "_build_parser", lambda: FakeParser())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "StrategyRunner", lambda: object())

    with pytest.raises(RuntimeError, match="Unsupported command"):
        cli.main()


def test_cli_module_dunder_main_executes(monkeypatch) -> None:
    class FakeBatch:
        row_count = 1
        window_count = 1
        schema_version = "1.0.0"
        run_id = "run-main"

    class FakeRunner:
        def export(self, strategy, config):
            del strategy, config
            return FakeBatch()

    class FakeStrategy:
        strategy_id = "fake-main"
        version = "1.0.0"

    monkeypatch.setattr("stacksats.loader.load_strategy", lambda *args, **kwargs: FakeStrategy())
    monkeypatch.setattr("stacksats.runner.StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "export",
            "--strategy",
            "dummy.py:Dummy",
            "--start-date",
            "2025-12-01",
            "--end-date",
            "2027-12-31",
        ],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'.*' found in sys.modules after import of package '.*'",
            category=RuntimeWarning,
        )
        runpy.run_module("stacksats.cli", run_name="__main__")
