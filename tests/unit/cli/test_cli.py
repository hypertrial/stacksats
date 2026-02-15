from __future__ import annotations

import json
import runpy
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from stacksats import cli


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
    class FakeRunner:
        def export(self, strategy, config):
            del config
            return pd.DataFrame({"id": [1, 2]})

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
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert '"rows": 2' in out


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
    class FakeRunner:
        def export(self, strategy, config):
            del strategy, config
            return pd.DataFrame({"id": [1]})

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
        ],
    )

    runpy.run_module("stacksats.cli", run_name="__main__")
