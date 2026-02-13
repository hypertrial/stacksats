from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

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
