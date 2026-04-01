from __future__ import annotations

from datetime import date
import subprocess
import sys
from pathlib import Path

import scripts.test_example_commands as example_commands


class _FrozenDate(date):
    @classmethod
    def today(cls) -> "_FrozenDate":
        return cls(2026, 3, 28)


def _tempdir_context(tmp_path: Path):
    class _Ctx:
        def __enter__(self):
            return str(tmp_path / "example-smoke-home")

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    return _Ctx()


def _record_commands(monkeypatch, tmp_path: Path, *, venv_exists: bool, returncodes: list[int]):
    recorded: list[dict[str, object]] = []
    writes: list[tuple[Path, date, int]] = []
    iterator = iter(returncodes)
    fake_script = tmp_path / "scripts" / "test_example_commands.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)
    fake_script.write_text("# fake script path for tests\n", encoding="utf-8")
    root_dir = fake_script.resolve().parents[1]
    venv_python = root_dir / "venv" / "bin" / "python"

    monkeypatch.setattr(example_commands, "date", _FrozenDate)
    monkeypatch.setattr(example_commands, "__file__", str(fake_script))
    monkeypatch.setattr(
        example_commands.tempfile,
        "TemporaryDirectory",
        lambda prefix: _tempdir_context(tmp_path),
    )
    if venv_exists:
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    def _fake_write(pq_path: Path, *, end_date: date, lookback_days: int = 3300) -> None:
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pq_path.write_bytes(b"parquet")
        writes.append((pq_path, end_date, lookback_days))

    def _fake_run(cmd, cwd, env):
        recorded.append({"cmd": list(cmd), "cwd": cwd, "env": dict(env)})
        return subprocess.CompletedProcess(cmd, next(iterator))

    monkeypatch.setattr(example_commands, "_write_synthetic_brk_parquet", _fake_write)
    monkeypatch.setattr(example_commands.subprocess, "run", _fake_run)
    return recorded, writes, root_dir, venv_python


def test_example_commands_main_uses_venv_python_when_available(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    recorded, writes, root_dir, venv_python = _record_commands(
        monkeypatch,
        tmp_path,
        venv_exists=True,
        returncodes=[0] * 9,
    )

    exit_code = example_commands.main()

    assert exit_code == 0
    assert len(recorded) == 9
    assert len(writes) == 1
    parquet_path, end_date, lookback_days = writes[0]
    assert parquet_path == tmp_path / "example-smoke-home" / "bitcoin_analytics.parquet"
    assert end_date == date(2026, 3, 28)
    assert lookback_days == 3300
    expected_output_dir = str(tmp_path / "example-smoke-home" / "output")
    expected_state_db_path = str(
        tmp_path / "example-smoke-home" / ".stacksats" / "run_state.sqlite3"
    )

    first = recorded[0]
    assert first["cwd"] == str(tmp_path / "example-smoke-home")
    assert first["cmd"][:3] == [str(venv_python), "-m", "stacksats.cli"]
    assert first["cmd"][3:] == ["demo", "backtest", "--output-dir", expected_output_dir]

    commands = [entry["cmd"][3:] for entry in recorded]
    assert ["strategy", "validate", "--strategy", example_commands.EXAMPLE_SPEC, "--start-date", "2025-03-28", "--end-date", "2026-03-28", "--min-win-rate", "0.0", "--no-strict"] in commands
    assert ["strategy", "backtest", "--strategy", example_commands.EXAMPLE_SPEC, "--start-date", "2025-03-28", "--end-date", "2026-03-28"] in commands
    assert ["strategy", "export", "--strategy", example_commands.EXAMPLE_SPEC, "--start-date", "2025-03-28", "--end-date", "2026-03-28", "--output-dir", expected_output_dir] in commands
    assert [
        "strategy",
        "decide-daily",
        "--strategy",
        "run-daily-paper",
        "--run-date",
        "2026-03-28",
        "--total-window-budget-usd",
        "1000",
        "--state-db-path",
        expected_state_db_path,
        "--output-dir",
        expected_output_dir,
    ] in commands

    env = first["env"]
    assert env["PATH"].startswith(str(venv_python.parent))
    assert env["HOME"] == str(tmp_path / "example-smoke-home")
    assert env["MPLCONFIGDIR"] == str(tmp_path / "example-smoke-home" / ".mplconfig")
    assert env["PYTHONPATH"].split(":")[0] == str(root_dir)
    assert env["STACKSATS_ANALYTICS_PARQUET"] == str(
        tmp_path / "example-smoke-home" / "bitcoin_analytics.parquet"
    )

    stdout = capsys.readouterr().out
    assert "Passed: 9" in stdout
    assert "Failed: 0" in stdout
    assert "Skipped: 2" in stdout


def test_example_commands_main_falls_back_to_runtime_python_and_reports_failures(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    recorded, _, root_dir, _ = _record_commands(
        monkeypatch,
        tmp_path,
        venv_exists=False,
        returncodes=[0, 1, 0, 0, 0, 0, 0, 0, 0],
    )

    exit_code = example_commands.main()

    assert exit_code == 1
    assert len(recorded) == 9
    first = recorded[0]
    assert first["cwd"] == str(tmp_path / "example-smoke-home")
    assert first["cmd"][:3] == [str(Path(sys.executable).resolve()), "-m", "stacksats.cli"]
    assert first["env"]["PYTHONPATH"].split(":")[0] == str(root_dir)

    stdout = capsys.readouterr().out
    assert "using current interpreter" in stdout
    assert "Passed: 8" in stdout
    assert "Failed: 1" in stdout
    assert "Skipped: 2" in stdout
