from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from stacksats import cli


def test_float_or_default_handles_invalid_and_non_finite_values() -> None:
    assert cli._float_or_default("bad", default=1.5) == 1.5
    assert cli._float_or_default(float("inf"), default=2.5) == 2.5
    assert cli._float_or_default(None, default=3.5) == 3.5
    assert cli._float_or_default("4.0", default=0.0) == 4.0


def test_backtest_result_from_json_uses_fallbacks_for_malformed_sections(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "backtest.json"
    payload_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "stacksats.viz.animation_data.load_backtest_payload",
        lambda path: {"summary_metrics": "bad", "provenance": []},
    )
    monkeypatch.setattr(
        "stacksats.viz.animation_data.spd_table_from_backtest_payload",
        lambda payload: pl.DataFrame({"date": [], "dynamic_percentile": [], "uniform_percentile": []}),
    )

    result = cli._backtest_result_from_json(payload_path)

    assert result.win_rate == 0.0
    assert result.score == 0.0
    assert result.exp_decay_percentile == 0.0
    assert result.uniform_exp_decay_percentile == 0.0
    assert result.strategy_id == "unknown"
    assert result.strategy_version == "0.0.0"
    assert result.config_hash == ""
    assert result.run_id == ""


def test_cli_reconcile_daily_path_prints_payload(monkeypatch, capsys) -> None:
    class _Runner:
        def reconcile_daily_run(self, strategy, run_date, mode, state_db_path):
            del strategy
            assert run_date == "2025-01-02"
            assert mode == "paper"
            assert state_db_path == ".stacksats/run_state.sqlite3"
            return {"status": "ok", "reconciled": True}

    monkeypatch.setattr(cli, "StrategyRunner", lambda: _Runner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())

    code = cli.main(
        [
            "strategy",
            "reconcile-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--run-date",
            "2025-01-02",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"status": "ok", "reconciled": True}


def test_cli_reconcile_daily_forwards_strategy_config(monkeypatch, capsys, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class _Runner:
        def reconcile_daily_run(self, strategy, run_date, mode, state_db_path):
            observed["strategy"] = strategy
            observed["run_date"] = run_date
            observed["mode"] = mode
            observed["state_db_path"] = state_db_path
            return {"status": "ok", "reconciled": True}

    def _load_strategy(spec, *, config_path=None):
        observed["strategy_spec"] = spec
        observed["config_path"] = config_path
        return object()

    config_path = tmp_path / "strategy.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "StrategyRunner", lambda: _Runner())
    monkeypatch.setattr(cli, "load_strategy", _load_strategy)

    code = cli.main(
        [
            "strategy",
            "reconcile-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--strategy-config",
            str(config_path),
            "--run-date",
            "2025-01-02",
        ]
    )

    assert code == 0
    assert observed["strategy_spec"] == "dummy.py:Dummy"
    assert observed["config_path"] == str(config_path)
    assert observed["run_date"] == "2025-01-02"
    assert observed["mode"] == "paper"
    assert observed["state_db_path"] == ".stacksats/run_state.sqlite3"
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "reconciled": True}


def test_cli_reconcile_daily_missing_run_exits_user_error(monkeypatch, capsys) -> None:
    class _Runner:
        def reconcile_daily_run(self, strategy, run_date, mode, state_db_path):
            del strategy, run_date, mode, state_db_path
            raise ValueError("No stored daily run exists for the requested key.")

    monkeypatch.setattr(cli, "StrategyRunner", lambda: _Runner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "strategy",
                "reconcile-daily",
                "--strategy",
                "dummy.py:Dummy",
                "--run-date",
                "2025-01-02",
            ]
        )
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "No stored daily run exists for the requested key." in err


def test_cli_reconcile_daily_invalid_strategy_config_exits_user_error(
    monkeypatch, capsys
) -> None:
    def _raise_json_error(*args, **kwargs):
        del args, kwargs
        raise json.JSONDecodeError("bad", doc="{", pos=1)

    monkeypatch.setattr(cli, "load_strategy", _raise_json_error)

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "strategy",
                "reconcile-daily",
                "--strategy",
                "dummy.py:Dummy",
                "--strategy-config",
                "bad.json",
                "--run-date",
                "2025-01-02",
            ]
        )
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "Invalid JSON in strategy config file." in err


def test_run_lifecycle_command_unsupported_raises() -> None:
    """_run_lifecycle_command with unsupported command raises ValueError."""
    from stacksats.runner import StrategyRunner

    parser = cli._build_parser()
    args = parser.parse_args(
        ["strategy", "validate", "--strategy", "simple-zscore"]
    )
    with pytest.raises(ValueError, match="Unsupported lifecycle command"):
        cli._run_lifecycle_command("unsupported", args, StrategyRunner())


def test_cli_data_unsupported_command_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """data command with unsupported subcommand triggers parser.error."""
    original_build = cli._build_parser

    def _build_parser_with_unsupported():
        parser = original_build()
        for action in parser._actions:
            if hasattr(action, "choices") and action.choices and "data" in action.choices:
                data_parser = action.choices["data"]
                for sub in data_parser._actions:
                    if hasattr(sub, "choices") and sub.choices is not None:
                        sub.add_parser("unsupported", help="hidden")
                        break
                break
        return parser
    monkeypatch.setattr(cli, "_build_parser", _build_parser_with_unsupported)
    with pytest.raises(SystemExit):
        cli.run(["data", "unsupported"])
