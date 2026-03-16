from __future__ import annotations

import json
from pathlib import Path

import polars as pl

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
        "stacksats.animation_data.load_backtest_payload",
        lambda path: {"summary_metrics": "bad", "provenance": []},
    )
    monkeypatch.setattr(
        "stacksats.animation_data.spd_table_from_backtest_payload",
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
