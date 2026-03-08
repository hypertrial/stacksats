from __future__ import annotations

from json import JSONDecodeError
from types import SimpleNamespace

import pytest
import requests

from stacksats import cli
from stacksats.data_btc import DataLoadError


def _parser_args() -> SimpleNamespace:
    return SimpleNamespace(
        strategy="dummy.py:Dummy",
        strategy_config=None,
        strategy_command="validate",
        start_date=None,
        end_date=None,
        min_win_rate=50.0,
        strict=False,
    )


def test_cli_run_daily_noop_status_prints_idempotent(monkeypatch, capsys) -> None:
    class _Result:
        status = "noop"

        @staticmethod
        def to_json():
            return {"status": "noop"}

    class _Runner:
        def run_daily(self, strategy, config):
            del strategy, config
            return _Result()

    monkeypatch.setattr(
        cli,
        "_build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                strategy="dummy.py:Dummy",
                strategy_config=None,
                strategy_command="run-daily",
                run_date=None,
                total_window_budget_usd=1000.0,
                mode="paper",
                state_db_path=".stacksats/run_state.sqlite3",
                output_dir="output",
                adapter=None,
                force=False,
                btc_price_col="price_usd",
            ),
            error=lambda message: (_ for _ in ()).throw(RuntimeError(message)),
        ),
    )
    monkeypatch.setattr(cli, "load_strategy", lambda *a, **k: object())
    monkeypatch.setattr(cli, "StrategyRunner", lambda: _Runner())
    cli.main()
    out = capsys.readouterr().out
    assert "Status: NO-OP (idempotent)" in out


@pytest.mark.parametrize(
    ("exc", "expected", "hint_expected"),
    [
        (JSONDecodeError("bad", doc="x", pos=0), "Invalid JSON in strategy config file.", True),
        (FileNotFoundError("missing"), "missing", False),
        (ModuleNotFoundError("missing_mod"), "Could not import strategy module", True),
        (AttributeError("bad attr"), "bad attr", False),
        (requests.RequestException("net"), "Failed to fetch required network data.", True),
        (DataLoadError("bad data"), "bad data", False),
        (ValueError("bad val"), "bad val", False),
    ],
)
def test_cli_error_handlers_exit_with_user_errors(
    monkeypatch, capsys, exc: Exception, expected: str, hint_expected: bool
) -> None:
    parser = SimpleNamespace(
        parse_args=_parser_args,
        error=lambda message: (_ for _ in ()).throw(RuntimeError(message)),
    )
    monkeypatch.setattr(cli, "_build_parser", lambda: parser)
    monkeypatch.setattr(cli, "StrategyRunner", lambda: object())

    def _raiser(*args, **kwargs):
        raise exc

    monkeypatch.setattr(cli, "load_strategy", _raiser)
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert expected in err
    if hint_expected:
        assert "Hint:" in err
