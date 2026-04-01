from __future__ import annotations

import importlib
import json
import runpy
import subprocess
import sys
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import pytest

from stacksats import cli
from stacksats.strategy_types import RunDailyConfig


def test_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-m", "stacksats.cli", "--help"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "strategy" in proc.stdout
    assert "demo" in proc.stdout
    assert "data" in proc.stdout
    assert "serve" in proc.stdout


def test_cli_help_examples_use_stable_packaged_strategy_ids() -> None:
    help_text = cli._build_parser().format_help()
    assert "simple-zscore" in help_text
    assert "run-daily-paper" in help_text
    assert "strategy_id values" in help_text
    assert "stacksats.strategies.experimental.model_example:ExampleMVRVStrategy" not in help_text
    assert "stacksats demo backtest" in help_text
    assert "stacksats data fetch" in help_text
    assert "stacksats strategy decide-daily" in help_text


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


def test_cli_strategy_decide_daily_maps_config(monkeypatch, capsys, tmp_path) -> None:
    class FakeDecisionResult:
        status = "decided"

        def to_json(self):
            return {"status": "decided", "decision_key": "decision-1"}

    class FakeRunner:
        def decide_daily(self, strategy, config):
            del strategy
            assert config.total_window_budget_usd == 1000.0
            assert config.btc_price_col == "price_usd"
            return FakeDecisionResult()

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
            "decide-daily",
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
    assert '"status": "decided"' in out
    assert "Status: DECIDED" in out


def test_cli_serve_agent_api_maps_config(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    def fake_start(args) -> None:
        observed["host"] = args.host
        observed["port"] = args.port
        observed["registry_path"] = args.registry_path
        observed["state_db_path"] = args.state_db_path
        observed["output_dir"] = args.output_dir
        observed["auth_token_env"] = args.auth_token_env
        observed["btc_price_col_default"] = args.btc_price_col_default

    monkeypatch.setattr(cli, "_start_agent_service_from_args", fake_start)

    code = cli.main(
        [
            "serve",
            "agent-api",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--registry-path",
            str(tmp_path / "registry.json"),
            "--state-db-path",
            str(tmp_path / "state.sqlite3"),
            "--output-dir",
            str(tmp_path / "output"),
            "--auth-token-env",
            "CUSTOM_STACKSATS_TOKEN",
            "--btc-price-col-default",
            "close_usd",
        ]
    )

    assert code == 0
    assert observed["host"] == "0.0.0.0"
    assert observed["port"] == 9001
    assert observed["registry_path"] == str(tmp_path / "registry.json")
    assert observed["state_db_path"] == str(tmp_path / "state.sqlite3")
    assert observed["output_dir"] == str(tmp_path / "output")
    assert observed["auth_token_env"] == "CUSTOM_STACKSATS_TOKEN"
    assert observed["btc_price_col_default"] == "close_usd"


def test_cli_serve_agent_api_missing_service_extra_exits_user_error(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "_start_agent_service_from_args",
        lambda _args: (_ for _ in ()).throw(
            ImportError(
                "Missing optional dependency 'fastapi'. Install 'service' extras with: "
                "pip install \"stacksats[service]\" to use agent HTTP service."
            )
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "agent-api"])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "stacksats[service]" in err


def test_start_agent_service_from_args_builds_service_config(monkeypatch) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "stacksats.service.start_agent_service",
        lambda config: observed.setdefault("config", config),
    )

    cli._start_agent_service_from_args(
        SimpleNamespace(
            host="0.0.0.0",
            port=9001,
            registry_path="/tmp/registry.json",
            state_db_path="/tmp/state.sqlite3",
            output_dir="/tmp/output",
            auth_token_env="TOKEN_ENV",
            btc_price_col_default="close_usd",
        )
    )

    config = observed["config"]
    assert config.host == "0.0.0.0"
    assert config.port == 9001
    assert config.registry_path == "/tmp/registry.json"
    assert config.auth_token_env == "TOKEN_ENV"


def test_cli_unsupported_serve_command_errors() -> None:
    class FakeParser:
        def parse_args(self, argv):
            del argv
            return SimpleNamespace(
                command="serve",
                serve_command="unknown",
                strategy_command=None,
                demo_command=None,
                data_command=None,
            )

        def error(self, message):
            raise SystemExit(message)

    original = cli._build_parser
    cli._build_parser = lambda: FakeParser()
    try:
        with pytest.raises(SystemExit) as exc:
            cli.run([])
    finally:
        cli._build_parser = original

    assert "Unsupported serve command." in str(exc.value)


def test_cli_strategy_run_daily_forwards_strategy_config(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    class FakeRunResult:
        status = "executed"

        @staticmethod
        def to_json():
            return {"status": "executed", "run_key": "rk-1"}

    class FakeRunner:
        def run_daily(self, strategy, config):
            observed["strategy"] = strategy
            observed["config"] = config
            return FakeRunResult()

    class FakeStrategy:
        strategy_id = "fake-strategy"
        version = "1.0.0"

    def fake_load_strategy(spec, *, config_path=None):
        observed["strategy_spec"] = spec
        observed["config_path"] = config_path
        return FakeStrategy()

    config_path = tmp_path / "strategy.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", fake_load_strategy)

    code = cli.main(
        [
            "strategy",
            "run-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--strategy-config",
            str(config_path),
            "--total-window-budget-usd",
            "1000",
            "--state-db-path",
            str(tmp_path / "state.sqlite3"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert code == 0
    assert observed["strategy_spec"] == "dummy.py:Dummy"
    assert observed["config_path"] == str(config_path)
    assert isinstance(observed["config"], RunDailyConfig)


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


def test_cli_strategy_decide_daily_failure_exits_nonzero_and_prints_status(
    monkeypatch,
    capsys,
) -> None:
    class FakeDecisionResult:
        status = "failed"

        @staticmethod
        def to_json():
            return {"status": "failed", "message": "Strict validation failed before daily decision."}

    class FakeRunner:
        def decide_daily(self, strategy, config):
            del strategy, config
            return FakeDecisionResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "strategy",
                "decide-daily",
                "--strategy",
                "dummy.py:Dummy",
                "--total-window-budget-usd",
                "1000",
            ]
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"status": "failed"' in out
    assert "Status: FAILED" in out


def test_cli_strategy_decide_daily_noop_prints_idempotent_status(
    monkeypatch,
    capsys,
) -> None:
    class FakeDecisionResult:
        status = "noop"

        @staticmethod
        def to_json():
            return {"status": "noop", "decision_key": "decision-1"}

    class FakeRunner:
        def decide_daily(self, strategy, config):
            del strategy, config
            return FakeDecisionResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())

    code = cli.main(
        [
            "strategy",
            "decide-daily",
            "--strategy",
            "dummy.py:Dummy",
            "--total-window-budget-usd",
            "1000",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert '"status": "noop"' in out
    assert "Status: NO-OP (idempotent)" in out


def test_cli_strategy_run_daily_failure_prints_adapter_error_payload(
    monkeypatch, capsys
) -> None:
    class FakeRunResult:
        status = "failed"

        @staticmethod
        def to_json():
            return {
                "status": "failed",
                "message": "Daily execution failed: Execution adapter must return DailyOrderReceipt.",
            }

    class FakeRunner:
        def run_daily(self, strategy, config):
            del strategy, config
            return FakeRunResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "strategy",
                "run-daily",
                "--strategy",
                "dummy.py:Dummy",
                "--total-window-budget-usd",
                "1000",
                "--mode",
                "live",
                "--adapter",
                "bad.py:BadAdapter",
            ]
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"status": "failed"' in out
    assert "Daily execution failed: Execution adapter must return DailyOrderReceipt." in out
    assert "Status: FAILED" in out


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


def test_cli_strategy_run_daily_invalid_strategy_config_exits_user_error(
    monkeypatch, capsys
) -> None:
    def _raise_json_error(*args, **kwargs):
        del args, kwargs
        raise json.JSONDecodeError("bad", doc="{", pos=1)

    monkeypatch.setattr(cli, "load_strategy", _raise_json_error)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "strategy",
                "run-daily",
                "--strategy",
                "dummy.py:Dummy",
                "--strategy-config",
                "bad.json",
                "--total-window-budget-usd",
                "1000",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Invalid JSON in strategy config file." in err


def test_cli_unsupported_command_routes_to_parser_error(monkeypatch) -> None:
    class FakeParser:
        def parse_args(self, argv=None):
            del argv
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
        with pytest.raises(SystemExit) as raised:
            runpy.run_module("stacksats.cli", run_name="__main__")
    assert raised.value.code == 0


def test_cli_demo_backtest_uses_packaged_demo_data(monkeypatch, tmp_path) -> None:
    observed = {}

    class FakeResult:
        strategy_id = "demo-strategy"
        strategy_version = "1.0.0"
        run_id = "run-demo"

        def summary(self) -> str:
            return "Score: 55.00%"

        def plot(self, output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return {}

        def to_json(self, path):
            Path(path).write_text("{}", encoding="utf-8")

    class FakeRunner:
        def __init__(self, data_provider=None):
            observed["provider_path"] = getattr(data_provider, "parquet_path", None)

        def backtest(self, strategy, config):
            del strategy
            observed["start_date"] = config.start_date
            observed["end_date"] = config.end_date
            return FakeResult()

    @contextmanager
    def fake_demo_path():
        demo = tmp_path / "demo.parquet"
        demo.write_bytes(b"demo")
        yield demo

    monkeypatch.setattr(cli, "StrategyRunner", FakeRunner)
    monkeypatch.setattr(cli, "packaged_demo_parquet_path", fake_demo_path)

    class FakeStrategy:
        strategy_id = "demo-strategy"

    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: FakeStrategy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "demo",
            "backtest",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    cli.main()
    assert observed["provider_path"] == str(tmp_path / "demo.parquet")
    assert observed["start_date"] == "2018-01-01"
    assert observed["end_date"] == "2025-12-31"


def test_cli_package_init_dunder_main_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBatch:
        row_count = 1
        window_count = 1
        schema_version = "1.0.0"
        run_id = "run-init"

        def to_csv(self, path):
            Path(path).write_text("date,weight\n", encoding="utf-8")

    class FakeRunner:
        def export(self, strategy, config):
            del strategy, config
            return FakeBatch()

    class FakeStrategy:
        strategy_id = "fake-init"
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
        with pytest.raises(SystemExit) as raised:
            runpy.run_module("stacksats.cli.__init__", run_name="__main__")

    assert raised.value.code == 0


def test_cli_package_main_dunder_main_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBatch:
        row_count = 1
        window_count = 1
        schema_version = "1.0.0"
        run_id = "run-main-module"

        def to_csv(self, path):
            Path(path).write_text("date,weight\n", encoding="utf-8")

    class FakeRunner:
        def export(self, strategy, config):
            del strategy, config
            return FakeBatch()

    class FakeStrategy:
        strategy_id = "fake-main-module"
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
        with pytest.raises(SystemExit) as raised:
            runpy.run_module("stacksats.cli.__main__", run_name="__main__")

    assert raised.value.code == 0


def test_cli_package_main_import_does_not_exit() -> None:
    module = importlib.import_module("stacksats.cli.__main__")

    assert callable(module.main)


def test_cli_data_fetch_invokes_fetch_assets(monkeypatch, capsys, tmp_path) -> None:
    observed = {}

    def fake_fetch_assets(**kwargs):
        observed.update(kwargs)
        parquet = tmp_path / "brk" / "merged_metrics.parquet"
        schema = tmp_path / "brk" / "schema.md"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"x")
        schema.write_text("schema", encoding="utf-8")
        return parquet, schema

    monkeypatch.setattr(cli, "fetch_assets", fake_fetch_assets)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "data",
            "fetch",
            "--target-dir",
            str(tmp_path / "brk"),
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert '"next": "stacksats data prepare"' in out
    assert str(tmp_path / "brk") in out
    assert observed["overwrite"] is False


def test_cli_data_prepare_uses_latest_fetched_parquet(monkeypatch, capsys, tmp_path) -> None:
    source = tmp_path / "brk" / "merged_metrics.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x")
    output = tmp_path / "managed" / "bitcoin_analytics.parquet"
    observed = {}

    monkeypatch.setattr(cli, "latest_fetched_parquet", lambda: source)

    def fake_prepare_runtime_parquet(src, *, output, overwrite):
        observed["source"] = src
        observed["output"] = output
        observed["overwrite"] = overwrite
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"prepared")
        return Path(output)

    monkeypatch.setattr(cli, "prepare_runtime_parquet", fake_prepare_runtime_parquet)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "data",
            "prepare",
            "--output",
            str(output),
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert str(source) in out
    assert observed["source"] == source
    assert observed["output"] == output
    assert observed["overwrite"] is False


def test_cli_data_doctor_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "data_doctor",
        lambda path_override=None: {
            "status": "missing",
            "resolved_path": None,
            "path_override": path_override,
        },
    )
    monkeypatch.setattr(sys, "argv", ["stacksats", "data", "doctor"])

    cli.main()
    out = capsys.readouterr().out
    assert '"status": "missing"' in out
