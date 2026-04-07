"""Strategy-centric CLI for StackSats."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

import numpy as np

from ..data.data_btc import BTCDataProvider, DataLoadError
from ..data.data_setup import (
    MANAGED_BRK_DIR,
    MANAGED_RUNTIME_PARQUET,
    data_doctor,
    fetch_assets,
    latest_fetched_parquet,
    packaged_demo_parquet_path,
    prepare_runtime_parquet,
)
from ..loader import load_strategy
from ..runner import StrategyRunner
from ..strategies.catalog import list_strategies
from ..strategy_types import (
    AgentServiceConfig,
    BacktestConfig,
    ComparisonConfig,
    DecideDailyConfig,
    ExportConfig,
    RunDailyConfig,
    ValidationConfig,
)

DEMO_STRATEGY_SPEC = "simple-zscore"
DEMO_START_DATE = "2018-01-01"
DEMO_END_DATE = "2025-12-31"

try:
    import requests as _requests
except ImportError:  # pragma: no cover - exercised only without network extras
    _requests = None


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults and preserve example formatting."""


def _is_requests_exception(exc: Exception) -> bool:
    return _requests is not None and isinstance(exc, _requests.RequestException)


def _ordered_compare_selectors(baseline: str, selectors: list[str]) -> list[str]:
    """Return baseline first, then unique selectors (catalog-style ordering)."""
    ordered = [baseline, *selectors]
    seen: set[str] = set()
    unique: list[str] = []
    for sel in ordered:
        if sel in seen:
            continue
        seen.add(sel)
        unique.append(sel)
    return unique


def _exit_user_error(message: str, *, hint: str | None = None, code: int = 2) -> None:
    print(f"Error: {message}", file=sys.stderr)
    if hint:
        print(f"Hint: {hint}", file=sys.stderr)
    raise SystemExit(code)


def _add_strategy_spec_arguments(command, *, required: bool, default: str | None) -> None:
    command.add_argument(
        "--strategy",
        required=required,
        default=default,
        help="built-in strategy_id or module_or_path:ClassName",
    )
    command.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )


def _add_validate_command(
    subparsers,
    *,
    help_text: str,
    command_name: str,
    strategy_required: bool,
    default_strategy: str | None,
    default_start: str | None,
    default_end: str | None,
    default_min_win_rate: float,
    default_strict: bool,
) -> argparse.ArgumentParser:
    command = subparsers.add_parser(
        command_name,
        help=help_text,
        formatter_class=_HelpFormatter,
    )
    _add_strategy_spec_arguments(
        command,
        required=strategy_required,
        default=default_strategy,
    )
    command.add_argument("--start-date", default=default_start)
    command.add_argument("--end-date", default=default_end)
    command.add_argument("--min-win-rate", type=float, default=default_min_win_rate)
    strict_group = command.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=default_strict,
        help="Run additional robustness gates.",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict robustness gates.",
    )
    return command


def _add_backtest_command(
    subparsers,
    *,
    help_text: str,
    command_name: str,
    strategy_required: bool,
    default_strategy: str | None,
    default_start: str | None,
    default_end: str | None,
) -> argparse.ArgumentParser:
    command = subparsers.add_parser(
        command_name,
        help=help_text,
        formatter_class=_HelpFormatter,
    )
    _add_strategy_spec_arguments(
        command,
        required=strategy_required,
        default=default_strategy,
    )
    command.add_argument("--start-date", default=default_start)
    command.add_argument("--end-date", default=default_end)
    command.add_argument("--output-dir", default="output")
    command.add_argument("--strategy-label", default=None)
    return command


def _add_export_command(
    subparsers,
    *,
    help_text: str,
    command_name: str,
    strategy_required: bool,
    default_strategy: str | None,
    default_start: str | None,
    default_end: str | None,
) -> argparse.ArgumentParser:
    command = subparsers.add_parser(
        command_name,
        help=help_text,
        formatter_class=_HelpFormatter,
    )
    _add_strategy_spec_arguments(
        command,
        required=strategy_required,
        default=default_strategy,
    )
    command.add_argument("--start-date", default=default_start, required=default_start is None)
    command.add_argument("--end-date", default=default_end, required=default_end is None)
    command.add_argument("--output-dir", default="output")
    return command


def _start_agent_service_from_args(args) -> None:
    from ..service import start_agent_service

    start_agent_service(
        AgentServiceConfig(
            host=args.host,
            port=args.port,
            registry_path=args.registry_path,
            state_db_path=args.state_db_path,
            output_dir=args.output_dir,
            auth_token_env=args.auth_token_env,
            btc_price_col_default=args.btc_price_col_default,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    stable_ids = ", ".join(entry.strategy_id for entry in list_strategies(tier="stable"))
    parser = argparse.ArgumentParser(
        prog="stacksats",
        description="StackSats CLI",
        formatter_class=_HelpFormatter,
        epilog=(
            "Built-in strategies accept strategy_id values such as simple-zscore "
            "and run-daily-paper.\n\n"
            "Examples:\n"
            "  stacksats demo backtest\n"
            "  stacksats data fetch\n"
            "  stacksats data prepare\n"
            "  stacksats strategy backtest --strategy "
            "simple-zscore --output-dir output\n"
            "  stacksats strategy decide-daily --strategy "
            "run-daily-paper --total-window-budget-usd 1000\n"
            "  stacksats serve agent-api --registry-path .stacksats/agent_service_registry.json\n"
            "  stacksats strategy animate --backtest-json "
            "output/my_strategy/1.0.0/run-1/backtest_result.json"
        ),
    )
    root = parser.add_subparsers(dest="command", required=True)

    strategy_parser = root.add_parser(
        "strategy",
        help="Strategy lifecycle commands",
        formatter_class=_HelpFormatter,
        epilog=(
            "Strategy selector formats:\n"
            "  built-in strategy_id\n"
            "  module_or_path:ClassName\n\n"
            f"Stable built-ins:\n  {stable_ids}\n\n"
            "Examples:\n"
            "  simple-zscore\n"
            "  my_strategy.py:MyStrategy"
        ),
    )
    strategy_sub = strategy_parser.add_subparsers(dest="strategy_command", required=True)
    _add_validate_command(
        strategy_sub,
        help_text="Validate strategy",
        command_name="validate",
        strategy_required=True,
        default_strategy=None,
        default_start=None,
        default_end=None,
        default_min_win_rate=50.0,
        default_strict=True,
    )
    _add_backtest_command(
        strategy_sub,
        help_text="Backtest strategy",
        command_name="backtest",
        strategy_required=True,
        default_strategy=None,
        default_start=None,
        default_end=None,
    )
    _add_export_command(
        strategy_sub,
        help_text="Export strategy artifacts",
        command_name="export",
        strategy_required=True,
        default_strategy=None,
        default_start=None,
        default_end=None,
    )
    compare_cmd = strategy_sub.add_parser(
        "compare",
        help="Compare strategies on a shared validation/backtest window",
        formatter_class=_HelpFormatter,
    )
    compare_cmd.add_argument(
        "--strategy",
        action="append",
        required=True,
        dest="compare_strategies",
        help="Built-in strategy_id or module_or_path:ClassName (repeatable)",
    )
    compare_cmd.add_argument(
        "--baseline",
        default="uniform",
        dest="compare_baseline",
        help="Baseline strategy_id used for relative deltas (must appear in the set).",
    )
    compare_cmd.add_argument("--start-date", default=None)
    compare_cmd.add_argument("--end-date", default=None)
    compare_cmd.add_argument("--min-win-rate", type=float, default=None)
    compare_cmd.add_argument(
        "--strict",
        dest="compare_strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override strict validation for the shared comparison run.",
    )
    compare_cmd.add_argument("--output-dir", default="output")
    compare_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="Optional JSON config path applied to each loaded strategy selector.",
    )
    decide_daily_cmd = strategy_sub.add_parser(
        "decide-daily",
        help="Generate an agent-facing daily decision payload",
        formatter_class=_HelpFormatter,
    )
    _add_strategy_spec_arguments(decide_daily_cmd, required=True, default=None)
    decide_daily_cmd.add_argument("--run-date", default=None)
    decide_daily_cmd.add_argument(
        "--total-window-budget-usd",
        required=True,
        type=float,
        help="Total USD budget associated with the full allocation window.",
    )
    decide_daily_cmd.add_argument("--state-db-path", default=".stacksats/run_state.sqlite3")
    decide_daily_cmd.add_argument("--output-dir", default="output")
    decide_daily_cmd.add_argument("--btc-price-col", default="price_usd")
    decide_daily_cmd.add_argument("--force", action="store_true")

    run_daily_cmd = strategy_sub.add_parser(
        "run-daily",
        help="Run idempotent daily execution",
        formatter_class=_HelpFormatter,
    )
    _add_strategy_spec_arguments(run_daily_cmd, required=True, default=None)
    run_daily_cmd.add_argument("--run-date", default=None)
    run_daily_cmd.add_argument(
        "--total-window-budget-usd",
        required=True,
        type=float,
        help="Total USD budget associated with the full allocation window.",
    )
    run_daily_cmd.add_argument("--mode", choices=("paper", "live"), default="paper")
    run_daily_cmd.add_argument("--adapter", default=None)
    run_daily_cmd.add_argument("--state-db-path", default=".stacksats/run_state.sqlite3")
    run_daily_cmd.add_argument("--output-dir", default="output")
    run_daily_cmd.add_argument("--btc-price-col", default="price_usd")
    run_daily_cmd.add_argument("--force", action="store_true")

    reconcile_cmd = strategy_sub.add_parser(
        "reconcile-daily",
        help="Reconcile a stored daily run against current data",
        formatter_class=_HelpFormatter,
    )
    _add_strategy_spec_arguments(reconcile_cmd, required=True, default=None)
    reconcile_cmd.add_argument("--run-date", required=True)
    reconcile_cmd.add_argument("--mode", choices=("paper", "live"), default="paper")
    reconcile_cmd.add_argument("--state-db-path", default=".stacksats/run_state.sqlite3")

    animate_cmd = strategy_sub.add_parser(
        "animate",
        help="Render a high-definition GIF from backtest_result.json",
        formatter_class=_HelpFormatter,
    )
    animate_cmd.add_argument(
        "--backtest-json",
        required=True,
        help="Path to backtest_result.json generated by strategy backtest.",
    )
    animate_cmd.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated GIF/manifest (default: parent of --backtest-json).",
    )
    animate_cmd.add_argument(
        "--output-name",
        default="strategy_vs_uniform_hd.gif",
        help="Output GIF filename.",
    )
    animate_cmd.add_argument("--fps", type=int, default=20)
    animate_cmd.add_argument("--width", type=int, default=1920)
    animate_cmd.add_argument("--height", type=int, default=1080)
    animate_cmd.add_argument("--max-frames", type=int, default=240)
    animate_cmd.add_argument(
        "--window-mode",
        choices=("rolling", "non-overlapping"),
        default="rolling",
    )

    serve_parser = root.add_parser(
        "serve",
        help="Hosted service commands",
        formatter_class=_HelpFormatter,
    )
    serve_sub = serve_parser.add_subparsers(dest="serve_command", required=True)
    agent_api_cmd = serve_sub.add_parser(
        "agent-api",
        help="Start the StackSats agent HTTP service",
        formatter_class=_HelpFormatter,
    )
    agent_api_cmd.add_argument("--host", default="127.0.0.1")
    agent_api_cmd.add_argument("--port", type=int, default=8000)
    agent_api_cmd.add_argument(
        "--registry-path",
        default=str(Path(".stacksats") / "agent_service_registry.json"),
    )
    agent_api_cmd.add_argument(
        "--state-db-path",
        default=str(Path(".stacksats") / "run_state.sqlite3"),
    )
    agent_api_cmd.add_argument("--output-dir", default="output")
    agent_api_cmd.add_argument(
        "--auth-token-env",
        default="STACKSATS_AGENT_API_TOKEN",
    )
    agent_api_cmd.add_argument(
        "--btc-price-col-default",
        default="price_usd",
    )

    demo_parser = root.add_parser(
        "demo",
        help="Offline demo workflows backed by packaged sample data",
        formatter_class=_HelpFormatter,
    )
    demo_sub = demo_parser.add_subparsers(dest="demo_command", required=True)
    _add_validate_command(
        demo_sub,
        help_text="Validate the packaged demo strategy and data",
        command_name="validate",
        strategy_required=False,
        default_strategy=DEMO_STRATEGY_SPEC,
        default_start=DEMO_START_DATE,
        default_end=DEMO_END_DATE,
        default_min_win_rate=0.0,
        default_strict=False,
    )
    _add_backtest_command(
        demo_sub,
        help_text="Backtest the packaged demo strategy and data",
        command_name="backtest",
        strategy_required=False,
        default_strategy=DEMO_STRATEGY_SPEC,
        default_start=DEMO_START_DATE,
        default_end=DEMO_END_DATE,
    )
    _add_export_command(
        demo_sub,
        help_text="Export demo strategy artifacts",
        command_name="export",
        strategy_required=False,
        default_strategy=DEMO_STRATEGY_SPEC,
        default_start=DEMO_START_DATE,
        default_end=DEMO_END_DATE,
    )

    data_parser = root.add_parser(
        "data",
        help="Explicit BRK data setup and diagnostics",
        formatter_class=_HelpFormatter,
    )
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    fetch_cmd = data_sub.add_parser("fetch", help="Download canonical BRK source data")
    fetch_cmd.add_argument("--manifest", default=None, help="Optional manifest override path.")
    fetch_cmd.add_argument("--target-dir", default=str(MANAGED_BRK_DIR))
    fetch_cmd.add_argument("--schema-dir", default=None)
    fetch_cmd.add_argument("--overwrite", action="store_true")

    prepare_cmd = data_sub.add_parser("prepare", help="Prepare runtime bitcoin_analytics.parquet")
    prepare_cmd.add_argument("--source", default=None)
    prepare_cmd.add_argument("--output", default=str(MANAGED_RUNTIME_PARQUET))
    prepare_cmd.add_argument("--overwrite", action="store_true")

    doctor_cmd = data_sub.add_parser("doctor", help="Inspect runtime data resolution and coverage")
    doctor_cmd.add_argument("--parquet-path", default=None)

    return parser


def _float_or_default(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _backtest_result_from_json(path: str | Path):
    from ..viz.animation_data import load_backtest_payload, spd_table_from_backtest_payload
    from ..api import BacktestResult

    payload = load_backtest_payload(path)
    spd_table = spd_table_from_backtest_payload(payload)

    summary_raw = payload.get("summary_metrics")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    provenance_raw = payload.get("provenance")
    provenance = provenance_raw if isinstance(provenance_raw, dict) else {}

    return BacktestResult(
        spd_table=spd_table,
        exp_decay_percentile=_float_or_default(summary.get("exp_decay_percentile")),
        win_rate=_float_or_default(summary.get("win_rate")),
        score=_float_or_default(summary.get("score")),
        uniform_exp_decay_percentile=_float_or_default(
            summary.get("uniform_exp_decay_percentile")
        ),
        strategy_id=str(provenance.get("strategy_id", "unknown")),
        strategy_version=str(provenance.get("version", "0.0.0")),
        config_hash=str(provenance.get("config_hash", "")),
        run_id=str(provenance.get("run_id", "")),
    )


def _run_lifecycle_command(command_name: str, args, runner: StrategyRunner) -> int:
    strategy = load_strategy(args.strategy, config_path=args.strategy_config)

    if command_name == "validate":
        result = runner.validate(
            strategy,
            ValidationConfig(
                start_date=args.start_date,
                end_date=args.end_date,
                min_win_rate=args.min_win_rate,
                strict=args.strict,
            ),
        )
        print(result.summary())
        for msg in result.messages:
            print(f"- {msg}")
        if not result.passed:
            raise SystemExit(1)
        return 0

    if command_name == "backtest":
        result = runner.backtest(
            strategy,
            BacktestConfig(
                start_date=args.start_date,
                end_date=args.end_date,
                strategy_label=args.strategy_label or strategy.strategy_id,
                output_dir=args.output_dir,
            ),
        )
        print(result.summary())
        output_root = (
            Path(args.output_dir)
            / result.strategy_id
            / result.strategy_version
            / result.run_id
        )
        output_root.mkdir(parents=True, exist_ok=True)
        result.plot(output_dir=str(output_root))
        output_path = output_root / "backtest_result.json"
        result.to_json(output_path)
        print(f"Saved: {output_root}")
        return 0

    if command_name == "export":
        batch = runner.export(
            strategy,
            ExportConfig(
                range_start=args.start_date,
                range_end=args.end_date,
                output_dir=args.output_dir,
            ),
        )
        output_root = (
            Path(args.output_dir)
            / strategy.strategy_id
            / strategy.version
            / batch.run_id
        )
        meta = {
            "rows": int(batch.row_count),
            "windows": int(batch.window_count),
            "strategy_id": strategy.strategy_id,
            "version": strategy.version,
            "schema_version": batch.schema_version,
            "output_dir": str(output_root),
        }
        print(json.dumps(meta, indent=2))
        print(f"Saved: {output_root}")
        return 0

    raise ValueError(f"Unsupported lifecycle command: {command_name}")


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    strategy_command = getattr(args, "strategy_command", None)
    demo_command = getattr(args, "demo_command", None)
    data_command = getattr(args, "data_command", None)
    serve_command = getattr(args, "serve_command", None)
    try:
        if command == "serve":
            if serve_command == "agent-api":
                try:
                    _start_agent_service_from_args(args)
                except ImportError as exc:
                    _exit_user_error(str(exc))
                return 0
            parser.error("Unsupported serve command.")

        if command == "data":
            if data_command == "fetch":
                manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
                parquet_path, schema_path = fetch_assets(
                    manifest_path=manifest,
                    target_dir=Path(args.target_dir),
                    schema_dir=Path(args.schema_dir) if args.schema_dir else None,
                    overwrite=args.overwrite,
                )
                print(
                    json.dumps(
                        {
                            "canonical_parquet": str(parquet_path),
                            "schema_path": str(schema_path),
                            "next": "stacksats data prepare",
                        },
                        indent=2,
                    )
                )
                return 0
            if data_command == "prepare":
                source = Path(args.source).expanduser() if args.source else latest_fetched_parquet()
                output = prepare_runtime_parquet(
                    source,
                    output=Path(args.output),
                    overwrite=args.overwrite,
                )
                print(
                    json.dumps(
                        {
                            "source": str(source.resolve()),
                            "runtime_parquet": str(output),
                            "next": "stacksats strategy backtest --strategy "
                            f"{DEMO_STRATEGY_SPEC}",
                        },
                        indent=2,
                    )
                )
                return 0
            if data_command == "doctor":
                print(json.dumps(data_doctor(args.parquet_path), indent=2))
                return 0
            parser.error("Unsupported data command.")

        if command == "demo":
            with packaged_demo_parquet_path() as demo_path:
                runner = StrategyRunner(BTCDataProvider(parquet_path=str(demo_path)))
                return _run_lifecycle_command(demo_command, args, runner)

        if strategy_command == "animate":
            result = _backtest_result_from_json(args.backtest_json)
            output_dir = (
                Path(args.output_dir).expanduser().resolve()
                if args.output_dir is not None
                else Path(args.backtest_json).expanduser().resolve().parent
            )
            paths = result.animate(
                output_dir=str(output_dir),
                fps=args.fps,
                width=args.width,
                height=args.height,
                max_frames=args.max_frames,
                filename=args.output_name,
                window_mode=args.window_mode,
                source_backtest_json=args.backtest_json,
            )
            print(json.dumps(paths, indent=2))
            print(f"Saved: {output_dir}")
            return 0

        if strategy_command in {"validate", "backtest", "export"}:
            return _run_lifecycle_command(strategy_command, args, StrategyRunner())

        if strategy_command == "compare":
            ordered = _ordered_compare_selectors(
                args.compare_baseline,
                args.compare_strategies,
            )
            strategies = [
                load_strategy(sel, config_path=args.strategy_config)
                for sel in ordered
            ]
            runner = StrategyRunner()
            result = runner.compare(
                strategies,
                ComparisonConfig(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    baseline=args.compare_baseline,
                    strict=args.compare_strict,
                    min_win_rate=args.min_win_rate,
                    output_dir=args.output_dir,
                ),
                selectors=ordered,
            )
            print(result.render_table())
            print(f"\nSaved: {result.artifact_path}")
            return 0

        strategy = load_strategy(args.strategy, config_path=args.strategy_config)
        runner = StrategyRunner()
        if strategy_command == "decide-daily":
            result = runner.decide_daily(
                strategy,
                DecideDailyConfig(
                    run_date=args.run_date,
                    total_window_budget_usd=args.total_window_budget_usd,
                    state_db_path=args.state_db_path,
                    output_dir=args.output_dir,
                    force=args.force,
                    btc_price_col=args.btc_price_col,
                ),
            )
            print(json.dumps(result.to_json(), indent=2))
            if result.status == "decided":
                print("Status: DECIDED")
                return 0
            if result.status == "noop":
                print("Status: NO-OP (idempotent)")
                return 0
            print("Status: FAILED")
            raise SystemExit(1)

        if strategy_command == "run-daily":
            result = runner.run_daily(
                strategy,
                RunDailyConfig(
                    run_date=args.run_date,
                    total_window_budget_usd=args.total_window_budget_usd,
                    mode=args.mode,
                    state_db_path=args.state_db_path,
                    output_dir=args.output_dir,
                    adapter_spec=args.adapter,
                    force=args.force,
                    btc_price_col=args.btc_price_col,
                ),
            )
            print(json.dumps(result.to_json(), indent=2))
            if result.status == "executed":
                print("Status: EXECUTED")
                return 0
            if result.status == "noop":
                print("Status: NO-OP (idempotent)")
                return 0
            print("Status: FAILED")
            raise SystemExit(1)

        if strategy_command == "reconcile-daily":
            result = runner.reconcile_daily_run(
                strategy,
                run_date=args.run_date,
                mode=args.mode,
                state_db_path=args.state_db_path,
            )
            print(json.dumps(result, indent=2))
            return 0

        parser.error("Unsupported command.")
    except JSONDecodeError as exc:
        _exit_user_error(
            "Invalid JSON in strategy config file.",
            hint=str(exc),
        )
    except (FileNotFoundError, FileExistsError) as exc:
        _exit_user_error(str(exc))
    except (ModuleNotFoundError, ImportError) as exc:
        _exit_user_error(
            f"Could not import strategy module: {exc}",
            hint="Check the module path and ensure dependencies are installed.",
        )
    except AttributeError as exc:
        _exit_user_error(str(exc))
    except DataLoadError as exc:
        _exit_user_error(str(exc))
    except ValueError as exc:
        _exit_user_error(str(exc))
    except Exception as exc:
        if _is_requests_exception(exc):
            _exit_user_error(
                "Failed to fetch required network data.",
                hint=str(exc),
            )
        raise


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
