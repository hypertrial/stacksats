"""Strategy-centric CLI for StackSats."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import requests

from .data_btc import DataLoadError
from .loader import load_strategy
from .runner import StrategyRunner
from .strategy_types import BacktestConfig, ExportConfig, RunDailyConfig, ValidationConfig


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults and preserve example formatting."""


def _exit_user_error(message: str, *, hint: str | None = None, code: int = 2) -> None:
    print(f"Error: {message}", file=sys.stderr)
    if hint:
        print(f"Hint: {hint}", file=sys.stderr)
    raise SystemExit(code)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stacksats",
        description="StackSats CLI",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  stacksats strategy validate --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy\n"
            "  stacksats strategy backtest --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy --output-dir output\n"
            "  stacksats strategy export --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy "
            "--start-date 2025-12-01 --end-date 2027-12-31\n"
            "  stacksats strategy run-daily --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy "
            "--total-window-budget-usd 1000\n"
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
            "Strategy spec format:\n"
            "  module_or_path:ClassName\n\n"
            "Example:\n"
            "  stacksats.strategies.examples:SimpleZScoreStrategy"
        ),
    )
    strategy_sub = strategy_parser.add_subparsers(dest="strategy_command", required=True)

    validate_cmd = strategy_sub.add_parser(
        "validate",
        help="Validate strategy",
        formatter_class=_HelpFormatter,
        epilog=(
            "Example:\n"
            "  stacksats strategy validate --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy --strict"
        ),
    )
    validate_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    validate_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )
    validate_cmd.add_argument("--start-date", default=None)
    validate_cmd.add_argument("--end-date", default=None)
    validate_cmd.add_argument("--min-win-rate", type=float, default=50.0)
    validate_strict_group = validate_cmd.add_mutually_exclusive_group()
    validate_strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Run additional robustness gates (enabled by default).",
    )
    validate_strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict robustness gates.",
    )

    backtest_cmd = strategy_sub.add_parser(
        "backtest",
        help="Backtest strategy",
        formatter_class=_HelpFormatter,
        epilog=(
            "Example:\n"
            "  stacksats strategy backtest --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy "
            "--start-date 2020-01-01 --end-date 2025-01-01"
        ),
    )
    backtest_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    backtest_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )
    backtest_cmd.add_argument("--start-date", default=None)
    backtest_cmd.add_argument("--end-date", default=None)
    backtest_cmd.add_argument("--output-dir", default="output")
    backtest_cmd.add_argument("--strategy-label", default=None)

    export_cmd = strategy_sub.add_parser(
        "export",
        help="Export strategy artifacts",
        formatter_class=_HelpFormatter,
        epilog=(
            "Example:\n"
            "  stacksats strategy export --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy "
            "--start-date 2025-12-01 --end-date 2027-12-31"
        ),
    )
    export_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    export_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )
    export_cmd.add_argument("--start-date", required=True)
    export_cmd.add_argument("--end-date", required=True)
    export_cmd.add_argument("--output-dir", default="output")

    run_daily_cmd = strategy_sub.add_parser(
        "run-daily",
        help="Run idempotent daily execution",
        formatter_class=_HelpFormatter,
        epilog=(
            "Example:\n"
            "  stacksats strategy run-daily --strategy "
            "stacksats.strategies.examples:SimpleZScoreStrategy "
            "--total-window-budget-usd 1000 --mode paper"
        ),
    )
    run_daily_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    run_daily_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )
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
    reconcile_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    reconcile_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for strategy parameters",
    )
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
    from .animation_data import load_backtest_payload, spd_table_from_backtest_payload
    from .api import BacktestResult

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    strategy_command = getattr(args, "strategy_command", None)
    try:
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
            return

        strategy = load_strategy(args.strategy, config_path=args.strategy_config)
        runner = StrategyRunner()
        if strategy_command == "validate":
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
            return

        if strategy_command == "backtest":
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
            return

        if strategy_command == "export":
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
            return

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
                return
            if result.status == "noop":
                print("Status: NO-OP (idempotent)")
                return
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
            return

        parser.error("Unsupported command.")
    except JSONDecodeError as exc:
        _exit_user_error(
            "Invalid JSON in strategy config file.",
            hint=str(exc),
        )
    except FileNotFoundError as exc:
        _exit_user_error(str(exc))
    except (ModuleNotFoundError, ImportError) as exc:
        _exit_user_error(
            f"Could not import strategy module: {exc}",
            hint="Check the module path and ensure dependencies are installed.",
        )
    except AttributeError as exc:
        _exit_user_error(str(exc))
    except requests.RequestException as exc:
        _exit_user_error(
            "Failed to fetch required network data.",
            hint=str(exc),
        )
    except DataLoadError as exc:
        _exit_user_error(str(exc))
    except ValueError as exc:
        _exit_user_error(str(exc))


if __name__ == "__main__":
    main()
