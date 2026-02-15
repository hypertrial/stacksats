"""Strategy-centric CLI for StackSats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import load_strategy
from .runner import StrategyRunner
from .strategy_types import BacktestConfig, ExportConfig, ValidationConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stacksats", description="StackSats CLI")
    root = parser.add_subparsers(dest="command", required=True)

    strategy_parser = root.add_parser("strategy", help="Strategy lifecycle commands")
    strategy_sub = strategy_parser.add_subparsers(dest="strategy_command", required=True)

    validate_cmd = strategy_sub.add_parser("validate", help="Validate strategy")
    validate_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    validate_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )
    validate_cmd.add_argument("--start-date", default=None)
    validate_cmd.add_argument("--end-date", default=None)
    validate_cmd.add_argument("--min-win-rate", type=float, default=50.0)
    validate_cmd.add_argument(
        "--strict",
        action="store_true",
        help="Run additional robustness gates (determinism, mutation, leakage, OOS, placebo).",
    )

    backtest_cmd = strategy_sub.add_parser("backtest", help="Backtest strategy")
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

    export_cmd = strategy_sub.add_parser("export", help="Export strategy artifacts")
    export_cmd.add_argument("--strategy", required=True, help="module_or_path:ClassName")
    export_cmd.add_argument(
        "--strategy-config",
        default=None,
        help="JSON config path for feature/signal/daily-intent parameters",
    )
    export_cmd.add_argument("--start-date", default="2025-12-01")
    export_cmd.add_argument("--end-date", default="2027-12-31")
    export_cmd.add_argument("--output-dir", default="output")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    strategy = load_strategy(args.strategy, config_path=args.strategy_config)
    runner = StrategyRunner()
    if args.strategy_command == "validate":
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

    if args.strategy_command == "backtest":
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

    if args.strategy_command == "export":
        batch = runner.export(
            strategy,
            ExportConfig(
                range_start=args.start_date,
                range_end=args.end_date,
                output_dir=args.output_dir,
            ),
        )
        meta = {
            "rows": int(batch.row_count),
            "windows": int(batch.window_count),
            "strategy_id": strategy.strategy_id,
            "version": strategy.version,
            "schema_version": batch.schema_version,
            "output_dir": args.output_dir,
        }
        print(json.dumps(meta, indent=2))
        return

    parser.error("Unsupported command.")


if __name__ == "__main__":
    main()
