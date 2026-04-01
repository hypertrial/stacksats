#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a Python-first custom strategy research loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.compare_strategies as compare_strategies
from stacksats.loader import load_strategy
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, ValidationConfig

DEFAULT_OUTPUT_PATH = ROOT / "output" / "research_strategy.json"


def _load_json_file(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def _load_column_map_config(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = _load_json_file(path)
    column_map: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("column-map config must map string library names to string columns.")
        column_map[key] = value
    return column_map


def _resolve_output_path(output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = ROOT / destination
    return destination


def _validate_research_inputs(
    *,
    dataframe_parquet: str | None,
    column_map_config: str | None,
) -> None:
    if column_map_config is not None and dataframe_parquet is None:
        raise ValueError("--column-map-config requires --dataframe-parquet.")


def _canonicalize_dataframe(df: pl.DataFrame, column_map: dict[str, str]) -> pl.DataFrame:
    inverse_map = {source: canonical for canonical, source in column_map.items() if source in df.columns}
    if not inverse_map:
        return df.clone()
    return df.rename(inverse_map)


def _validation_payload(result) -> dict[str, object]:
    return {
        "summary": result.summary(),
        "passed": bool(result.passed),
        "forward_leakage_ok": bool(result.forward_leakage_ok),
        "weight_constraints_ok": bool(result.weight_constraints_ok),
        "win_rate": float(result.win_rate),
        "win_rate_ok": bool(result.win_rate_ok),
        "min_win_rate": float(result.min_win_rate),
        "messages": list(result.messages),
        "diagnostics": dict(result.diagnostics or {}),
    }


def _backtest_payload(result) -> dict[str, object]:
    payload = result.to_json()
    return {
        "summary": result.summary(),
        "provenance": payload["provenance"],
        "summary_metrics": payload["summary_metrics"],
    }


def _run_primary_strategy(
    *,
    strategy_selector: str,
    strategy_config_path: str | None,
    validation_config: ValidationConfig,
    backtest_config: BacktestConfig,
    dataframe_parquet: str | None,
    column_map_config: str | None,
):
    strategy = load_strategy(strategy_selector, config_path=strategy_config_path)
    metadata = strategy.metadata()
    spec = strategy.spec()

    if dataframe_parquet is None:
        validation = strategy.validate(validation_config)
        backtest = strategy.backtest(backtest_config)
        return strategy, metadata, spec, validation, backtest, None

    source_df = pl.read_parquet(Path(dataframe_parquet).expanduser())
    column_map = _load_column_map_config(column_map_config)
    runner = StrategyRunner.from_dataframe(source_df, column_map=column_map or None)
    validation = runner.validate(strategy, validation_config)
    backtest = runner.backtest(strategy, backtest_config)
    canonical_df = _canonicalize_dataframe(source_df, column_map)
    return strategy, metadata, spec, validation, backtest, canonical_df


def build_research_payload(
    *,
    strategy_selector: str,
    strategy_config_path: str | None,
    dataframe_parquet: str | None,
    column_map_config: str | None,
    start_date: str,
    end_date: str,
    min_win_rate: float,
    strict: bool,
    baseline: str,
    compare_strategies_selectors: list[str],
) -> dict[str, object]:
    _validate_research_inputs(
        dataframe_parquet=dataframe_parquet,
        column_map_config=column_map_config,
    )
    validation_config = ValidationConfig(
        start_date=start_date,
        end_date=end_date,
        min_win_rate=min_win_rate,
        strict=strict,
    )
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        strategy_label=strategy_selector,
    )

    (
        _strategy,
        metadata,
        spec,
        validation,
        backtest,
        canonical_btc_df,
    ) = _run_primary_strategy(
        strategy_selector=strategy_selector,
        strategy_config_path=strategy_config_path,
        validation_config=validation_config,
        backtest_config=backtest_config,
        dataframe_parquet=dataframe_parquet,
        column_map_config=column_map_config,
    )

    comparison_payload = None
    if compare_strategies_selectors:
        selector_config_paths = (
            {strategy_selector: strategy_config_path} if strategy_config_path is not None else None
        )
        comparison_payload = compare_strategies.compare_strategies(
            selectors=[strategy_selector, *compare_strategies_selectors],
            baseline=baseline,
            start_date=start_date,
            end_date=end_date,
            strict=strict,
            min_win_rate=min_win_rate,
            output_path=None,
            btc_df=canonical_btc_df,
            selector_config_paths=selector_config_paths,
        )

    return {
        "strategy_selector": strategy_selector,
        "resolved_strategy_id": metadata.strategy_id,
        "strategy_version": metadata.version,
        "intent_mode": spec.intent_mode,
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "strict": strict,
            "min_win_rate": min_win_rate,
        },
        "data_source": {
            "mode": "dataframe-parquet" if dataframe_parquet is not None else "canonical-runtime",
            "dataframe_parquet": dataframe_parquet,
            "column_map_config": column_map_config,
        },
        "validation": _validation_payload(validation),
        "backtest": _backtest_payload(backtest),
        "comparison": comparison_payload,
    }


def _print_summary(payload: dict[str, object]) -> None:
    print(
        f"Strategy: {payload['resolved_strategy_id']} "
        f"({payload['intent_mode']}) | "
        f"Window: {payload['window']['start_date']} -> {payload['window']['end_date']} | "
        f"Data: {payload['data_source']['mode']}"
    )
    print(payload["validation"]["summary"])
    print(payload["backtest"]["summary"])
    comparison = payload.get("comparison")
    if isinstance(comparison, dict):
        print()
        print(compare_strategies._render_table(comparison["rows"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Python-first research loop.")
    parser.add_argument("--strategy", required=True, help="Built-in strategy_id or module_or_path:ClassName")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--strategy-config")
    parser.add_argument("--dataframe-parquet")
    parser.add_argument("--column-map-config")
    parser.add_argument("--min-win-rate", type=float, default=50.0)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict validation for the research loop.",
    )
    parser.add_argument("--baseline", default="uniform")
    parser.add_argument(
        "--compare-strategy",
        action="append",
        default=[],
        dest="compare_strategies_selectors",
        help="Additional selectors to compare on the same fixed window.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT)),
        help="JSON output path, relative to the repo root by default.",
    )
    args = parser.parse_args()

    payload = build_research_payload(
        strategy_selector=args.strategy,
        strategy_config_path=args.strategy_config,
        dataframe_parquet=args.dataframe_parquet,
        column_map_config=args.column_map_config,
        start_date=args.start_date,
        end_date=args.end_date,
        min_win_rate=args.min_win_rate,
        strict=bool(args.strict),
        baseline=args.baseline,
        compare_strategies_selectors=args.compare_strategies_selectors,
    )

    destination = _resolve_output_path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(payload)
    print(f"\nSaved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
