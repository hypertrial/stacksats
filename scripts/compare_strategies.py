#!/usr/bin/env python3
# ruff: noqa: E402
"""Compare built-in and custom strategies on a shared validation/backtest window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stacksats.loader import load_strategy
from stacksats.strategies.catalog import (
    backtest_config_for_strategy,
    find_strategy_catalog_entry,
    validation_config_for_strategy,
)
from stacksats.strategy_types import BacktestConfig, ValidationConfig

DEFAULT_OUTPUT_PATH = ROOT / "output" / "strategy_compare.json"


def _dedupe_selectors(baseline: str, selectors: list[str]) -> list[str]:
    ordered = [baseline, *selectors]
    seen: set[str] = set()
    unique: list[str] = []
    for selector in ordered:
        if selector in seen:
            continue
        seen.add(selector)
        unique.append(selector)
    return unique


def _resolve_bounds(
    selectors: list[str],
    *,
    start_date: str | None,
    end_date: str | None,
    strict: bool | None,
    min_win_rate: float | None,
) -> tuple[str, str, bool, float]:
    if (start_date is None) != (end_date is None):
        raise ValueError("Pass both --start-date and --end-date together.")

    catalog_entries = [find_strategy_catalog_entry(selector) for selector in selectors]
    if any(entry is None for entry in catalog_entries):
        if start_date is None or end_date is None:
            raise ValueError(
                "Custom strategy comparisons require explicit --start-date and --end-date."
            )
        resolved_strict = True if strict is None else strict
        resolved_min_win_rate = 50.0 if min_win_rate is None else min_win_rate
        return start_date, end_date, resolved_strict, resolved_min_win_rate

    typed_entries = [entry for entry in catalog_entries if entry is not None]
    if start_date is None:
        start_date = max(
            str(backtest_config_for_strategy(entry.strategy_id).start_date)
            for entry in typed_entries
        )
    if end_date is None:
        end_date = min(
            str(backtest_config_for_strategy(entry.strategy_id).end_date)
            for entry in typed_entries
        )
    if strict is None:
        strict = any(
            validation_config_for_strategy(entry.strategy_id).strict for entry in typed_entries
        )
    if min_win_rate is None:
        min_win_rate = max(
            validation_config_for_strategy(entry.strategy_id).min_win_rate
            for entry in typed_entries
        )
    return start_date, end_date, strict, min_win_rate


def _build_validation_config(
    *,
    start_date: str,
    end_date: str,
    strict: bool,
    min_win_rate: float,
) -> ValidationConfig:
    return ValidationConfig(
        start_date=start_date,
        end_date=end_date,
        strict=strict,
        min_win_rate=min_win_rate,
    )


def _build_backtest_config(
    *,
    start_date: str,
    end_date: str,
    strategy_label: str,
) -> BacktestConfig:
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        strategy_label=strategy_label,
    )


def compare_strategies(
    *,
    selectors: list[str],
    baseline: str = "uniform",
    start_date: str | None = None,
    end_date: str | None = None,
    strict: bool | None = None,
    min_win_rate: float | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    """Run validate+backtest for a set of strategies on a shared comparison window."""
    if not selectors:
        raise ValueError("Provide at least one --strategy selector.")

    ordered_selectors = _dedupe_selectors(baseline, selectors)
    (
        resolved_start_date,
        resolved_end_date,
        resolved_strict,
        resolved_min_win_rate,
    ) = _resolve_bounds(
        ordered_selectors,
        start_date=start_date,
        end_date=end_date,
        strict=strict,
        min_win_rate=min_win_rate,
    )

    rows: list[dict[str, object]] = []
    baseline_score: float | None = None
    baseline_exp_decay: float | None = None
    for selector in ordered_selectors:
        strategy = load_strategy(selector)
        metadata = strategy.metadata()
        catalog_entry = find_strategy_catalog_entry(metadata.strategy_id)
        validation = strategy.validate(
            _build_validation_config(
                start_date=resolved_start_date,
                end_date=resolved_end_date,
                strict=resolved_strict,
                min_win_rate=resolved_min_win_rate,
            )
        )
        backtest = strategy.backtest(
            _build_backtest_config(
                start_date=resolved_start_date,
                end_date=resolved_end_date,
                strategy_label=metadata.strategy_id,
            )
        )
        if selector == baseline:
            baseline_score = float(backtest.score)
            baseline_exp_decay = float(backtest.exp_decay_percentile)

        diagnostics = dict(validation.diagnostics or {})
        judgment = diagnostics.get("judgment")
        if not isinstance(judgment, str) or not judgment:
            judgment = "validation-passed" if validation.passed else "validation-failed"
        rows.append(
            {
                "selector": selector,
                "resolved_strategy_id": metadata.strategy_id,
                "intent_mode": strategy.spec().intent_mode,
                "tier": catalog_entry.tier if catalog_entry is not None else None,
                "promotion_stage": (
                    catalog_entry.promotion_stage if catalog_entry is not None else None
                ),
                "validation_passed": bool(validation.passed),
                "win_rate": float(backtest.win_rate),
                "score": float(backtest.score),
                "exp_decay_percentile": float(backtest.exp_decay_percentile),
                "multiple_vs_uniform": backtest.exp_decay_multiple_vs_uniform,
                "judgment_label": judgment,
                "is_baseline": selector == baseline,
            }
        )

    for row in rows:
        score = float(row["score"])
        exp_decay = float(row["exp_decay_percentile"])
        row["score_delta_vs_baseline"] = (
            None if baseline_score is None else score - baseline_score
        )
        row["exp_decay_delta_vs_baseline"] = (
            None if baseline_exp_decay is None else exp_decay - baseline_exp_decay
        )

    payload = {
        "baseline_selector": baseline,
        "comparison_window": {
            "start_date": resolved_start_date,
            "end_date": resolved_end_date,
            "strict": resolved_strict,
            "min_win_rate": resolved_min_win_rate,
        },
        "rows": rows,
    }

    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _format_metric(value: object, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.2f}{suffix}"
    return str(value)


def _render_table(rows: list[dict[str, object]]) -> str:
    headers = (
        "Selector",
        "Intent",
        "Tier",
        "Win Rate",
        "Score",
        "Exp-Decay",
        "Vs Uniform",
        "Judgment",
    )
    rendered_rows = [
        (
            str(row["selector"]),
            str(row["intent_mode"]),
            str(row["tier"] or "custom"),
            _format_metric(row["win_rate"], suffix="%"),
            _format_metric(row["score"], suffix="%"),
            _format_metric(row["exp_decay_percentile"], suffix="%"),
            _format_metric(row["multiple_vs_uniform"], suffix="x"),
            str(row["judgment_label"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(item[index]) for item in rendered_rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for row in rendered_rows:
        lines.append(
            "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare strategies on a shared window.")
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        dest="strategies",
        help="Built-in strategy_id or module_or_path:ClassName",
    )
    parser.add_argument(
        "--baseline",
        default="uniform",
        help="Baseline selector to include first in the comparison table.",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--min-win-rate", type=float)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override strict validation for the shared comparison run.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT)),
        help="JSON output path, relative to the repo root by default.",
    )
    args = parser.parse_args()

    payload = compare_strategies(
        selectors=args.strategies,
        baseline=args.baseline,
        start_date=args.start_date,
        end_date=args.end_date,
        strict=args.strict,
        min_win_rate=args.min_win_rate,
        output_path=args.output_path,
    )
    print(_render_table(payload["rows"]))
    print(f"\nSaved {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
