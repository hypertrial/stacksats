#!/usr/bin/env python3
"""Compare DuckDB alpha strategy against MVRV plus on shared horizon."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import warnings

import numpy as np

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, ValidationConfig


def _load_strategy(strategy_spec: str):
    module_name, class_name = strategy_spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    return strategy_cls()


def _mean_btc_per_million(spd_table, column: str) -> float:
    return float((spd_table[column] * 1_000_000.0 / 100_000_000.0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promotion gate for DuckDB alpha strategy."
    )
    parser.add_argument(
        "--baseline",
        default="stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy",
    )
    parser.add_argument(
        "--candidate",
        default="stacksats.strategies.model_duckdb_alpha:DuckDBAlphaStrategy",
    )
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2025-05-31")
    parser.add_argument("--baseline-score", type=float, default=None)
    parser.add_argument("--baseline-win-rate", type=float, default=None)
    parser.add_argument("--baseline-exp-decay-percentile", type=float, default=None)
    parser.add_argument(
        "--baseline-mean-dynamic-sats-per-dollar",
        type=float,
        default=None,
        help="Optional precomputed baseline mean dynamic_sats_per_dollar for quick loops.",
    )
    parser.add_argument("--score-lift", type=float, default=8.0)
    parser.add_argument("--win-rate-lift", type=float, default=5.0)
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument(
        "--skip-strict",
        action="store_true",
        help="Skip strict validation (useful for quick research loops).",
    )
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    if args.duckdb_path:
        os.environ["STACKSATS_ANALYTICS_DUCKDB"] = args.duckdb_path
    baseline_override_fields = (
        args.baseline_score,
        args.baseline_win_rate,
        args.baseline_exp_decay_percentile,
    )
    if any(value is not None for value in baseline_override_fields) and not all(
        value is not None for value in baseline_override_fields
    ):
        raise ValueError(
            "When overriding baseline metrics, provide all of: "
            "--baseline-score, --baseline-win-rate, --baseline-exp-decay-percentile."
        )
    use_baseline_override = all(value is not None for value in baseline_override_fields)

    runner = StrategyRunner()
    baseline_strategy = _load_strategy(args.baseline)
    candidate_strategy = _load_strategy(args.candidate)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in log",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in log1p",
            category=RuntimeWarning,
        )
        baseline_backtest = None
        if not use_baseline_override:
            baseline_backtest = runner.backtest(
                baseline_strategy,
                BacktestConfig(start_date=args.start_date, end_date=args.end_date),
            )
        candidate_backtest = runner.backtest(
            candidate_strategy,
            BacktestConfig(start_date=args.start_date, end_date=args.end_date),
        )
        if args.skip_strict:
            strict_validation = None
        else:
            strict_validation = runner.validate(
                candidate_strategy,
                ValidationConfig(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    strict=True,
                    max_permutation_pvalue=0.05,
                    min_bootstrap_ci_lower_excess=0.5,
                ),
            )

    if baseline_backtest is not None:
        baseline_score = float(baseline_backtest.score)
        baseline_win_rate = float(baseline_backtest.win_rate)
        baseline_exp_decay_percentile = float(baseline_backtest.exp_decay_percentile)
    else:
        baseline_score = float(args.baseline_score or 0.0)
        baseline_win_rate = float(args.baseline_win_rate or 0.0)
        baseline_exp_decay_percentile = float(args.baseline_exp_decay_percentile or 0.0)

    score_delta = float(candidate_backtest.score - baseline_score)
    win_rate_delta = float(candidate_backtest.win_rate - baseline_win_rate)
    if baseline_backtest is not None:
        baseline_btc_per_million = _mean_btc_per_million(
            baseline_backtest.spd_table,
            "dynamic_sats_per_dollar",
        )
    elif args.baseline_mean_dynamic_sats_per_dollar is not None:
        baseline_btc_per_million = float(
            args.baseline_mean_dynamic_sats_per_dollar * 1_000_000.0 / 100_000_000.0
        )
    else:
        baseline_btc_per_million = float("nan")
    candidate_btc_per_million = _mean_btc_per_million(
        candidate_backtest.spd_table,
        "dynamic_sats_per_dollar",
    )
    btc_lift_per_million = (
        float(candidate_btc_per_million - baseline_btc_per_million)
        if not np.isnan(baseline_btc_per_million)
        else float("nan")
    )

    strict_passed = bool(strict_validation.passed) if strict_validation is not None else True
    strict_messages = strict_validation.messages if strict_validation is not None else []
    strict_diagnostics = strict_validation.diagnostics if strict_validation is not None else {}
    passed = (
        score_delta >= float(args.score_lift)
        and win_rate_delta >= float(args.win_rate_lift)
        and strict_passed
    )
    summary = {
        "baseline": {
            "strategy": args.baseline,
            "score": baseline_score,
            "win_rate": baseline_win_rate,
            "exp_decay_percentile": baseline_exp_decay_percentile,
            "source": "computed" if baseline_backtest is not None else "override",
        },
        "candidate": {
            "strategy": args.candidate,
            "score": float(candidate_backtest.score),
            "win_rate": float(candidate_backtest.win_rate),
            "exp_decay_percentile": float(candidate_backtest.exp_decay_percentile),
        },
        "gates": {
            "required_score_lift": float(args.score_lift),
            "required_win_rate_lift": float(args.win_rate_lift),
            "score_delta": score_delta,
            "win_rate_delta": win_rate_delta,
            "strict_validation_passed": strict_passed,
            "strict_validation_skipped": bool(args.skip_strict),
            "passed": passed,
        },
        "btc_uplift": {
            "baseline_btc_per_1m_window": baseline_btc_per_million,
            "candidate_btc_per_1m_window": candidate_btc_per_million,
            "delta_btc_per_1m_window": btc_lift_per_million,
        },
        "strict_validation_messages": strict_messages,
        "strict_validation_diagnostics": strict_diagnostics,
    }

    print(json.dumps(summary, indent=2))
    if args.enforce and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
