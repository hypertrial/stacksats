"""User-facing API for custom strategy backtesting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model_development import precompute_features
from .prelude import BACKTEST_START, WINDOW_OFFSET, backtest_dynamic_dca, load_data
from .strategies.base import WindowStrategy
from .strategies.mvrv import MVRVStrategy


@dataclass(slots=True)
class BacktestResult:
    """Structured backtest result."""

    spd_table: pd.DataFrame
    exp_decay_percentile: float
    win_rate: float
    score: float

    def summary(self) -> str:
        """Return a concise text summary of key metrics."""
        return (
            f"Score: {self.score:.2f}% | "
            f"Win Rate: {self.win_rate:.2f}% | "
            f"Exp-Decay Percentile: {self.exp_decay_percentile:.2f}% | "
            f"Windows: {len(self.spd_table)}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the SPD table."""
        return self.spd_table.copy()

    def to_json(self, path: str | Path | None = None) -> dict:
        """Serialize result to a JSON-compatible dictionary."""
        payload = {
            "summary_metrics": {
                "score": float(self.score),
                "win_rate": float(self.win_rate),
                "exp_decay_percentile": float(self.exp_decay_percentile),
                "windows": int(len(self.spd_table)),
            },
            "window_level_data": self.spd_table.reset_index().to_dict(orient="records"),
        }
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def plot(self, output_dir: str = "output") -> dict[str, str]:
        """Generate standard backtest plots and return output paths."""
        from .backtest import (
            create_cumulative_performance,
            create_excess_percentile_distribution,
            create_performance_comparison_chart,
            create_performance_metrics_summary,
            create_win_loss_comparison,
            export_metrics_json,
        )

        excess_percentile = (
            self.spd_table["dynamic_percentile"] - self.spd_table["uniform_percentile"]
        )
        uniform_pct_safe = self.spd_table["uniform_percentile"].replace(0, 0.01)
        relative_improvements = excess_percentile / uniform_pct_safe * 100
        wins = (self.spd_table["dynamic_percentile"] > self.spd_table["uniform_percentile"]).sum()
        losses = len(self.spd_table) - wins

        metrics = {
            "score": float(self.score),
            "win_rate": float(self.win_rate),
            "exp_decay_percentile": float(self.exp_decay_percentile),
            "mean_excess": float(excess_percentile.mean()),
            "median_excess": float(excess_percentile.median()),
            "relative_improvement_pct_mean": float(relative_improvements.mean()),
            "relative_improvement_pct_median": float(relative_improvements.median()),
            "mean_ratio": float(
                (
                    self.spd_table["dynamic_percentile"]
                    / self.spd_table["uniform_percentile"].replace(0, np.nan)
                )
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .mean()
            ),
            "median_ratio": float(
                (
                    self.spd_table["dynamic_percentile"]
                    / self.spd_table["uniform_percentile"].replace(0, np.nan)
                )
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .median()
            ),
            "total_windows": int(len(self.spd_table)),
            "wins": int(wins),
            "losses": int(losses),
        }

        create_performance_comparison_chart(self.spd_table, output_dir=output_dir)
        create_excess_percentile_distribution(self.spd_table, output_dir=output_dir)
        create_win_loss_comparison(self.spd_table, output_dir=output_dir)
        create_cumulative_performance(self.spd_table, output_dir=output_dir)
        create_performance_metrics_summary(self.spd_table, metrics, output_dir=output_dir)
        export_metrics_json(self.spd_table, metrics, output_dir=output_dir)

        return {
            "performance_comparison": str(Path(output_dir) / "performance_comparison.svg"),
            "excess_distribution": str(Path(output_dir) / "excess_percentile_distribution.svg"),
            "win_loss": str(Path(output_dir) / "win_loss_comparison.svg"),
            "cumulative_performance": str(Path(output_dir) / "cumulative_performance.svg"),
            "metrics_summary": str(Path(output_dir) / "metrics_summary.svg"),
            "metrics_json": str(Path(output_dir) / "metrics.json"),
        }


@dataclass(slots=True)
class ValidationResult:
    """Structured validation result for a strategy."""

    passed: bool
    forward_leakage_ok: bool
    weight_constraints_ok: bool
    win_rate: float
    win_rate_ok: bool
    messages: list[str]

    def summary(self) -> str:
        """Return a concise validation summary string."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | "
            f"Forward Leakage: {self.forward_leakage_ok} | "
            f"Weight Constraints: {self.weight_constraints_ok} | "
            f"Win Rate: {self.win_rate:.2f}% (>=50%: {self.win_rate_ok})"
        )


def run_backtest(
    strategy: WindowStrategy,
    *,
    btc_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_label: str = "custom-strategy",
) -> BacktestResult:
    """Run a rolling-window backtest for any `WindowStrategy`.

    The strategy is responsible only for weight generation.
    Data loading, feature precomputation, and SPD scoring are handled here.
    """
    if btc_df is None:
        btc_df = load_data()

    features_df = precompute_features(btc_df)

    def _strategy_fn(df_window: pd.DataFrame) -> pd.Series:
        if df_window.empty:
            return pd.Series(dtype=float)

        window_start = df_window.index.min()
        window_end = df_window.index.max()
        return strategy.compute_weights(
            features_df=features_df,
            start_date=window_start,
            end_date=window_end,
            current_date=window_end,
        )

    spd_table, exp_decay_percentile = backtest_dynamic_dca(
        btc_df,
        _strategy_fn,
        features_df=features_df,
        strategy_label=strategy_label,
        start_date=start_date,
        end_date=end_date,
    )

    win_rate = (
        (spd_table["dynamic_percentile"] > spd_table["uniform_percentile"]).mean()
        * 100
    )
    score = (0.5 * win_rate) + (0.5 * exp_decay_percentile)

    return BacktestResult(
        spd_table=spd_table,
        exp_decay_percentile=exp_decay_percentile,
        win_rate=win_rate,
        score=score,
    )


def validate_strategy(
    strategy: WindowStrategy,
    *,
    btc_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_win_rate: float = 50.0,
) -> ValidationResult:
    """Validate a strategy for leakage, weight constraints, and performance."""
    if btc_df is None:
        btc_df = load_data()

    if start_date is None:
        # Default validation window should match the project backtest regime
        # instead of very early sparse-history periods.
        start_date = BACKTEST_START
    if end_date is None:
        end_date = btc_df.index.max().strftime("%Y-%m-%d")

    features_df = precompute_features(btc_df)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    backtest_idx = btc_df.loc[start_ts:end_ts].index

    messages: list[str] = []
    forward_leakage_ok = True
    weight_constraints_ok = True

    if len(backtest_idx) == 0:
        return ValidationResult(
            passed=False,
            forward_leakage_ok=False,
            weight_constraints_ok=False,
            win_rate=0.0,
            win_rate_ok=False,
            messages=["No data available in the requested date range."],
        )

    probe_step = max(len(backtest_idx) // 50, 1)
    for probe in backtest_idx[::probe_step]:
        window_start = max(start_ts, probe - WINDOW_OFFSET)
        if window_start > probe:
            continue

        full_weights = strategy.compute_weights(
            features_df=features_df,
            start_date=window_start,
            end_date=probe,
            current_date=probe,
        )

        masked_features = features_df.copy()
        masked_features.loc[masked_features.index > probe, :] = np.nan
        masked_weights = strategy.compute_weights(
            features_df=masked_features,
            start_date=window_start,
            end_date=probe,
            current_date=probe,
        )

        if probe in full_weights.index and probe in masked_weights.index:
            if not np.isclose(
                float(full_weights.loc[probe]),
                float(masked_weights.loc[probe]),
                rtol=1e-9,
                atol=1e-12,
            ):
                forward_leakage_ok = False
                messages.append(
                    f"Forward leakage detected near {probe.strftime('%Y-%m-%d')}."
                )
                break

    max_window_start = end_ts - WINDOW_OFFSET
    if start_ts <= max_window_start:
        window_starts = pd.date_range(start=start_ts, end=max_window_start, freq="D")
        for window_start in window_starts:
            window_end = window_start + WINDOW_OFFSET
            weights = strategy.compute_weights(
                features_df=features_df,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
            )
            if weights.empty:
                continue
            if bool((weights < 0).any()):
                weight_constraints_ok = False
                messages.append(
                    f"Negative weights found in window {window_start.date()} -> {window_end.date()}."
                )
                break
            weight_sum = float(weights.sum())
            if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
                weight_constraints_ok = False
                messages.append(
                    f"Weights do not sum to 1.0 for window {window_start.date()} -> {window_end.date()} "
                    f"(sum={weight_sum:.10f})."
                )
                break

    backtest_result = run_backtest(
        strategy,
        btc_df=btc_df,
        start_date=start_date,
        end_date=end_date,
        strategy_label="validation-run",
    )
    win_rate_ok = backtest_result.win_rate >= min_win_rate
    if not win_rate_ok:
        messages.append(
            f"Win rate below threshold: {backtest_result.win_rate:.2f}% < {min_win_rate:.2f}%."
        )

    if not messages:
        messages.append("All validation checks passed.")

    passed = forward_leakage_ok and weight_constraints_ok and win_rate_ok
    return ValidationResult(
        passed=passed,
        forward_leakage_ok=forward_leakage_ok,
        weight_constraints_ok=weight_constraints_ok,
        win_rate=float(backtest_result.win_rate),
        win_rate_ok=win_rate_ok,
        messages=messages,
    )


def validate_strategy_cli() -> None:
    """CLI entrypoint for validating a StackSats strategy."""
    parser = argparse.ArgumentParser(description="Validate a StackSats strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--min-win-rate", type=float, default=50.0, help="Minimum win rate percent")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Custom strategy spec in 'module_or_path:ClassName' format",
    )
    args = parser.parse_args()

    if args.strategy:
        from .loader import load_strategy

        strategy = load_strategy(args.strategy)
    else:
        strategy = MVRVStrategy()

    result = validate_strategy(
        strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        min_win_rate=args.min_win_rate,
    )
    print(result.summary())
    for message in result.messages:
        print(f"- {message}")
    if not result.passed:
        raise SystemExit(1)
