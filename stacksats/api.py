"""Result types for strategy lifecycle operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WIN_RATE_TOLERANCE = 1e-10


@dataclass(slots=True)
class BacktestResult:
    """Structured backtest result."""

    spd_table: pd.DataFrame
    exp_decay_percentile: float
    win_rate: float
    score: float
    strategy_id: str = "unknown"
    strategy_version: str = "0.0.0"
    config_hash: str = ""
    run_id: str = ""

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
            "provenance": {
                "strategy_id": self.strategy_id,
                "version": self.strategy_version,
                "config_hash": self.config_hash,
                "run_id": self.run_id,
            },
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
        wins = int((excess_percentile > WIN_RATE_TOLERANCE).sum())
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
    min_win_rate: float = 50.0

    def summary(self) -> str:
        """Return a concise validation summary string."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | "
            f"Forward Leakage: {self.forward_leakage_ok} | "
            f"Weight Constraints: {self.weight_constraints_ok} | "
            f"Win Rate: {self.win_rate:.2f}% (>={self.min_win_rate:.2f}%: {self.win_rate_ok})"
        )
