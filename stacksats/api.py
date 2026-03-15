"""Result types for strategy lifecycle operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

WIN_RATE_TOLERANCE = 1e-10


@dataclass(slots=True)
class BacktestResult:
    """Structured backtest result."""

    spd_table: pl.DataFrame
    exp_decay_percentile: float
    win_rate: float
    score: float
    uniform_exp_decay_percentile: float = 0.0
    strategy_id: str = "unknown"
    strategy_version: str = "0.0.0"
    config_hash: str = ""
    run_id: str = ""

    @property
    def exp_decay_multiple_vs_uniform(self) -> float | None:
        """Return dynamic-vs-uniform exp-decay ratio, or None when undefined."""
        uniform_exp_decay = float(self.uniform_exp_decay_percentile)
        if not np.isfinite(uniform_exp_decay) or uniform_exp_decay <= 0.0:
            return None
        multiple = float(self.exp_decay_percentile) / uniform_exp_decay
        if not np.isfinite(multiple):
            return None
        return multiple

    def summary(self) -> str:
        """Return a concise text summary of key metrics."""
        exp_decay_multiple = self.exp_decay_multiple_vs_uniform
        exp_decay_multiple_str = (
            f"{exp_decay_multiple:.3f}x" if exp_decay_multiple is not None else "n/a"
        )
        return (
            f"Score: {self.score:.2f}% | "
            f"Win Rate: {self.win_rate:.2f}% | "
            f"Exp-Decay Percentile: {self.exp_decay_percentile:.2f}% | "
            f"Uniform Exp-Decay: {self.uniform_exp_decay_percentile:.2f}% | "
            f"Exp-Decay Multiple: {exp_decay_multiple_str} | "
            f"Windows: {self.spd_table.height}"
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Return the SPD table."""
        return self.spd_table.clone()

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
                "uniform_exp_decay_percentile": float(self.uniform_exp_decay_percentile),
                "exp_decay_multiple_vs_uniform": self.exp_decay_multiple_vs_uniform,
                "windows": self.spd_table.height,
            },
            "window_level_data": self.spd_table.to_dicts(),
        }
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def plot(self, output_dir: str = "output") -> dict[str, str]:
        """Export backtest metrics to JSON. Returns path to metrics file."""
        from .backtest import export_metrics_json

        dyn = self.spd_table["dynamic_percentile"]
        uni = self.spd_table["uniform_percentile"]
        excess_percentile = dyn - uni
        uniform_pct_safe = uni.replace(0, 0.01)
        relative_improvements = excess_percentile / uniform_pct_safe * 100
        wins = int((excess_percentile > WIN_RATE_TOLERANCE).sum())
        losses = self.spd_table.height - wins

        ratio = dyn / uni.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fill_null(0.0)

        metrics = {
            "score": float(self.score),
            "win_rate": float(self.win_rate),
            "exp_decay_percentile": float(self.exp_decay_percentile),
            "uniform_exp_decay_percentile": float(self.uniform_exp_decay_percentile),
            "exp_decay_multiple_vs_uniform": self.exp_decay_multiple_vs_uniform,
            "mean_excess": float(excess_percentile.mean()),
            "median_excess": float(excess_percentile.median()),
            "relative_improvement_pct_mean": float(relative_improvements.mean()),
            "relative_improvement_pct_median": float(relative_improvements.median()),
            "mean_ratio": float(ratio.mean()),
            "median_ratio": float(ratio.median()),
            "total_windows": self.spd_table.height,
            "wins": int(wins),
            "losses": int(losses),
        }

        path = export_metrics_json(self.spd_table, metrics, output_dir=output_dir)
        return {"metrics_json": path}

    def animate(
        self,
        output_dir: str = "output",
        *,
        fps: int = 20,
        width: int = 1920,
        height: int = 1080,
        max_frames: int = 240,
        filename: str = "strategy_vs_uniform_hd.gif",
        window_mode: str = "rolling",
        source_backtest_json: str | Path | None = None,
    ) -> dict[str, str]:
        """Render an animated strategy-vs-uniform GIF and write a manifest JSON."""
        from .animation_data import prepare_animation_frame_data
        from .animation_render import render_strategy_vs_uniform_gif

        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        frame_data = prepare_animation_frame_data(
            self.spd_table,
            window_mode=window_mode,
            max_frames=max_frames,
        )
        gif_path = output_root / filename
        render_meta = render_strategy_vs_uniform_gif(
            frame_data,
            gif_path,
            fps=fps,
            width=width,
            height=height,
        )

        source_json_path: str | None = None
        if source_backtest_json is not None:
            source_json_path = str(Path(source_backtest_json).expanduser().resolve())

        manifest_payload = {
            "frames": int(render_meta["frames"]),
            "fps": int(render_meta["fps"]),
            "width": int(render_meta["width"]),
            "height": int(render_meta["height"]),
            "window_mode": window_mode,
            "source_backtest_json": source_json_path,
            "output_gif": str(gif_path),
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "run_id": self.run_id,
        }
        manifest_path = output_root / "animation_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )
        return {
            "gif": str(gif_path),
            "manifest_json": str(manifest_path),
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
    diagnostics: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a concise validation summary string."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | "
            f"Forward Leakage: {self.forward_leakage_ok} | "
            f"Weight Constraints: {self.weight_constraints_ok} | "
            f"Win Rate: {self.win_rate:.2f}% (>={self.min_win_rate:.2f}%: {self.win_rate_ok})"
        )


@dataclass(frozen=True, slots=True)
class DailyOrderRequest:
    """Execution request for a single strategy daily order."""

    strategy_id: str
    strategy_version: str
    run_date: str
    mode: str
    weight_today: float
    notional_usd: float
    price_usd: float
    quantity_btc: float
    btc_price_col: str


@dataclass(frozen=True, slots=True)
class DailyOrderReceipt:
    """Execution receipt for a submitted daily order."""

    status: str
    external_order_id: str | None
    filled_notional_usd: float
    filled_quantity_btc: float
    fill_price_usd: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DailyRunResult:
    """Structured result for a daily strategy execution."""

    status: str
    strategy_id: str
    strategy_version: str
    run_date: str
    run_key: str
    mode: str
    idempotency_hit: bool
    forced_rerun: bool
    weight_today: float | None
    order_notional_usd: float | None
    btc_quantity: float | None
    price_usd: float | None
    adapter_name: str
    state_db_path: str
    artifact_path: str | None
    message: str
    order_receipt: DailyOrderReceipt | None = None
    bootstrap: bool = False
    validation_receipt_id: int | None = None
    validation_passed: bool | None = None
    data_hash: str = ""
    feature_snapshot_hash: str = ""

    def summary(self) -> str:
        """Return concise daily run status summary."""
        return (
            f"Daily Run {self.status.upper()} | Strategy: {self.strategy_id}@{self.strategy_version} | "
            f"Date: {self.run_date} | Mode: {self.mode} | Run Key: {self.run_key} | "
            f"Idempotency Hit: {self.idempotency_hit}"
        )

    def to_json(self, path: str | Path | None = None) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
