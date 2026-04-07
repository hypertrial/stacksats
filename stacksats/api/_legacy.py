"""Result types for strategy lifecycle operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .._contract import PUBLIC_ARTIFACT_SCHEMA_VERSION

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
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
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
        from ..backtest import export_metrics_json

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
        from ..viz.animation_data import prepare_animation_frame_data
        from ..viz.animation_render import render_strategy_vs_uniform_gif

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
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
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
    strategy_id: str = ""
    min_win_rate: float = 50.0
    diagnostics: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a concise validation summary string."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | "
            f"No Forward Leakage: {self.forward_leakage_ok} | "
            f"Weight Constraints: {self.weight_constraints_ok} | "
            f"Win Rate: {self.win_rate:.2f}% (>={self.min_win_rate:.2f}%: {self.win_rate_ok})"
        )


@dataclass(slots=True)
class DailyDecisionResult:
    """Structured result for a daily agent-facing decision."""

    status: str
    strategy_id: str
    strategy_version: str
    run_date: str
    decision_key: str
    idempotency_hit: bool
    forced_rerun: bool
    weight_today: float | None
    recommended_notional_usd: float | None
    recommended_quantity_btc: float | None
    reference_price_usd: float | None
    btc_price_col: str
    state_db_path: str
    artifact_path: str | None
    message: str
    validation_receipt_id: int | None = None
    validation_passed: bool | None = None
    data_hash: str = ""
    feature_snapshot_hash: str = ""
    bootstrap: bool = False

    def summary(self) -> str:
        """Return concise daily decision status summary."""
        return (
            f"Daily Decision {self.status.upper()} | "
            f"Strategy: {self.strategy_id}@{self.strategy_version} | "
            f"Date: {self.run_date} | "
            f"Decision Key: {self.decision_key} | "
            f"Idempotency Hit: {self.idempotency_hit}"
        )

    def to_json(self, path: str | Path | None = None) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["schema_version"] = PUBLIC_ARTIFACT_SCHEMA_VERSION
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionReceiptEvent:
    """Append-only execution receipt event reported by an external agent."""

    decision_key: str
    event_id: str
    event_type: str
    event_time: str
    broker_name: str | None = None
    broker_account_ref: str | None = None
    external_order_id: str | None = None
    filled_notional_usd: float | None = None
    filled_quantity_btc: float | None = None
    fill_price_usd: float | None = None
    message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["schema_version"] = PUBLIC_ARTIFACT_SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class ExecutionStatusResult:
    """Current execution lifecycle and reconciliation summary for a decision."""

    decision_key: str
    strategy_id: str
    run_date: str
    decision_status: str
    execution_status: str
    reconciliation_status: str
    recommended_notional_usd: float | None
    recommended_quantity_btc: float | None
    filled_notional_usd: float
    filled_quantity_btc: float
    average_fill_price_usd: float | None
    receipt_count: int
    latest_event_type: str | None
    latest_event_time: str | None
    message: str

    def summary(self) -> str:
        """Return concise execution lifecycle summary."""
        return (
            f"Execution {self.execution_status.upper()} | "
            f"Reconciliation: {self.reconciliation_status.upper()} | "
            f"Decision Key: {self.decision_key} | Receipts: {self.receipt_count}"
        )

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["schema_version"] = PUBLIC_ARTIFACT_SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class ExecutionReceiptHistoryResult:
    """Stored append-only execution receipt history for a decision."""

    decision_key: str
    receipts: list[ExecutionReceiptEvent]

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
            "decision_key": self.decision_key,
            "receipts": [receipt.to_json() for receipt in self.receipts],
        }


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


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """Per-strategy metrics within a comparison run."""

    selector: str
    strategy_id: str
    strategy_version: str
    intent_mode: str
    tier: str | None
    promotion_stage: str | None
    validation_passed: bool
    judgment_label: str
    win_rate: float
    score: float
    exp_decay_percentile: float
    multiple_vs_uniform: float | None
    score_delta_vs_baseline: float | None
    exp_decay_delta_vs_baseline: float | None
    is_baseline: bool


@dataclass(slots=True)
class ComparisonResult:
    """Structured result of a multi-strategy comparison."""

    baseline_selector: str
    comparison_window: dict[str, object]
    rows: list[ComparisonRow]
    run_id: str = ""
    config_hash: str = ""
    artifact_path: str | None = None

    def summary(self) -> str:
        """Return a concise comparison overview."""
        best = max(self.rows, key=lambda r: r.score, default=None)
        best_id = best.strategy_id if best else "n/a"
        return (
            f"Comparison | strategies={len(self.rows)} | baseline={self.baseline_selector} | "
            f"window={self.comparison_window.get('start_date')}..{self.comparison_window.get('end_date')} | "
            f"best_score_strategy={best_id}"
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Return row metrics as a Polars DataFrame."""
        if not self.rows:
            return pl.DataFrame(
                schema={
                    "selector": pl.Utf8,
                    "strategy_id": pl.Utf8,
                    "strategy_version": pl.Utf8,
                    "intent_mode": pl.Utf8,
                    "tier": pl.Utf8,
                    "promotion_stage": pl.Utf8,
                    "validation_passed": pl.Boolean,
                    "judgment_label": pl.Utf8,
                    "win_rate": pl.Float64,
                    "score": pl.Float64,
                    "exp_decay_percentile": pl.Float64,
                    "multiple_vs_uniform": pl.Float64,
                    "score_delta_vs_baseline": pl.Float64,
                    "exp_decay_delta_vs_baseline": pl.Float64,
                    "is_baseline": pl.Boolean,
                }
            )
        return pl.DataFrame([asdict(row) for row in self.rows])

    def render_table(self) -> str:
        """Return a fixed-width text table of comparison rows."""
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
                row.selector,
                row.intent_mode,
                str(row.tier or "custom"),
                f"{row.win_rate:.2f}%",
                f"{row.score:.2f}%",
                f"{row.exp_decay_percentile:.2f}%",
                (
                    f"{row.multiple_vs_uniform:.3f}x"
                    if row.multiple_vs_uniform is not None
                    else "n/a"
                ),
                row.judgment_label,
            )
            for row in self.rows
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

    def to_json(self, path: str | Path | None = None) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        payload: dict[str, object] = {
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
            "baseline_selector": self.baseline_selector,
            "comparison_window": dict(self.comparison_window),
            "rows": [asdict(row) for row in self.rows],
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "artifact_path": self.artifact_path,
        }
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "ComparisonResult":
        """Load a comparison result from a JSON-compatible dict."""
        rows_raw = payload.get("rows") or []
        rows: list[ComparisonRow] = []
        for item in rows_raw:
            if not isinstance(item, dict):
                continue
            rows.append(
                ComparisonRow(
                    selector=str(item.get("selector", "")),
                    strategy_id=str(item.get("strategy_id", "")),
                    strategy_version=str(item.get("strategy_version", "")),
                    intent_mode=str(item.get("intent_mode", "")),
                    tier=item.get("tier") if item.get("tier") is not None else None,
                    promotion_stage=(
                        str(item["promotion_stage"])
                        if item.get("promotion_stage") is not None
                        else None
                    ),
                    validation_passed=bool(item.get("validation_passed", False)),
                    judgment_label=str(item.get("judgment_label", "")),
                    win_rate=float(item.get("win_rate", 0.0)),
                    score=float(item.get("score", 0.0)),
                    exp_decay_percentile=float(item.get("exp_decay_percentile", 0.0)),
                    multiple_vs_uniform=(
                        float(item["multiple_vs_uniform"])
                        if item.get("multiple_vs_uniform") is not None
                        else None
                    ),
                    score_delta_vs_baseline=(
                        float(item["score_delta_vs_baseline"])
                        if item.get("score_delta_vs_baseline") is not None
                        else None
                    ),
                    exp_decay_delta_vs_baseline=(
                        float(item["exp_decay_delta_vs_baseline"])
                        if item.get("exp_decay_delta_vs_baseline") is not None
                        else None
                    ),
                    is_baseline=bool(item.get("is_baseline", False)),
                )
            )
        window = payload.get("comparison_window")
        if not isinstance(window, dict):
            window = {}
        return cls(
            baseline_selector=str(payload.get("baseline_selector", "")),
            comparison_window=dict(window),
            rows=rows,
            run_id=str(payload.get("run_id", "")),
            config_hash=str(payload.get("config_hash", "")),
            artifact_path=(
                str(payload["artifact_path"])
                if payload.get("artifact_path") is not None
                else None
            ),
        )
