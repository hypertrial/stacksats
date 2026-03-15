"""Data preparation helpers for strategy-vs-uniform backtest animations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import polars as pl

REQUIRED_SPD_COLUMNS = (
    "window",
    "dynamic_percentile",
    "uniform_percentile",
    "excess_percentile",
    "dynamic_sats_per_dollar",
    "uniform_sats_per_dollar",
)

_NUMERIC_SPD_COLUMNS = (
    "dynamic_percentile",
    "uniform_percentile",
    "excess_percentile",
    "dynamic_sats_per_dollar",
    "uniform_sats_per_dollar",
)

_WIN_EPS = 1e-10


def load_backtest_payload(path: str | Path) -> dict[str, object]:
    """Load backtest payload JSON and validate the top-level shape."""
    payload_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid backtest JSON: {payload_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Backtest JSON root must be an object.")
    return payload


def spd_table_from_backtest_payload(payload: dict[str, object]) -> pl.DataFrame:
    """Construct raw SPD table DataFrame from backtest payload."""
    raw = payload.get("window_level_data")
    if not isinstance(raw, list):
        raise ValueError(
            "Backtest JSON is missing 'window_level_data' list. "
            "Provide a valid backtest_result.json artifact."
        )
    if not raw:
        raise ValueError("Backtest JSON has empty 'window_level_data'.")
    frame = pl.DataFrame(raw)
    if frame.height == 0:
        raise ValueError("Backtest JSON window_level_data could not be parsed.")
    return frame


def load_spd_table_from_backtest_json(path: str | Path) -> pl.DataFrame:
    """Load raw SPD table from backtest_result.json."""
    payload = load_backtest_payload(path)
    return spd_table_from_backtest_payload(payload)


def _parse_window_date(s: str) -> datetime:
    """Parse ISO date string to datetime."""
    from datetime import date
    try:
        return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        pass
    # Fallback for date-only
    try:
        d = date.fromisoformat(s.strip())
        return datetime.combine(d, datetime.min.time())
    except ValueError as exc:
        raise ValueError(f"Invalid date: {s!r}") from exc


def _extract_window_bounds(window_label: str) -> tuple[datetime, datetime]:
    if not isinstance(window_label, str) or "→" not in window_label:
        raise ValueError(f"Invalid window label: {window_label!r}")
    parts = [part.strip() for part in window_label.split("→", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid window label: {window_label!r}")
    start = _parse_window_date(parts[0])
    end = _parse_window_date(parts[1])
    return start, end


def _normalize_spd_frame(spd_table: pl.DataFrame) -> pl.DataFrame:
    frame = spd_table.clone()
    if "window" not in frame.columns:
        if "index" in frame.columns:
            frame = frame.rename({"index": "window"})
        else:
            raise ValueError(
                "SPD table must have 'window' or 'index' column with window labels."
            )

    missing = [col for col in REQUIRED_SPD_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            "SPD table missing required columns for animation: " + ", ".join(missing)
        )

    frame = frame.select(REQUIRED_SPD_COLUMNS)

    window_col = frame["window"]
    starts = []
    ends = []
    for label in window_col:
        start, end = _extract_window_bounds(str(label))
        starts.append(start)
        ends.append(end)

    frame = frame.with_columns(
        pl.Series("window_start", starts),
        pl.Series("window_end", ends),
    )

    for col in _NUMERIC_SPD_COLUMNS:
        frame = frame.with_columns(
            pl.col(col)
            .cast(pl.Float64, strict=False)
            .replace([float("inf"), float("-inf")], None)
            .alias(col)
        )

    frame = frame.sort(["window_start", "window_end"])

    for col in _NUMERIC_SPD_COLUMNS:
        if frame[col].null_count() == frame.height:
            raise ValueError(
                f"SPD table contains no valid numeric values for: {col}"
            )

    frame = frame.with_columns([
        pl.col(c).fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0).alias(c)
        for c in _NUMERIC_SPD_COLUMNS
    ])

    return frame


def _select_non_overlapping_windows(frame: pl.DataFrame) -> pl.DataFrame:
    selected_rows: list[pl.DataFrame] = []
    last_end: datetime | None = None
    for row in frame.iter_rows(named=True):
        ws = row["window_start"]
        we = row["window_end"]
        if last_end is None or ws > last_end:
            selected_rows.append(pl.DataFrame([row]))
            last_end = we
    if not selected_rows:
        return frame.clear()
    return pl.concat(selected_rows)


def _downsample_frame(frame: pl.DataFrame, max_frames: int) -> pl.DataFrame:
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0.")
    n = frame.height
    if n <= max_frames:
        return frame

    idx = np.linspace(0, n - 1, max_frames, dtype=int)
    idx = np.unique(idx)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return frame.filter(pl.int_range(0, n).is_in(pl.lit(idx.tolist())))


def prepare_animation_frame_data(
    spd_table: pl.DataFrame,
    *,
    window_mode: str = "rolling",
    max_frames: int = 240,
) -> pl.DataFrame:
    """Prepare deterministic animation frame data from SPD windows."""
    frame = _normalize_spd_frame(spd_table)
    if frame.height == 0:
        raise ValueError("SPD table is empty.")

    if window_mode == "rolling":
        selected = frame
    elif window_mode == "non-overlapping":
        selected = _select_non_overlapping_windows(frame)
    else:
        raise ValueError("window_mode must be 'rolling' or 'non-overlapping'.")

    selected = _downsample_frame(selected, max_frames=max_frames)

    wins = selected["dynamic_percentile"] > (selected["uniform_percentile"] + _WIN_EPS)
    selected = selected.with_columns(
        (wins.cum_sum() / (pl.int_range(1, pl.len() + 1)) * 100.0).alias("win_rate_to_date"),
        pl.col("excess_percentile").cum_sum().alias("cumulative_excess"),
    )

    cumulative_dynamic = selected["dynamic_sats_per_dollar"].cum_sum()
    cumulative_uniform = selected["uniform_sats_per_dollar"].cum_sum()
    cumulative_btc_pct = pl.when(cumulative_uniform.abs() > _WIN_EPS).then(
        (cumulative_dynamic / cumulative_uniform - 1.0) * 100.0
    ).otherwise(0.0)

    selected = selected.with_columns(
        cumulative_dynamic.alias("cumulative_dynamic_sats_per_dollar"),
        cumulative_uniform.alias("cumulative_uniform_sats_per_dollar"),
        cumulative_btc_pct.alias("cumulative_btc_vs_uniform_pct"),
    )

    return selected
