"""Data preparation helpers for strategy-vs-uniform backtest animations."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

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


def spd_table_from_backtest_payload(payload: dict[str, object]) -> pd.DataFrame:
    """Construct raw SPD table DataFrame from backtest payload."""
    raw = payload.get("window_level_data")
    if not isinstance(raw, list):
        raise ValueError(
            "Backtest JSON is missing 'window_level_data' list. "
            "Provide a valid backtest_result.json artifact."
        )
    if not raw:
        raise ValueError("Backtest JSON has empty 'window_level_data'.")
    frame = pd.DataFrame(raw)
    if frame.empty:
        raise ValueError("Backtest JSON window_level_data could not be parsed.")
    return frame


def load_spd_table_from_backtest_json(path: str | Path) -> pd.DataFrame:
    """Load raw SPD table from backtest_result.json."""
    payload = load_backtest_payload(path)
    return spd_table_from_backtest_payload(payload)


def _extract_window_bounds(window_label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(window_label, str) or "→" not in window_label:
        raise ValueError(f"Invalid window label: {window_label!r}")
    parts = [part.strip() for part in window_label.split("→", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid window label: {window_label!r}")
    try:
        start = pd.to_datetime(parts[0])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid window label: {window_label!r}") from exc
    try:
        end = pd.to_datetime(parts[1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid window label: {window_label!r}") from exc
    return start, end


def _normalize_spd_frame(spd_table: pd.DataFrame) -> pd.DataFrame:
    frame = spd_table.copy()
    if "window" not in frame.columns:
        if "index" in frame.columns:
            frame = frame.rename(columns={"index": "window"})
        else:
            frame = frame.reset_index().rename(columns={"index": "window"})

    missing = [col for col in REQUIRED_SPD_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            "SPD table missing required columns for animation: " + ", ".join(missing)
        )

    frame = frame.loc[:, REQUIRED_SPD_COLUMNS].copy()

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for label in frame["window"]:
        start, end = _extract_window_bounds(str(label))
        starts.append(start)
        ends.append(end)

    frame["window_start"] = pd.to_datetime(starts)
    frame["window_end"] = pd.to_datetime(ends)

    for col in _NUMERIC_SPD_COLUMNS:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        frame[col] = numeric

    frame = frame.sort_values(["window_start", "window_end"], kind="mergesort")
    frame = frame.reset_index(drop=True)

    numeric_cols = list(_NUMERIC_SPD_COLUMNS)
    if frame[numeric_cols].isna().all().any():
        bad_cols = [col for col in _NUMERIC_SPD_COLUMNS if frame[col].isna().all()]
        raise ValueError(
            "SPD table contains no valid numeric values for: " + ", ".join(bad_cols)
        )

    frame[numeric_cols] = frame[numeric_cols].ffill().bfill().fillna(0.0)

    return frame


def _select_non_overlapping_windows(frame: pd.DataFrame) -> pd.DataFrame:
    selected_idx: list[int] = []
    last_end: pd.Timestamp | None = None
    for idx, row in frame.iterrows():
        if last_end is None or row["window_start"] > last_end:
            selected_idx.append(int(idx))
            last_end = row["window_end"]
    return frame.iloc[selected_idx].reset_index(drop=True)


def _downsample_frame(frame: pd.DataFrame, max_frames: int) -> pd.DataFrame:
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0.")
    if len(frame) <= max_frames:
        return frame.reset_index(drop=True)

    idx = np.linspace(0, len(frame) - 1, max_frames, dtype=int)
    idx = np.unique(idx)
    if idx[-1] != len(frame) - 1:
        idx = np.append(idx, len(frame) - 1)
    return frame.iloc[idx].reset_index(drop=True)


def prepare_animation_frame_data(
    spd_table: pd.DataFrame,
    *,
    window_mode: str = "rolling",
    max_frames: int = 240,
) -> pd.DataFrame:
    """Prepare deterministic animation frame data from SPD windows."""
    frame = _normalize_spd_frame(spd_table)
    if frame.empty:
        raise ValueError("SPD table is empty.")

    if window_mode == "rolling":
        selected = frame
    elif window_mode == "non-overlapping":
        selected = _select_non_overlapping_windows(frame)
    else:
        raise ValueError("window_mode must be 'rolling' or 'non-overlapping'.")

    selected = _downsample_frame(selected, max_frames=max_frames)

    wins = selected["dynamic_percentile"] > (selected["uniform_percentile"] + _WIN_EPS)
    selected["win_rate_to_date"] = wins.expanding().mean() * 100.0
    selected["cumulative_excess"] = selected["excess_percentile"].cumsum()

    cumulative_dynamic = selected["dynamic_sats_per_dollar"].cumsum()
    cumulative_uniform = selected["uniform_sats_per_dollar"].cumsum()
    selected["cumulative_dynamic_sats_per_dollar"] = cumulative_dynamic
    selected["cumulative_uniform_sats_per_dollar"] = cumulative_uniform
    selected["cumulative_btc_vs_uniform_pct"] = np.where(
        cumulative_uniform.abs() > _WIN_EPS,
        (cumulative_dynamic / cumulative_uniform - 1.0) * 100.0,
        0.0,
    )

    return selected.reset_index(drop=True)
