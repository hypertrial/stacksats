"""Data preparation helpers for strategy-vs-uniform backtest animations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class AnimationStoryboard:
    """Render-ready animation metadata with a repeated frame sequence."""

    frame_data: pl.DataFrame
    sequence_indices: tuple[int, ...]
    intro_hold_frames: int
    outro_hold_frames: int
    raw_window_count: int
    selected_window_count: int
    start_date: str
    end_date: str


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
    try:
        d = date.fromisoformat(s.strip())
        return datetime.combine(d, datetime.min.time())
    except ValueError as exc:
        raise ValueError(f"Invalid date: {s!r}") from exc


def _extract_window_bounds(window_label: str) -> tuple[datetime, datetime]:
    """Parse the canonical window label used in animation artifacts."""
    if not isinstance(window_label, str) or "→" not in window_label:
        raise ValueError(f"Invalid window label: {window_label!r}")
    parts = [part.strip() for part in window_label.split("→", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid window label: {window_label!r}")
    return _parse_window_date(parts[0]), _parse_window_date(parts[1])


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

    frame = frame.select(REQUIRED_SPD_COLUMNS).with_columns(
        pl.col("window")
        .str.split_exact("→", 1)
        .struct.field("field_0")
        .str.strip_chars()
        .str.to_datetime(strict=False)
        .alias("window_start"),
        pl.col("window")
        .str.split_exact("→", 1)
        .struct.field("field_1")
        .str.strip_chars()
        .str.to_datetime(strict=False)
        .alias("window_end"),
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
            raise ValueError(f"SPD table contains no valid numeric values for: {col}")

    return frame.with_columns(
        [
            pl.col(c)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .fill_null(0.0)
            .alias(c)
            for c in _NUMERIC_SPD_COLUMNS
        ]
    )


def _select_non_overlapping_windows(frame: pl.DataFrame) -> pl.DataFrame:
    selected_rows: list[pl.DataFrame] = []
    last_end: datetime | None = None
    for row in frame.iter_rows(named=True):
        window_start = row["window_start"]
        window_end = row["window_end"]
        if last_end is None or window_start > last_end:
            selected_rows.append(pl.DataFrame([row]))
            last_end = window_end
    if not selected_rows:
        return frame.clear()
    return pl.concat(selected_rows)


def _compute_running_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    wins = frame["dynamic_percentile"] > (frame["uniform_percentile"] + _WIN_EPS)
    cumulative_dynamic = frame["dynamic_sats_per_dollar"].cum_sum()
    cumulative_uniform = frame["uniform_sats_per_dollar"].cum_sum()
    cumulative_btc_pct = pl.when(cumulative_uniform.abs() > _WIN_EPS).then(
        (cumulative_dynamic / cumulative_uniform - 1.0) * 100.0
    ).otherwise(0.0)
    cumulative_btc_series = cumulative_btc_pct.alias("cumulative_btc_vs_uniform_pct")
    raw_count = frame.height

    enriched = frame.with_row_index("frame_index").with_columns(
        (wins.cum_sum() / pl.int_range(1, pl.len() + 1) * 100.0).alias("win_rate_to_date"),
        pl.col("excess_percentile").cum_sum().alias("cumulative_excess"),
        cumulative_dynamic.alias("cumulative_dynamic_sats_per_dollar"),
        cumulative_uniform.alias("cumulative_uniform_sats_per_dollar"),
        cumulative_btc_series,
        pl.lit(raw_count).alias("raw_window_count"),
        pl.lit(raw_count).alias("selected_window_count"),
        pl.col("window_start").dt.strftime("%Y-%m-%d").alias("window_start_label"),
        pl.col("window_end").dt.strftime("%Y-%m-%d").alias("window_end_label"),
    )

    cumulative = enriched["cumulative_btc_vs_uniform_pct"].to_numpy()
    dynamic = enriched["dynamic_percentile"].to_numpy()
    uniform = enriched["uniform_percentile"].to_numpy()
    starts = enriched["window_start"].to_list()
    extrema_score = np.zeros(len(cumulative), dtype=float)
    is_local_extremum = np.zeros(len(cumulative), dtype=bool)
    is_year_boundary = np.zeros(len(cumulative), dtype=bool)
    lead_change = np.zeros(len(cumulative), dtype=bool)
    sign_change = np.zeros(len(cumulative), dtype=bool)
    is_new_high = np.zeros(len(cumulative), dtype=bool)
    is_new_low = np.zeros(len(cumulative), dtype=bool)

    if len(cumulative) > 0:
        lead_state = dynamic > (uniform + _WIN_EPS)
        sign_state = np.sign(np.where(np.abs(cumulative) <= _WIN_EPS, 0.0, cumulative))
        running_max = cumulative[0]
        running_min = cumulative[0]
        for idx in range(1, len(cumulative)):
            is_year_boundary[idx] = starts[idx].year != starts[idx - 1].year
            lead_change[idx] = bool(lead_state[idx] != lead_state[idx - 1])
            sign_change[idx] = bool(sign_state[idx] != sign_state[idx - 1])
            if cumulative[idx] > running_max + _WIN_EPS:
                running_max = cumulative[idx]
                is_new_high[idx] = True
            if cumulative[idx] < running_min - _WIN_EPS:
                running_min = cumulative[idx]
                is_new_low[idx] = True
        for idx in range(1, len(cumulative) - 1):
            left = cumulative[idx] - cumulative[idx - 1]
            right = cumulative[idx + 1] - cumulative[idx]
            if (left > _WIN_EPS and right < -_WIN_EPS) or (
                left < -_WIN_EPS and right > _WIN_EPS
            ):
                is_local_extremum[idx] = True
                extrema_score[idx] = abs(left) + abs(right)

    milestone_labels: list[str | None] = []
    for idx in range(len(cumulative)):
        label: str | None = None
        if sign_change[idx]:
            label = (
                "Turns positive vs uniform"
                if cumulative[idx] >= 0.0
                else "Falls behind uniform"
            )
        elif is_new_high[idx]:
            label = "New cumulative high"
        elif is_new_low[idx]:
            label = "New cumulative low"
        elif lead_change[idx]:
            label = "Window percentile lead changes"
        milestone_labels.append(label)

    return enriched.with_columns(
        pl.Series("lead_change", lead_change),
        pl.Series("cumulative_sign_change", sign_change),
        pl.Series("is_local_extremum", is_local_extremum),
        pl.Series("extremum_score", extrema_score),
        pl.Series("is_year_boundary", is_year_boundary),
        pl.Series("is_new_high", is_new_high),
        pl.Series("is_new_low", is_new_low),
        pl.Series("milestone_label", milestone_labels),
    )


def _compress_required_indices(
    essential: list[int],
    local_extrema: list[int],
    max_frames: int,
) -> list[int]:
    if len(essential) >= max_frames:
        if max_frames == 1:
            return [essential[-1]]
        pick = np.linspace(0, len(essential) - 1, max_frames, dtype=int)
        return sorted({essential[idx] for idx in pick})

    selected = list(essential)
    remaining = max_frames - len(selected)
    if remaining <= 0:
        return sorted(selected)

    for idx in local_extrema:
        if idx in selected:
            continue
        selected.append(idx)
        if len(selected) >= max_frames:
            return sorted(selected)
    return sorted(selected)


def _select_frame_indices(frame: pl.DataFrame, max_frames: int) -> list[int]:
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0.")
    n = frame.height
    if n <= max_frames:
        return list(range(n))

    first_last = [0, n - 1]
    sign_changes = [
        int(idx)
        for idx in frame.filter(pl.col("cumulative_sign_change"))["frame_index"].to_list()
    ]
    year_boundaries = [
        int(idx)
        for idx in frame.filter(pl.col("is_year_boundary"))["frame_index"].to_list()
    ]
    lead_changes = [
        int(idx) for idx in frame.filter(pl.col("lead_change"))["frame_index"].to_list()
    ]
    local_extrema_df = frame.filter(pl.col("is_local_extremum")).sort(
        ["extremum_score", "frame_index"], descending=[True, False]
    )
    local_extrema = [int(idx) for idx in local_extrema_df["frame_index"].to_list()]

    cumulative = frame["cumulative_btc_vs_uniform_pct"].to_numpy()
    global_extrema = sorted(
        {
            int(np.argmax(cumulative)),
            int(np.argmin(cumulative)),
        }
    )
    essential = sorted(
        {
            *first_last,
            *sign_changes,
            *year_boundaries,
            *global_extrema,
        }
    )
    selected = _compress_required_indices(essential, local_extrema, max_frames)

    remaining = max_frames - len(selected)
    if remaining <= 0:
        return sorted(selected)

    event_candidates = [idx for idx in lead_changes if idx not in selected]
    for idx in event_candidates:
        selected.append(idx)
        if len(selected) >= max_frames:
            return sorted(selected)

    still_remaining = max_frames - len(selected)
    if still_remaining <= 0:
        return sorted(selected)

    even_fill = np.linspace(0, n - 1, max_frames, dtype=int)
    for idx in np.unique(even_fill):
        as_int = int(idx)
        if as_int in selected:
            continue
        selected.append(as_int)
        if len(selected) >= max_frames:
            break
    return sorted(selected)


def _apply_frame_selection(frame: pl.DataFrame, max_frames: int) -> pl.DataFrame:
    indices = _select_frame_indices(frame, max_frames=max_frames)
    selected = frame.filter(pl.col("frame_index").is_in(indices)).sort("frame_index")
    return selected.with_columns(pl.lit(selected.height).alias("selected_window_count"))


def _downsample_frame(frame: pl.DataFrame, max_frames: int) -> pl.DataFrame:
    """Backward-compatible wrapper for deterministic event-aware frame selection."""
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0.")
    if "cumulative_sign_change" not in frame.columns:
        n = frame.height
        if n <= max_frames:
            return frame
        idx = np.linspace(0, n - 1, max_frames, dtype=int)
        idx = np.unique(idx)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return frame.filter(pl.int_range(0, n).is_in(pl.lit(idx.tolist())))

    working = frame if "frame_index" in frame.columns else frame.with_row_index("frame_index")
    return _apply_frame_selection(working, max_frames=max_frames)


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
        normalized = frame
    elif window_mode == "non-overlapping":
        normalized = _select_non_overlapping_windows(frame)
    else:
        raise ValueError("window_mode must be 'rolling' or 'non-overlapping'.")

    enriched = _compute_running_metrics(normalized).with_columns(
        pl.lit(window_mode).alias("window_mode")
    )
    return _apply_frame_selection(enriched, max_frames=max_frames)


def build_animation_storyboard(frame_data: pl.DataFrame, *, fps: int) -> AnimationStoryboard:
    """Build render sequence metadata with intro/outro holds."""
    if frame_data.is_empty():
        raise ValueError("Animation frame data is empty.")
    if fps <= 0:
        raise ValueError("fps must be > 0.")

    intro_hold = max(2, int(round(fps * 0.6)))
    outro_hold = max(6, int(round(fps * 1.2)))
    base_indices = tuple(range(frame_data.height))
    sequence = (
        tuple([0] * intro_hold)
        + base_indices
        + tuple([frame_data.height - 1] * outro_hold)
    )
    return AnimationStoryboard(
        frame_data=frame_data,
        sequence_indices=sequence,
        intro_hold_frames=intro_hold,
        outro_hold_frames=outro_hold,
        raw_window_count=int(frame_data["raw_window_count"][0]),
        selected_window_count=int(frame_data["selected_window_count"][0]),
        start_date=str(frame_data["window_start_label"][0]),
        end_date=str(frame_data["window_end_label"][-1]),
    )
