"""GIF rendering helpers for strategy-vs-uniform backtest animations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .matplotlib_setup import configure_matplotlib_env

REQUIRED_ANIMATION_COLUMNS = (
    "window_start",
    "dynamic_percentile",
    "uniform_percentile",
    "excess_percentile",
    "cumulative_excess",
    "win_rate_to_date",
)


@dataclass(frozen=True, slots=True)
class AnimationStyle:
    """Visual tokens for the strategy-vs-uniform animation."""

    facecolor: str = "#0b1220"
    grid_color: str = "#22304b"
    text_color: str = "#e6edf7"
    dynamic_color: str = "#4ade80"
    uniform_color: str = "#60a5fa"
    positive_area: str = "#22c55e"
    negative_area: str = "#ef4444"
    marker_color: str = "#fbbf24"
    line_width: float = 2.6
    grid_alpha: float = 0.38


DEFAULT_ANIMATION_STYLE = AnimationStyle()


def _validate_animation_frame_data(frame_data: pd.DataFrame) -> None:
    if frame_data.empty:
        raise ValueError("Animation frame data is empty.")
    missing = [col for col in REQUIRED_ANIMATION_COLUMNS if col not in frame_data.columns]
    if missing:
        raise ValueError(
            "Animation data missing required columns: " + ", ".join(sorted(missing))
        )


def _axis_limits(values_a: np.ndarray, values_b: np.ndarray | None = None) -> tuple[float, float]:
    merged = values_a
    if values_b is not None:
        merged = np.concatenate([values_a, values_b])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    pad = max(0.5, span * 0.10)
    if span <= 1e-12:
        return (lower - 1.0, upper + 1.0)
    return (lower - pad, upper + pad)


def render_strategy_vs_uniform_gif(
    frame_data: pd.DataFrame,
    output_path: str | Path,
    *,
    fps: int = 20,
    width: int = 1920,
    height: int = 1080,
    title: str = "Dynamic Strategy vs Uniform DCA",
    style: AnimationStyle = DEFAULT_ANIMATION_STYLE,
) -> dict[str, int | str]:
    """Render a high-definition GIF of strategy-vs-uniform performance."""
    _validate_animation_frame_data(frame_data)
    if fps <= 0:
        raise ValueError("fps must be > 0.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0.")

    configure_matplotlib_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ordered = frame_data.reset_index(drop=True).copy()
    x = pd.to_datetime(ordered["window_start"]).to_numpy()
    dynamic = pd.to_numeric(ordered["dynamic_percentile"], errors="coerce").to_numpy()
    uniform = pd.to_numeric(ordered["uniform_percentile"], errors="coerce").to_numpy()
    excess = pd.to_numeric(ordered["excess_percentile"], errors="coerce").to_numpy()
    cumulative_excess = pd.to_numeric(
        ordered["cumulative_excess"], errors="coerce"
    ).to_numpy()
    win_rate = pd.to_numeric(ordered["win_rate_to_date"], errors="coerce").to_numpy()

    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=style.facecolor,
    )
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.14)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax_top)

    for axis in (ax_top, ax_bottom):
        axis.set_facecolor(style.facecolor)
        axis.grid(True, color=style.grid_color, alpha=style.grid_alpha, linewidth=0.8)
        axis.tick_params(colors=style.text_color, labelsize=10)
        axis.spines["bottom"].set_color(style.grid_color)
        axis.spines["top"].set_color(style.grid_color)
        axis.spines["left"].set_color(style.grid_color)
        axis.spines["right"].set_color(style.grid_color)

    top_min, top_max = _axis_limits(dynamic, uniform)
    bot_min, bot_max = _axis_limits(cumulative_excess)
    ax_top.set_ylim(top_min, top_max)
    ax_bottom.set_ylim(bot_min, bot_max)
    ax_top.set_xlim(x[0], x[-1])

    ax_top.set_ylabel("Window Percentile", color=style.text_color, fontsize=12)
    ax_bottom.set_ylabel("Cumulative Excess", color=style.text_color, fontsize=12)
    ax_bottom.set_xlabel("Window Start Date", color=style.text_color, fontsize=12)
    ax_top.set_title(title, color=style.text_color, fontsize=18, pad=12, weight="bold")

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_bottom.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))

    dynamic_line, = ax_top.plot(
        [],
        [],
        color=style.dynamic_color,
        linewidth=style.line_width,
        label="Dynamic Percentile",
    )
    uniform_line, = ax_top.plot(
        [],
        [],
        color=style.uniform_color,
        linewidth=style.line_width,
        label="Uniform Percentile",
    )
    excess_line, = ax_bottom.plot(
        [],
        [],
        color=style.text_color,
        linewidth=2.0,
        label="Cumulative Excess",
    )
    vline_top = ax_top.axvline(x=x[0], color=style.marker_color, alpha=0.9, linewidth=1.3)
    vline_bottom = ax_bottom.axvline(
        x=x[0], color=style.marker_color, alpha=0.9, linewidth=1.3
    )
    ax_top.axhline(50.0, color=style.grid_color, linestyle="--", linewidth=1.0, alpha=0.6)
    ax_bottom.axhline(0.0, color=style.grid_color, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_top.legend(loc="upper left", facecolor=style.facecolor, edgecolor=style.grid_color)

    stats_text = ax_top.text(
        0.99,
        0.98,
        "",
        transform=ax_top.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color=style.text_color,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": style.facecolor,
            "edgecolor": style.grid_color,
            "alpha": 0.92,
        },
    )
    date_text = ax_bottom.text(
        0.01,
        0.92,
        "",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=style.text_color,
    )

    fill_positive = [None]
    fill_negative = [None]

    def _update(frame_idx: int):
        end = frame_idx + 1
        x_slice = x[:end]
        dynamic_slice = dynamic[:end]
        uniform_slice = uniform[:end]
        cumulative_slice = cumulative_excess[:end]

        dynamic_line.set_data(x_slice, dynamic_slice)
        uniform_line.set_data(x_slice, uniform_slice)
        excess_line.set_data(x_slice, cumulative_slice)
        vline_top.set_xdata([x[end - 1], x[end - 1]])
        vline_bottom.set_xdata([x[end - 1], x[end - 1]])

        if fill_positive[0] is not None:
            fill_positive[0].remove()
        if fill_negative[0] is not None:
            fill_negative[0].remove()
        fill_positive[0] = ax_bottom.fill_between(
            x_slice,
            0.0,
            cumulative_slice,
            where=(cumulative_slice >= 0.0),
            interpolate=True,
            color=style.positive_area,
            alpha=0.30,
        )
        fill_negative[0] = ax_bottom.fill_between(
            x_slice,
            0.0,
            cumulative_slice,
            where=(cumulative_slice < 0.0),
            interpolate=True,
            color=style.negative_area,
            alpha=0.30,
        )

        current_excess = float(excess[end - 1])
        current_cumulative = float(cumulative_excess[end - 1])
        current_win_rate = float(win_rate[end - 1])
        stats_text.set_text(
            f"Excess: {current_excess:+.2f} pct\n"
            f"Cumulative: {current_cumulative:+.2f}\n"
            f"Win-rate-to-date: {current_win_rate:.2f}%"
        )
        date_text.set_text(f"Window: {pd.Timestamp(x[end - 1]).date().isoformat()}")
        return dynamic_line, uniform_line, excess_line, vline_top, vline_bottom

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(ordered),
        interval=1000.0 / float(fps),
        blit=False,
        repeat=False,
    )
    writer = PillowWriter(fps=fps)
    animation.save(str(output_file), writer=writer, dpi=dpi)
    plt.close(fig)

    return {
        "gif_path": str(output_file),
        "frames": int(len(ordered)),
        "fps": int(fps),
        "width": int(width),
        "height": int(height),
    }
