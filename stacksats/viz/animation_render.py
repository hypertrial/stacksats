"""GIF and video rendering helpers for strategy-vs-uniform animations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import polars as pl

from .._optional import import_optional
from .animation_data import build_animation_storyboard
from .matplotlib_setup import configure_matplotlib_env

REQUIRED_ANIMATION_COLUMNS = (
    "window_start",
    "dynamic_percentile",
    "uniform_percentile",
    "excess_percentile",
    "cumulative_btc_vs_uniform_pct",
    "win_rate_to_date",
)


@dataclass(frozen=True, slots=True)
class AnimationStyle:
    """Visual tokens for the shareable strategy-vs-uniform animation."""

    figure_facecolor: str = "#08111d"
    panel_facecolor: str = "#101b2b"
    header_facecolor: str = "#0d1727"
    grid_color: str = "#2a3d57"
    text_color: str = "#f2ede3"
    muted_text: str = "#a7b3c6"
    strategy_color: str = "#39d98a"
    uniform_color: str = "#74b9ff"
    cumulative_color: str = "#f6c267"
    positive_area: str = "#1f8f62"
    negative_area: str = "#b95a53"
    marker_color: str = "#f6c267"
    hero_line_width: float = 3.0
    secondary_line_width: float = 2.2
    grid_alpha: float = 0.24


DEFAULT_ANIMATION_STYLE = AnimationStyle()


def _validate_animation_frame_data(frame_data: pl.DataFrame) -> None:
    if frame_data.height == 0:
        raise ValueError("Animation frame data is empty.")
    missing = [col for col in REQUIRED_ANIMATION_COLUMNS if col not in frame_data.columns]
    if missing:
        raise ValueError(
            "Animation data missing required columns: " + ", ".join(sorted(missing))
        )


def _axis_limits(values_a: np.ndarray, values_b: np.ndarray | None = None) -> tuple[float, float]:
    merged = values_a if values_b is None else np.concatenate([values_a, values_b])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    pad = max(0.5, span * 0.12)
    if span <= 1e-12:
        return (lower - 1.0, upper + 1.0)
    return (lower - pad, upper + pad)


def _compact_layout(width: int, height: int) -> bool:
    return width < 900 or height < 500


def _ensure_storyboard_columns(frame_data: pl.DataFrame) -> pl.DataFrame:
    enriched = frame_data.clone()
    if "raw_window_count" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(enriched.height).alias("raw_window_count"))
    if "selected_window_count" not in enriched.columns:
        enriched = enriched.with_columns(
            pl.lit(enriched.height).alias("selected_window_count")
        )
    if "window_start_label" not in enriched.columns:
        enriched = enriched.with_columns(
            pl.col("window_start").dt.strftime("%Y-%m-%d").alias("window_start_label")
        )
    if "window_end_label" not in enriched.columns:
        if "window_end" in enriched.columns:
            enriched = enriched.with_columns(
                pl.col("window_end").dt.strftime("%Y-%m-%d").alias("window_end_label")
            )
        else:
            enriched = enriched.with_columns(
                pl.col("window_start").dt.strftime("%Y-%m-%d").alias("window_end_label")
            )
    if "window" not in enriched.columns:
        enriched = enriched.with_columns(
            (
                pl.col("window_start_label") + pl.lit(" → ") + pl.col("window_end_label")
            ).alias("window")
        )
    for name, value in (
        ("milestone_label", [None] * enriched.height),
        ("is_new_high", [False] * enriched.height),
        ("is_new_low", [False] * enriched.height),
        ("window_mode", ["rolling"] * enriched.height),
    ):
        if name not in enriched.columns:
            enriched = enriched.with_columns(pl.Series(name, value))
    return enriched


def _configure_axes(axis, *, style: AnimationStyle, compact: bool) -> None:
    axis.set_facecolor(style.panel_facecolor)
    axis.grid(True, color=style.grid_color, alpha=style.grid_alpha, linewidth=0.8)
    axis.tick_params(colors=style.muted_text, labelsize=8 if compact else 10)
    for side in ("bottom", "top", "left", "right"):
        axis.spines[side].set_color(style.grid_color)


def _resolve_writer(animation_mod, media_format: str, *, fps: int):
    if media_format == "gif":
        return animation_mod.PillowWriter(fps=fps)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Video export requires a system 'ffmpeg' binary. "
            "Install ffmpeg or omit --video-format."
        )
    FFMpegWriter = animation_mod.FFMpegWriter
    if media_format == "mp4":
        return FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    if media_format == "webm":
        return FFMpegWriter(
            fps=fps,
            codec="libvpx-vp9",
            extra_args=["-pix_fmt", "yuv420p"],
        )
    raise ValueError("media_format must be 'gif', 'mp4', or 'webm'.")


def _render_strategy_vs_uniform_media(
    frame_data: pl.DataFrame,
    output_path: str | Path,
    *,
    media_format: str,
    fps: int = 20,
    width: int = 1920,
    height: int = 1080,
    title: str = "Strategy vs Uniform DCA",
    style: AnimationStyle = DEFAULT_ANIMATION_STYLE,
) -> dict[str, int | str]:
    _validate_animation_frame_data(frame_data)
    if fps <= 0:
        raise ValueError("fps must be > 0.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0.")

    normalized = _ensure_storyboard_columns(frame_data)
    storyboard = build_animation_storyboard(normalized, fps=fps)
    compact = _compact_layout(width, height)

    configure_matplotlib_env()
    matplotlib = import_optional(
        "matplotlib",
        extra="viz",
        feature="strategy animation rendering",
    )
    matplotlib.use("Agg")
    mdates = import_optional(
        "matplotlib.dates",
        extra="viz",
        feature="strategy animation rendering",
    )
    plt = import_optional(
        "matplotlib.pyplot",
        extra="viz",
        feature="strategy animation rendering",
    )
    animation_mod = import_optional(
        "matplotlib.animation",
        extra="viz",
        feature="strategy animation rendering",
    )
    FuncAnimation = animation_mod.FuncAnimation

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ordered = storyboard.frame_data
    x = ordered["window_start"].to_numpy()
    dynamic = ordered["dynamic_percentile"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    uniform = ordered["uniform_percentile"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    excess = ordered["excess_percentile"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    cumulative = (
        ordered["cumulative_btc_vs_uniform_pct"]
        .cast(pl.Float64)
        .fill_null(np.nan)
        .to_numpy()
    )
    win_rate = ordered["win_rate_to_date"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    windows = ordered["window"].to_list()
    milestone_labels = ordered["milestone_label"].to_list()
    new_highs = (
        ordered["is_new_high"].to_numpy()
        if "is_new_high" in ordered.columns
        else np.zeros(ordered.height, dtype=bool)
    )
    new_lows = (
        ordered["is_new_low"].to_numpy()
        if "is_new_low" in ordered.columns
        else np.zeros(ordered.height, dtype=bool)
    )
    window_mode = (
        str(ordered["window_mode"][0]) if "window_mode" in ordered.columns else "rolling"
    )

    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=style.figure_facecolor,
    )
    grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[0.70, 2.6, 1.3, 0.42],
        hspace=0.18 if compact else 0.14,
        top=0.96,
        bottom=0.06,
    )
    ax_header = fig.add_subplot(grid[0, 0])
    ax_hero = fig.add_subplot(grid[1, 0])
    ax_secondary = fig.add_subplot(grid[2, 0], sharex=ax_hero)
    ax_footer = fig.add_subplot(grid[3, 0])

    for axis in (ax_header, ax_footer):
        axis.set_facecolor(style.header_facecolor)
        axis.set_xticks([])
        axis.set_yticks([])
        for side in ("bottom", "top", "left", "right"):
            axis.spines[side].set_visible(False)

    for axis in (ax_hero, ax_secondary):
        _configure_axes(axis, style=style, compact=compact)

    hero_min, hero_max = _axis_limits(cumulative)
    secondary_min, secondary_max = _axis_limits(dynamic, uniform)
    ax_hero.set_ylim(hero_min, hero_max)
    ax_secondary.set_ylim(secondary_min, secondary_max)
    ax_hero.set_xlim(x[0], x[-1])

    ax_hero.set_ylabel(
        "Cumulative BTC vs Uniform (%)",
        color=style.text_color,
        fontsize=11 if compact else 13,
    )
    ax_secondary.set_ylabel(
        "Per-Window Percentile",
        color=style.text_color,
        fontsize=10 if compact else 12,
    )
    ax_secondary.set_xlabel(
        "Window Start",
        color=style.muted_text,
        fontsize=9 if compact else 11,
    )
    ax_secondary.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_secondary.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax_hero.axhline(0.0, color=style.grid_color, linestyle="--", linewidth=1.0, alpha=0.9)
    ax_secondary.axhline(
        50.0, color=style.grid_color, linestyle="--", linewidth=1.0, alpha=0.7
    )

    final_advantage = float(cumulative[-1])
    final_win_rate = float(win_rate[-1])
    final_excess = float(excess[-1])
    raw_windows = storyboard.raw_window_count
    selected_windows = storyboard.selected_window_count

    ax_header.text(
        0.02,
        0.72,
        title,
        color=style.text_color,
        fontsize=18 if compact else 22,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax_header.text(
        0.02,
        0.34,
        (
            f"{storyboard.start_date} to {storyboard.end_date}  |  "
            f"{window_mode} windows  |  {selected_windows}/{raw_windows} selected"
        ),
        color=style.muted_text,
        fontsize=9 if compact else 11,
        ha="left",
        va="center",
    )
    ax_header.text(
        0.02,
        0.10,
        "Top: cumulative outcome  |  Bottom: per-window percentile",
        color=style.muted_text,
        fontsize=8 if compact else 10,
        ha="left",
        va="center",
    )
    if compact:
        ax_header.text(
            0.98,
            0.52,
            f"Final cumulative BTC vs uniform {final_advantage:+.2f}%",
            color=style.cumulative_color,
            fontsize=10,
            fontweight="semibold",
            ha="right",
            va="center",
        )
    else:
        for xpos, label, value, color in (
            (
                0.63,
                "Final cumulative BTC vs uniform",
                f"{final_advantage:+.2f}%",
                style.cumulative_color,
            ),
            (0.79, "Final win rate", f"{final_win_rate:.1f}%", style.strategy_color),
            (0.93, "Latest excess", f"{final_excess:+.2f} pct", style.text_color),
        ):
            ax_header.text(
                xpos,
                0.68,
                label,
                color=style.muted_text,
                fontsize=9,
                ha="center",
                va="center",
            )
            ax_header.text(
                xpos,
                0.34,
                value,
                color=color,
                fontsize=13,
                fontweight="semibold",
                ha="center",
                va="center",
            )

    hero_line, = ax_hero.plot(
        [],
        [],
        color=style.cumulative_color,
        linewidth=style.hero_line_width,
        solid_capstyle="round",
    )
    strategy_line, = ax_secondary.plot(
        [],
        [],
        color=style.strategy_color,
        linewidth=style.secondary_line_width,
        solid_capstyle="round",
    )
    uniform_line, = ax_secondary.plot(
        [],
        [],
        color=style.uniform_color,
        linewidth=style.secondary_line_width,
        solid_capstyle="round",
    )
    hero_marker, = ax_hero.plot([], [], marker="o", color=style.marker_color, markersize=7)
    strategy_marker, = ax_secondary.plot(
        [], [], marker="o", color=style.strategy_color, markersize=6
    )
    uniform_marker, = ax_secondary.plot(
        [], [], marker="o", color=style.uniform_color, markersize=6
    )
    hero_vline = ax_hero.axvline(x=x[0], color=style.marker_color, linewidth=1.2, alpha=0.85)
    secondary_vline = ax_secondary.axvline(
        x=x[0], color=style.marker_color, linewidth=1.2, alpha=0.85
    )
    hero_direct_label = ax_hero.text(
        0.0,
        0.0,
        "Cumulative BTC vs uniform",
        color=style.cumulative_color,
        fontsize=9 if compact else 11,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )
    strategy_direct_label = ax_secondary.text(
        0.0,
        0.0,
        "Strategy",
        color=style.strategy_color,
        fontsize=8 if compact else 10,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )
    uniform_direct_label = ax_secondary.text(
        0.0,
        0.0,
        "Uniform",
        color=style.uniform_color,
        fontsize=8 if compact else 10,
        fontweight="semibold",
        ha="left",
        va="top",
    )
    current_stats_text = ax_hero.text(
        0.02,
        0.94,
        "",
        transform=ax_hero.transAxes,
        color=style.text_color,
        fontsize=9 if compact else 11,
        va="top",
        ha="left",
    )
    milestone_text = ax_hero.text(
        0.02,
        0.06,
        "",
        transform=ax_hero.transAxes,
        color=style.marker_color,
        fontsize=8 if compact else 10,
        fontweight="semibold",
        va="bottom",
        ha="left",
    )
    footer_text = ax_footer.text(
        0.02,
        0.5,
        "",
        color=style.muted_text,
        fontsize=8 if compact else 10,
        ha="left",
        va="center",
    )

    fill_positive = [None]
    fill_negative = [None]

    def _update(sequence_idx: int):
        data_idx = storyboard.sequence_indices[sequence_idx]
        end = data_idx + 1
        x_slice = x[:end]
        cumulative_slice = cumulative[:end]
        dynamic_slice = dynamic[:end]
        uniform_slice = uniform[:end]

        hero_line.set_data(x_slice, cumulative_slice)
        strategy_line.set_data(x_slice, dynamic_slice)
        uniform_line.set_data(x_slice, uniform_slice)
        hero_marker.set_data([x[end - 1]], [cumulative[end - 1]])
        strategy_marker.set_data([x[end - 1]], [dynamic[end - 1]])
        uniform_marker.set_data([x[end - 1]], [uniform[end - 1]])
        hero_vline.set_xdata([x[end - 1], x[end - 1]])
        secondary_vline.set_xdata([x[end - 1], x[end - 1]])

        if fill_positive[0] is not None:
            fill_positive[0].remove()
        if fill_negative[0] is not None:
            fill_negative[0].remove()
        fill_positive[0] = ax_hero.fill_between(
            x_slice,
            0.0,
            cumulative_slice,
            where=(cumulative_slice >= 0.0),
            interpolate=True,
            color=style.positive_area,
            alpha=0.28,
        )
        fill_negative[0] = ax_hero.fill_between(
            x_slice,
            0.0,
            cumulative_slice,
            where=(cumulative_slice < 0.0),
            interpolate=True,
            color=style.negative_area,
            alpha=0.26,
        )

        hero_direct_label.set_position((x[end - 1], cumulative[end - 1]))
        strategy_direct_label.set_position((x[end - 1], dynamic[end - 1]))
        uniform_direct_label.set_position((x[end - 1], uniform[end - 1]))

        current_stats_text.set_text(
            f"Current excess {float(excess[end - 1]):+.2f} pct\n"
            f"Win-rate-to-date {float(win_rate[end - 1]):.1f}%"
        )

        milestone = milestone_labels[end - 1]
        if compact:
            milestone_text.set_text("")
        else:
            if milestone:
                suffix = ""
                if bool(new_highs[end - 1]):
                    suffix = "  |  Best cumulative level so far"
                elif bool(new_lows[end - 1]):
                    suffix = "  |  Lowest cumulative level so far"
                milestone_text.set_text(milestone + suffix)
            else:
                milestone_text.set_text("")

        footer_text.set_text(
            f"Window {windows[end - 1]}  |  Mode {window_mode}  |  "
            f"Windows {selected_windows}/{raw_windows} shown"
        )
        return (
            hero_line,
            strategy_line,
            uniform_line,
            hero_marker,
            strategy_marker,
            uniform_marker,
            hero_vline,
            secondary_vline,
        )

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(storyboard.sequence_indices),
        interval=1000.0 / float(fps),
        blit=False,
        repeat=False,
    )
    writer = _resolve_writer(animation_mod, media_format, fps=fps)
    try:
        animation.save(str(output_file), writer=writer, dpi=dpi)
    except Exception as exc:
        if media_format == "webm":
            raise RuntimeError(
                "WebM export failed. Your ffmpeg build may not support VP9. "
                "Try '--video-format mp4' instead."
            ) from exc
        raise
    finally:
        plt.close(fig)

    return {
        "frames": int(len(storyboard.sequence_indices)),
        "fps": int(fps),
        "width": int(width),
        "height": int(height),
        "path": str(output_file),
    }


def render_strategy_vs_uniform_gif(
    frame_data: pl.DataFrame,
    output_path: str | Path,
    *,
    fps: int = 20,
    width: int = 1920,
    height: int = 1080,
    title: str = "Strategy vs Uniform DCA",
    style: AnimationStyle = DEFAULT_ANIMATION_STYLE,
) -> dict[str, int | str]:
    """Render a polished GIF of strategy-vs-uniform performance."""
    return _render_strategy_vs_uniform_media(
        frame_data,
        output_path,
        media_format="gif",
        fps=fps,
        width=width,
        height=height,
        title=title,
        style=style,
    )


def render_strategy_vs_uniform_video(
    frame_data: pl.DataFrame,
    output_path: str | Path,
    *,
    video_format: str,
    fps: int = 20,
    width: int = 1920,
    height: int = 1080,
    title: str = "Strategy vs Uniform DCA",
    style: AnimationStyle = DEFAULT_ANIMATION_STYLE,
) -> dict[str, int | str]:
    """Render an MP4/WebM video of strategy-vs-uniform performance."""
    return _render_strategy_vs_uniform_media(
        frame_data,
        output_path,
        media_format=video_format,
        fps=fps,
        width=width,
        height=height,
        title=title,
        style=style,
    )
