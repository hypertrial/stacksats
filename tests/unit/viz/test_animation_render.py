from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from stacksats.viz.animation_render import (
    _compact_layout,
    _ensure_storyboard_columns,
    _resolve_writer,
    render_strategy_vs_uniform_gif,
)


def _frame_data(n: int = 6) -> pl.DataFrame:
    base = datetime(2024, 1, 1)
    starts = [base + timedelta(days=i) for i in range(n)]
    dynamic = np.linspace(40.0, 65.0, n)
    uniform = np.linspace(42.0, 55.0, n)
    excess = dynamic - uniform
    return pl.DataFrame({
        "window_start": starts,
        "dynamic_percentile": dynamic,
        "uniform_percentile": uniform,
        "excess_percentile": excess,
        "cumulative_btc_vs_uniform_pct": np.cumsum(excess),
        "win_rate_to_date": np.linspace(50.0, 75.0, n),
    })


def test_render_strategy_vs_uniform_gif_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "strategy_vs_uniform_hd.gif"
    meta = render_strategy_vs_uniform_gif(
        _frame_data(5),
        output_path,
        fps=5,
        width=640,
        height=360,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert meta["frames"] == 14
    assert meta["fps"] == 5
    assert meta["width"] == 640
    assert meta["height"] == 360


def test_render_strategy_vs_uniform_gif_rejects_missing_columns(tmp_path: Path) -> None:
    bad = pl.DataFrame({
        "window_start": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    })
    with pytest.raises(ValueError, match="missing required columns"):
        render_strategy_vs_uniform_gif(bad, tmp_path / "out.gif")


def test_render_strategy_vs_uniform_gif_rejects_bad_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps must be > 0"):
        render_strategy_vs_uniform_gif(_frame_data(3), tmp_path / "out.gif", fps=0)
    with pytest.raises(ValueError, match="width and height must be > 0"):
        render_strategy_vs_uniform_gif(_frame_data(3), tmp_path / "out.gif", width=0)


def test_compact_layout_switches_at_small_dimensions() -> None:
    assert _compact_layout(640, 360) is True
    assert _compact_layout(1280, 720) is False


def test_resolve_writer_uses_pillow_for_gif() -> None:
    class FakePillowWriter:
        def __init__(self, fps):
            self.fps = fps

    animation_mod = SimpleNamespace(PillowWriter=FakePillowWriter, FFMpegWriter=object)
    writer = _resolve_writer(animation_mod, "gif", fps=8)
    assert isinstance(writer, FakePillowWriter)
    assert writer.fps == 8


def test_resolve_writer_requires_ffmpeg_for_video(monkeypatch: pytest.MonkeyPatch) -> None:
    animation_mod = SimpleNamespace(PillowWriter=object, FFMpegWriter=object)
    monkeypatch.setattr(
        "stacksats.viz.animation_render.shutil.which",
        lambda name: None,
    )
    with pytest.raises(RuntimeError, match="ffmpeg"):
        _resolve_writer(animation_mod, "mp4", fps=8)


def test_resolve_writer_uses_ffmpeg_for_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFFMpegWriter:
        def __init__(self, *, fps, codec, extra_args):
            self.fps = fps
            self.codec = codec
            self.extra_args = extra_args

    animation_mod = SimpleNamespace(PillowWriter=object, FFMpegWriter=FakeFFMpegWriter)
    monkeypatch.setattr(
        "stacksats.viz.animation_render.shutil.which",
        lambda name: "/usr/bin/ffmpeg",
    )
    writer = _resolve_writer(animation_mod, "mp4", fps=12)
    assert writer.codec == "libx264"
    assert "-pix_fmt" in writer.extra_args


def test_resolve_writer_uses_ffmpeg_for_webm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFFMpegWriter:
        def __init__(self, *, fps, codec, extra_args):
            self.fps = fps
            self.codec = codec
            self.extra_args = extra_args

    animation_mod = SimpleNamespace(PillowWriter=object, FFMpegWriter=FakeFFMpegWriter)
    monkeypatch.setattr(
        "stacksats.viz.animation_render.shutil.which",
        lambda name: "/usr/bin/ffmpeg",
    )
    writer = _resolve_writer(animation_mod, "webm", fps=12)
    assert writer.codec == "libvpx-vp9"


def test_storyboard_columns_include_updated_copy_defaults() -> None:
    enriched = _ensure_storyboard_columns(_frame_data(4))
    assert "window_start_label" in enriched.columns
    assert "window_end_label" in enriched.columns
    assert "window" in enriched.columns


def test_updated_animation_copy_is_present() -> None:
    content = Path("stacksats/viz/animation_render.py").read_text(encoding="utf-8")
    assert "Cumulative BTC vs Uniform (%)" in content
    assert "Per-Window Percentile" in content
    assert "Cumulative BTC vs uniform" in content
    assert "Final cumulative BTC vs uniform" in content
    assert "Top: cumulative outcome  |  Bottom: per-window percentile" in content
    assert "Windows {selected_windows}/{raw_windows} shown" in content
