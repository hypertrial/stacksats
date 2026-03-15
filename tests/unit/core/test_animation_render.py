from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stacksats.animation_render import render_strategy_vs_uniform_gif


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
    assert meta["frames"] == 5
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
