from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stacksats.animation_data import (
    load_backtest_payload,
    load_spd_table_from_backtest_json,
    prepare_animation_frame_data,
    spd_table_from_backtest_payload,
)


def _sample_spd_table() -> pl.DataFrame:
    return pl.DataFrame({
        "window": [
            "2024-01-03 → 2024-01-10",
            "2024-01-01 → 2024-01-08",
            "2024-01-02 → 2024-01-09",
        ],
        "dynamic_percentile": [55.0, float("inf"), 47.0],
        "uniform_percentile": [50.0, 52.0, None],
        "excess_percentile": [5.0, -2.0, -1.0],
        "dynamic_sats_per_dollar": [4200.0, 4100.0, None],
        "uniform_sats_per_dollar": [4000.0, 4200.0, 4300.0],
    })


def test_prepare_animation_frame_data_sorts_cleans_and_adds_running_metrics() -> None:
    prepared = prepare_animation_frame_data(_sample_spd_table(), max_frames=10)
    assert list(prepared["window_start"]) == sorted(prepared["window_start"])
    assert np.isfinite(prepared["dynamic_percentile"].to_numpy()).all()
    assert np.isfinite(prepared["uniform_percentile"].to_numpy()).all()
    assert np.isfinite(prepared["dynamic_sats_per_dollar"].to_numpy()).all()
    assert np.isclose(float(prepared["cumulative_excess"][-1]), 2.0)
    assert np.isclose(float(prepared["win_rate_to_date"][-1]), 33.33333333333333)
    cum_dynamic = float(prepared["cumulative_dynamic_sats_per_dollar"][-1])
    cum_uniform = float(prepared["cumulative_uniform_sats_per_dollar"][-1])
    expected_pct = (cum_dynamic / cum_uniform - 1.0) * 100.0
    assert np.isclose(
        float(prepared["cumulative_btc_vs_uniform_pct"][-1]),
        expected_pct,
    )


def test_prepare_animation_frame_data_supports_non_overlapping_mode() -> None:
    spd = pl.DataFrame({
        "window": [
            "2024-01-01 → 2024-01-10",
            "2024-01-05 → 2024-01-12",
            "2024-01-13 → 2024-01-20",
        ],
        "dynamic_percentile": [55.0, 54.0, 60.0],
        "uniform_percentile": [50.0, 50.0, 52.0],
        "excess_percentile": [5.0, 4.0, 8.0],
        "dynamic_sats_per_dollar": [4200.0, 4200.0, 4400.0],
        "uniform_sats_per_dollar": [4000.0, 4000.0, 4100.0],
    })
    prepared = prepare_animation_frame_data(spd, window_mode="non-overlapping")
    assert prepared.height == 2
    starts = prepared["window_start"]
    assert [d.strftime("%Y-%m-%d") for d in starts] == [
        "2024-01-01",
        "2024-01-13",
    ]


def test_prepare_animation_frame_data_downsampling_is_deterministic() -> None:
    from datetime import datetime, timedelta

    windows = []
    base = datetime(2024, 1, 1)
    for day in range(10):
        start = base + timedelta(days=day)
        end = start + timedelta(days=7)
        windows.append(f"{start.date().isoformat()} → {end.date().isoformat()}")
    spd = pl.DataFrame({
        "window": windows,
        "dynamic_percentile": np.linspace(45.0, 55.0, 10),
        "uniform_percentile": np.linspace(44.0, 54.0, 10),
        "excess_percentile": np.linspace(1.0, 1.0, 10),
        "dynamic_sats_per_dollar": np.linspace(4000.0, 4100.0, 10),
        "uniform_sats_per_dollar": np.linspace(3900.0, 4000.0, 10),
    })

    prepared_a = prepare_animation_frame_data(spd, max_frames=4)
    prepared_b = prepare_animation_frame_data(spd, max_frames=4)
    assert list(prepared_a["window_start"]) == list(prepared_b["window_start"])
    assert prepared_a.height == 4
    last_start = prepared_a["window_start"][-1]
    assert last_start.strftime("%Y-%m-%d") == "2024-01-10"


def test_prepare_animation_frame_data_raises_for_missing_required_columns() -> None:
    bad = pl.DataFrame({"window": ["2024-01-01 → 2024-01-08"]})
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_animation_frame_data(bad)


def test_prepare_animation_frame_data_raises_for_invalid_window_mode() -> None:
    with pytest.raises(ValueError, match="window_mode must be 'rolling' or 'non-overlapping'"):
        prepare_animation_frame_data(_sample_spd_table(), window_mode="invalid")


def test_spd_table_loaders_support_backtest_result_payload_shape(tmp_path: Path) -> None:
    payload = {
        "summary_metrics": {"score": 50.0, "win_rate": 55.0, "exp_decay_percentile": 60.0},
        "window_level_data": [
            {
                "index": "2024-01-01 → 2024-01-08",
                "dynamic_percentile": 55.0,
                "uniform_percentile": 50.0,
                "excess_percentile": 5.0,
                "dynamic_sats_per_dollar": 4100.0,
                "uniform_sats_per_dollar": 3900.0,
            }
        ],
    }
    json_path = tmp_path / "backtest_result.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded_payload = load_backtest_payload(json_path)
    frame_from_payload = spd_table_from_backtest_payload(loaded_payload)
    frame_from_json = load_spd_table_from_backtest_json(json_path)
    assert list(frame_from_payload.columns) == list(frame_from_json.columns)
    prepared = prepare_animation_frame_data(frame_from_json)
    assert prepared.height == 1
    assert prepared["window_start"][0].strftime("%Y-%m-%d") == "2024-01-01"


def test_load_backtest_payload_raises_for_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid backtest JSON"):
        load_backtest_payload(bad_path)
