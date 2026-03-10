from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stacksats.animation_data import (
    load_backtest_payload,
    load_spd_table_from_backtest_json,
    prepare_animation_frame_data,
    spd_table_from_backtest_payload,
)


def _sample_spd_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window": [
                "2024-01-03 → 2024-01-10",
                "2024-01-01 → 2024-01-08",
                "2024-01-02 → 2024-01-09",
            ],
            "dynamic_percentile": [55.0, np.inf, 47.0],
            "uniform_percentile": [50.0, 52.0, np.nan],
            "excess_percentile": [5.0, -2.0, -1.0],
            "dynamic_sats_per_dollar": [4200.0, 4100.0, np.nan],
            "uniform_sats_per_dollar": [4000.0, 4200.0, 4300.0],
        }
    )


def test_prepare_animation_frame_data_sorts_cleans_and_adds_running_metrics() -> None:
    prepared = prepare_animation_frame_data(_sample_spd_table(), max_frames=10)
    assert list(prepared["window_start"]) == sorted(prepared["window_start"])
    assert np.isfinite(prepared["dynamic_percentile"]).all()
    assert np.isfinite(prepared["uniform_percentile"]).all()
    assert np.isfinite(prepared["dynamic_sats_per_dollar"]).all()
    assert np.isclose(float(prepared["cumulative_excess"].iloc[-1]), 2.0)
    assert np.isclose(float(prepared["win_rate_to_date"].iloc[-1]), 33.33333333333333)


def test_prepare_animation_frame_data_supports_non_overlapping_mode() -> None:
    spd = pd.DataFrame(
        {
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
        }
    )
    prepared = prepare_animation_frame_data(spd, window_mode="non-overlapping")
    assert len(prepared) == 2
    assert list(prepared["window_start"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-13",
    ]


def test_prepare_animation_frame_data_downsampling_is_deterministic() -> None:
    windows = []
    for day in range(10):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        end = start + pd.Timedelta(days=7)
        windows.append(f"{start.date().isoformat()} → {end.date().isoformat()}")
    spd = pd.DataFrame(
        {
            "window": windows,
            "dynamic_percentile": np.linspace(45.0, 55.0, 10),
            "uniform_percentile": np.linspace(44.0, 54.0, 10),
            "excess_percentile": np.linspace(1.0, 1.0, 10),
            "dynamic_sats_per_dollar": np.linspace(4000.0, 4100.0, 10),
            "uniform_sats_per_dollar": np.linspace(3900.0, 4000.0, 10),
        }
    )

    prepared_a = prepare_animation_frame_data(spd, max_frames=4)
    prepared_b = prepare_animation_frame_data(spd, max_frames=4)
    assert list(prepared_a["window_start"]) == list(prepared_b["window_start"])
    assert len(prepared_a) == 4
    assert prepared_a["window_start"].iloc[-1] == pd.Timestamp("2024-01-10")


def test_prepare_animation_frame_data_raises_for_missing_required_columns() -> None:
    bad = pd.DataFrame({"window": ["2024-01-01 → 2024-01-08"]})
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
    assert len(prepared) == 1
    assert prepared["window_start"].iloc[0] == pd.Timestamp("2024-01-01")


def test_load_backtest_payload_raises_for_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid backtest JSON"):
        load_backtest_payload(bad_path)
