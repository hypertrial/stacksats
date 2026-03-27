from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stacksats import BacktestConfig, ExportConfig, UniformStrategy
import stacksats.animation_render as animation_render
from tests.test_helpers import btc_frame

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "snapshots" / "public_contract_snapshots.json"


def _sample_btc_df():
    return btc_frame(start="2022-01-01", days=520, price_start=20000.0, price_step=50.0).with_columns(
        mvrv=np.linspace(0.8, 2.2, 520)
    )


def test_public_artifact_contract_snapshots(tmp_path) -> None:
    snapshots = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["artifacts"]
    result = UniformStrategy().backtest(
        BacktestConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            strategy_label="uniform-test",
        ),
        btc_df=_sample_btc_df(),
    )

    backtest_payload = result.to_json()
    assert sorted(backtest_payload.keys()) == snapshots["backtest_result"]["top_level"]
    assert sorted(backtest_payload["provenance"].keys()) == snapshots["backtest_result"]["provenance"]
    assert (
        sorted(backtest_payload["summary_metrics"].keys())
        == snapshots["backtest_result"]["summary_metrics"]
    )
    assert (
        sorted(backtest_payload["window_level_data"][0].keys())
        == snapshots["backtest_result"]["window_level_row"]
    )

    metrics_path = Path(result.plot(output_dir=str(tmp_path))["metrics_json"])
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert sorted(metrics_payload.keys()) == snapshots["metrics"]["top_level"]
    assert sorted(metrics_payload["summary_metrics"].keys()) == snapshots["metrics"]["summary_metrics"]
    assert (
        sorted(metrics_payload["window_level_data"][0].keys())
        == snapshots["metrics"]["window_level_row"]
    )

    def _fake_render(frame_data, gif_path, fps, width, height):
        Path(gif_path).write_text("gif", encoding="utf-8")
        return {"frames": frame_data.height, "fps": fps, "width": width, "height": height}

    original_render = animation_render.render_strategy_vs_uniform_gif
    animation_render.render_strategy_vs_uniform_gif = _fake_render
    try:
        manifest_path = Path(
            result.animate(
                output_dir=str(tmp_path),
                source_backtest_json=tmp_path / "backtest_result.json",
            )["manifest_json"]
        )
    finally:
        animation_render.render_strategy_vs_uniform_gif = original_render

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sorted(manifest_payload.keys()) == snapshots["animation_manifest"]["top_level"]

    UniformStrategy().export(
        ExportConfig(
            range_start="2022-01-01",
            range_end="2022-12-31",
            output_dir=str(tmp_path),
        ),
        btc_df=_sample_btc_df(),
    )
    export_payload = json.loads(next(tmp_path.glob("**/artifacts.json")).read_text(encoding="utf-8"))
    assert sorted(export_payload.keys()) == snapshots["export_artifacts"]["top_level"]
    assert sorted(export_payload["files"].keys()) == snapshots["export_artifacts"]["files"]
