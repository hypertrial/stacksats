from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stacksats import BacktestConfig, ExportConfig, UniformStrategy
import stacksats.viz.animation_render as animation_render
from stacksats.api import ComparisonResult, ComparisonRow, DailyDecisionResult
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

    def _fake_render(frame_data, gif_path, fps, width, height, **kwargs):
        del kwargs
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

    decision_payload = DailyDecisionResult(
        status="decided",
        strategy_id="uniform",
        strategy_version="1.0.0",
        run_date="2024-12-31",
        decision_key="decision-123",
        idempotency_hit=False,
        forced_rerun=False,
        weight_today=0.05,
        recommended_notional_usd=50.0,
        recommended_quantity_btc=0.001,
        reference_price_usd=50000.0,
        btc_price_col="price_usd",
        state_db_path=str(tmp_path / "state.sqlite3"),
        artifact_path=str(tmp_path / "decision_result.json"),
        message="Daily decision completed.",
        validation_receipt_id=7,
        validation_passed=True,
        data_hash="data-hash",
        feature_snapshot_hash="feature-hash",
    ).to_json()
    assert sorted(decision_payload.keys()) == snapshots["decision_result"]["top_level"]

    compare_payload = ComparisonResult(
        baseline_selector="uniform",
        comparison_window={
            "start_date": "2022-01-01",
            "end_date": "2023-01-01",
            "strict": True,
            "min_win_rate": 50.0,
        },
        rows=[
            ComparisonRow(
                selector="uniform",
                strategy_id="uniform",
                strategy_version="1.0.0",
                intent_mode="propose",
                tier="stable",
                promotion_stage="promoted",
                validation_passed=True,
                judgment_label="validation-passed",
                win_rate=55.0,
                score=60.0,
                exp_decay_percentile=50.0,
                multiple_vs_uniform=1.1,
                score_delta_vs_baseline=0.0,
                exp_decay_delta_vs_baseline=0.0,
                is_baseline=True,
            )
        ],
        run_id="compare-run",
        config_hash="abc123",
        artifact_path=str(tmp_path / "comparison_result.json"),
    ).to_json()
    assert sorted(compare_payload.keys()) == snapshots["comparison_result"]["top_level"]
    assert (
        sorted(compare_payload["comparison_window"].keys())
        == snapshots["comparison_result"]["comparison_window"]
    )
    assert sorted(compare_payload["rows"][0].keys()) == snapshots["comparison_result"]["row"]

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
