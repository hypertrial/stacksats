"""Tests for StrategyRunner.compare and comparison result types."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from stacksats import (
    ComparisonConfig,
    ComparisonResult,
    ComparisonRow,
    SimpleZScoreStrategy,
    StrategyRunner,
    UniformStrategy,
)
from tests.test_helpers import btc_frame


def _btc_df(*, days: int = 400):
    return btc_frame(
        start="2022-01-01",
        days=days,
        price_start=20000.0,
        price_step=50.0,
    ).with_columns(mvrv=np.linspace(0.8, 2.2, days))


def test_compare_empty_strategies_raises() -> None:
    with pytest.raises(ValueError, match="at least one strategy"):
        StrategyRunner().compare([], ComparisonConfig(start_date="2022-01-01", end_date="2023-01-01"))


def test_compare_baseline_not_in_set_raises() -> None:
    with pytest.raises(ValueError, match="Baseline strategy_id"):
        StrategyRunner().compare(
            [SimpleZScoreStrategy()],
            ComparisonConfig(
                start_date="2022-01-01",
                end_date="2023-01-01",
                baseline="uniform",
            ),
        )


def test_compare_selectors_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="selectors length must match"):
        StrategyRunner().compare(
            [UniformStrategy()],
            ComparisonConfig(
                start_date="2022-01-01",
                end_date="2023-01-01",
                baseline="uniform",
            ),
            selectors=["a", "b"],
        )


def test_comparison_result_to_json_from_json_roundtrip() -> None:
    row = ComparisonRow(
        selector="uniform",
        strategy_id="uniform",
        strategy_version="1.0.0",
        intent_mode="propose",
        tier="stable",
        promotion_stage="promoted",
        validation_passed=True,
        judgment_label="validation-passed",
        win_rate=50.0,
        score=55.0,
        exp_decay_percentile=40.0,
        multiple_vs_uniform=1.05,
        score_delta_vs_baseline=0.0,
        exp_decay_delta_vs_baseline=0.0,
        is_baseline=True,
    )
    original = ComparisonResult(
        baseline_selector="uniform",
        comparison_window={
            "start_date": "2022-01-01",
            "end_date": "2023-01-01",
            "strict": True,
            "min_win_rate": 50.0,
        },
        rows=[row],
        run_id="r1",
        config_hash="abc",
        artifact_path=None,
    )
    payload = original.to_json()
    assert payload["schema_version"] == "1.0.0"
    restored = ComparisonResult.from_json(payload)
    assert restored.baseline_selector == original.baseline_selector
    assert len(restored.rows) == 1
    assert restored.rows[0].strategy_id == "uniform"
    assert restored.rows[0].is_baseline is True


def test_comparison_result_render_table_and_dataframe() -> None:
    rows = [
        ComparisonRow(
            selector="uniform",
            strategy_id="uniform",
            strategy_version="1.0.0",
            intent_mode="propose",
            tier="stable",
            promotion_stage="promoted",
            validation_passed=True,
            judgment_label="validation-passed",
            win_rate=50.0,
            score=55.0,
            exp_decay_percentile=40.0,
            multiple_vs_uniform=1.0,
            score_delta_vs_baseline=0.0,
            exp_decay_delta_vs_baseline=0.0,
            is_baseline=True,
        ),
        ComparisonRow(
            selector="simple-zscore",
            strategy_id="simple-zscore",
            strategy_version="1.0.0",
            intent_mode="propose",
            tier="stable",
            promotion_stage="promoted",
            validation_passed=True,
            judgment_label="validation-passed",
            win_rate=51.0,
            score=56.0,
            exp_decay_percentile=41.0,
            multiple_vs_uniform=1.02,
            score_delta_vs_baseline=1.0,
            exp_decay_delta_vs_baseline=1.0,
            is_baseline=False,
        ),
    ]
    result = ComparisonResult(
        baseline_selector="uniform",
        comparison_window={},
        rows=rows,
    )
    text = result.render_table()
    assert "Selector" in text
    assert "uniform" in text
    assert "simple-zscore" in text
    df = result.to_dataframe()
    assert df.height == 2
    assert "score_delta_vs_baseline" in df.columns


@pytest.mark.slow
def test_runner_compare_uniform_and_simple_zscore(tmp_path) -> None:
    btc = _btc_df(days=450)
    runner = StrategyRunner()
    result = runner.compare(
        [UniformStrategy(), SimpleZScoreStrategy()],
        ComparisonConfig(
            start_date="2022-01-01",
            end_date="2023-03-01",
            baseline="uniform",
            strict=False,
            min_win_rate=0.0,
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
    )
    assert len(result.rows) == 2
    baseline_rows = [r for r in result.rows if r.is_baseline]
    assert len(baseline_rows) == 1
    assert baseline_rows[0].strategy_id == "uniform"
    assert baseline_rows[0].score_delta_vs_baseline == 0.0
    assert baseline_rows[0].exp_decay_delta_vs_baseline == 0.0
    other = next(r for r in result.rows if not r.is_baseline)
    assert other.strategy_id == "simple-zscore"
    assert result.artifact_path is not None
    path = tmp_path / "uniform" / "comparison" / result.run_id / "comparison_result.json"
    assert path.is_file()
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["schema_version"] == "1.0.0"
    assert len(disk["rows"]) == 2


@pytest.mark.slow
def test_runner_compare_preserves_selectors_on_rows(tmp_path) -> None:
    btc = _btc_df(days=450)
    selectors = ["alias-uniform", "alias-zscore"]
    result = StrategyRunner().compare(
        [UniformStrategy(), SimpleZScoreStrategy()],
        ComparisonConfig(
            start_date="2022-01-01",
            end_date="2023-03-01",
            baseline="uniform",
            strict=False,
            min_win_rate=0.0,
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
        selectors=selectors,
    )
    assert len(result.rows) == 2
    by_id = {r.strategy_id: r for r in result.rows}
    assert by_id["uniform"].selector == "alias-uniform"
    assert by_id["simple-zscore"].selector == "alias-zscore"


def test_compare_to_benchmarks_requires_catalog_entry() -> None:
    class _Custom(UniformStrategy):
        strategy_id = "custom-xyz-compare-test"
        version = "0.0.1"

    with pytest.raises(ValueError, match="catalog"):
        _Custom().compare_to_benchmarks(
            ComparisonConfig(start_date="2022-01-01", end_date="2023-01-01")
        )


def test_compare_to_benchmarks_raises_when_no_benchmarks_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stacksats.strategies.catalog.find_strategy_catalog_entry",
        lambda _sid: SimpleNamespace(strategy_id="uniform", benchmark_strategy_ids=()),
    )
    with pytest.raises(ValueError, match="no benchmark_strategy_ids"):
        UniformStrategy().compare_to_benchmarks(
            ComparisonConfig(start_date="2022-01-01", end_date="2023-01-01")
        )


@pytest.mark.slow
def test_compare_to_benchmarks_simple_zscore(tmp_path) -> None:
    btc = _btc_df(days=450)
    result = SimpleZScoreStrategy().compare_to_benchmarks(
        ComparisonConfig(
            start_date="2022-01-01",
            end_date="2023-03-01",
            baseline="uniform",
            strict=False,
            min_win_rate=0.0,
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
    )
    ids = {r.strategy_id for r in result.rows}
    assert ids == {"simple-zscore", "uniform", "mvrv"}
