from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import polars as pl
import pytest

from stacksats.api import BacktestResult, ValidationResult
from stacksats.strategy_types import BacktestConfig, ValidationConfig
from stacksats.strategies.catalog import StrategyCatalogEntry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_script_module(name: str):
    root = _repo_root()
    path = root / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare_strategies = _load_script_module("compare_strategies")


class _FakeStrategy:
    def __init__(
        self,
        *,
        strategy_id: str,
        intent_mode: str,
        win_rate: float,
        score: float,
        exp_decay_percentile: float,
        uniform_exp_decay_percentile: float = 50.0,
        validation_passed: bool = True,
    ) -> None:
        self._strategy_id = strategy_id
        self._intent_mode = intent_mode
        self._backtest_result = BacktestResult(
            spd_table=pl.DataFrame({"window_start": [], "window_end": []}),
            exp_decay_percentile=exp_decay_percentile,
            win_rate=win_rate,
            score=score,
            uniform_exp_decay_percentile=uniform_exp_decay_percentile,
            strategy_id=strategy_id,
            strategy_version="1.0.0",
        )
        self._validation_result = ValidationResult(
            passed=validation_passed,
            forward_leakage_ok=True,
            weight_constraints_ok=True,
            win_rate=win_rate,
            win_rate_ok=validation_passed,
            messages=[],
            strategy_id=strategy_id,
            min_win_rate=50.0,
            diagnostics={},
        )
        self.validation_calls: list[ValidationConfig] = []
        self.backtest_calls: list[BacktestConfig] = []

    def metadata(self):
        return SimpleNamespace(strategy_id=self._strategy_id)

    def spec(self):
        return SimpleNamespace(intent_mode=self._intent_mode)

    def validate(self, config: ValidationConfig):
        self.validation_calls.append(config)
        return self._validation_result

    def backtest(self, config: BacktestConfig):
        self.backtest_calls.append(config)
        return self._backtest_result


def _entry(
    strategy_id: str,
    *,
    tier: str = "stable",
    promotion_stage: str = "promoted",
) -> StrategyCatalogEntry:
    return StrategyCatalogEntry(
        strategy_id=strategy_id,
        strategy_spec=f"pkg.{strategy_id}:Strategy",
        class_name="Strategy",
        module_path=f"pkg.{strategy_id}",
        tier=tier,
        public_export=(tier == "stable"),
        audit_enabled=True,
        family="signals",
        description="test entry",
        docs_slug=strategy_id,
        tags=("test",),
        owner="StackSats Maintainers",
        benchmark_strategy_ids=("uniform",),
        promotion_stage=promotion_stage,
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={"start_date": "2018-01-01", "end_date": "2025-12-31"},
    )


def test_compare_strategies_derives_shared_defaults_for_builtins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategies = {
        "uniform": _FakeStrategy(
            strategy_id="uniform",
            intent_mode="propose",
            win_rate=50.0,
            score=50.0,
            exp_decay_percentile=50.0,
        ),
        "alpha": _FakeStrategy(
            strategy_id="alpha",
            intent_mode="profile",
            win_rate=61.0,
            score=63.0,
            exp_decay_percentile=64.0,
        ),
        "beta": _FakeStrategy(
            strategy_id="beta",
            intent_mode="profile",
            win_rate=55.0,
            score=58.0,
            exp_decay_percentile=57.0,
        ),
    }
    entries = {
        "uniform": _entry("uniform"),
        "alpha": _entry("alpha"),
        "beta": _entry("beta", tier="experimental", promotion_stage="candidate"),
    }
    backtests = {
        "uniform": BacktestConfig(start_date="2018-01-01", end_date="2025-12-31"),
        "alpha": BacktestConfig(start_date="2020-01-01", end_date="2025-12-31"),
        "beta": BacktestConfig(start_date="2019-01-01", end_date="2024-12-31"),
    }
    validations = {
        "uniform": ValidationConfig(min_win_rate=50.0, strict=False),
        "alpha": ValidationConfig(min_win_rate=55.0, strict=False),
        "beta": ValidationConfig(min_win_rate=60.0, strict=True),
    }

    monkeypatch.setattr(compare_strategies, "load_strategy", lambda selector: strategies[selector])
    monkeypatch.setattr(
        compare_strategies,
        "find_strategy_catalog_entry",
        lambda selector: entries.get(selector),
    )
    monkeypatch.setattr(
        compare_strategies,
        "backtest_config_for_strategy",
        lambda strategy_id: backtests[strategy_id],
    )
    monkeypatch.setattr(
        compare_strategies,
        "validation_config_for_strategy",
        lambda strategy_id: validations[strategy_id],
    )

    output_path = tmp_path / "compare.json"
    payload = compare_strategies.compare_strategies(
        selectors=["alpha", "beta"],
        output_path=output_path,
    )

    assert payload["baseline_selector"] == "uniform"
    assert payload["comparison_window"] == {
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "strict": True,
        "min_win_rate": 60.0,
    }
    assert [row["selector"] for row in payload["rows"]] == ["uniform", "alpha", "beta"]
    assert payload["rows"][2]["tier"] == "experimental"
    assert payload["rows"][2]["promotion_stage"] == "candidate"
    assert output_path.exists()

    for strategy in strategies.values():
        assert strategy.validation_calls[0].start_date == "2020-01-01"
        assert strategy.validation_calls[0].end_date == "2024-12-31"
        assert strategy.validation_calls[0].strict is True
        assert strategy.validation_calls[0].min_win_rate == 60.0
        assert strategy.backtest_calls[0].start_date == "2020-01-01"
        assert strategy.backtest_calls[0].end_date == "2024-12-31"


def test_compare_strategies_requires_explicit_dates_for_custom_selectors() -> None:
    with pytest.raises(ValueError, match="Custom strategy comparisons require explicit"):
        compare_strategies._resolve_bounds(
            ["uniform", "my_strategy.py:MyStrategy"],
            start_date=None,
            end_date=None,
            strict=None,
            min_win_rate=None,
        )


def test_compare_strategies_uses_explicit_dates_for_mixed_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategies = {
        "uniform": _FakeStrategy(
            strategy_id="uniform",
            intent_mode="propose",
            win_rate=50.0,
            score=50.0,
            exp_decay_percentile=50.0,
        ),
        "custom.py:CustomStrategy": _FakeStrategy(
            strategy_id="custom-strategy",
            intent_mode="profile",
            win_rate=58.0,
            score=59.0,
            exp_decay_percentile=61.0,
        ),
    }
    monkeypatch.setattr(compare_strategies, "load_strategy", lambda selector: strategies[selector])
    monkeypatch.setattr(
        compare_strategies,
        "find_strategy_catalog_entry",
        lambda selector: _entry("uniform") if selector == "uniform" else None,
    )
    monkeypatch.setattr(
        compare_strategies,
        "backtest_config_for_strategy",
        lambda strategy_id: BacktestConfig(start_date="2018-01-01", end_date="2025-12-31"),
    )
    monkeypatch.setattr(
        compare_strategies,
        "validation_config_for_strategy",
        lambda strategy_id: ValidationConfig(min_win_rate=50.0, strict=False),
    )

    payload = compare_strategies.compare_strategies(
        selectors=["custom.py:CustomStrategy"],
        start_date="2021-01-01",
        end_date="2021-12-31",
        output_path=tmp_path / "compare.json",
    )

    assert payload["comparison_window"] == {
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "strict": True,
        "min_win_rate": 50.0,
    }
    assert payload["rows"][1]["tier"] is None
    assert payload["rows"][1]["selector"] == "custom.py:CustomStrategy"


def test_render_table_includes_expected_headers() -> None:
    rendered = compare_strategies._render_table(
        [
            {
                "selector": "uniform",
                "intent_mode": "propose",
                "tier": "stable",
                "win_rate": 50.0,
                "score": 50.0,
                "exp_decay_percentile": 50.0,
                "multiple_vs_uniform": 1.0,
                "judgment_label": "baseline",
            }
        ]
    )

    assert "Selector" in rendered
    assert "Vs Uniform" in rendered
    assert "uniform" in rendered
