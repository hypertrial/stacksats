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


def _patch_runner_compare_stub(
    monkeypatch: pytest.MonkeyPatch,
    compare_module,
) -> None:
    """Avoid real StrategyRunner.compare (needs BaseStrategy) when using _FakeStrategy."""

    from stacksats.api import ComparisonResult, ComparisonRow

    def stub_compare(_self, strategies, config, *, btc_df=None, selectors=None):
        assert config.start_date is not None and config.end_date is not None
        vc = ValidationConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            strict=config.strict,
            min_win_rate=config.min_win_rate,
        )
        baseline_score: float | None = None
        baseline_exp_decay: float | None = None
        raw_rows: list[dict[str, object]] = []
        for idx, strategy in enumerate(strategies):
            md = strategy.metadata()
            strategy_version = str(getattr(md, "version", None) or "1.0.0")
            sel = selectors[idx] if selectors is not None else str(md.strategy_id)
            catalog_entry = compare_module.find_strategy_catalog_entry(md.strategy_id)
            val = strategy.validate(vc, btc_df=btc_df)
            bt = strategy.backtest(
                BacktestConfig(
                    start_date=config.start_date,
                    end_date=config.end_date,
                    strategy_label=str(md.strategy_id),
                ),
                btc_df=btc_df,
            )
            if str(md.strategy_id) == config.baseline:
                baseline_score = float(bt.score)
                baseline_exp_decay = float(bt.exp_decay_percentile)
            diag = dict(val.diagnostics or {})
            judgment = diag.get("judgment")
            if not isinstance(judgment, str) or not judgment:
                judgment = "validation-passed" if val.passed else "validation-failed"
            raw_rows.append(
                {
                    "selector": sel,
                    "strategy_id": str(md.strategy_id),
                    "strategy_version": strategy_version,
                    "intent_mode": strategy.spec().intent_mode,
                    "tier": catalog_entry.tier if catalog_entry is not None else None,
                    "promotion_stage": (
                        catalog_entry.promotion_stage if catalog_entry is not None else None
                    ),
                    "validation_passed": bool(val.passed),
                    "judgment_label": judgment,
                    "win_rate": float(bt.win_rate),
                    "score": float(bt.score),
                    "exp_decay_percentile": float(bt.exp_decay_percentile),
                    "multiple_vs_uniform": bt.exp_decay_multiple_vs_uniform,
                    "is_baseline": str(md.strategy_id) == config.baseline,
                }
            )
        assert baseline_score is not None and baseline_exp_decay is not None
        rows: list[ComparisonRow] = []
        for r in raw_rows:
            rows.append(
                ComparisonRow(
                    selector=str(r["selector"]),
                    strategy_id=str(r["strategy_id"]),
                    strategy_version=str(r["strategy_version"]),
                    intent_mode=str(r["intent_mode"]),
                    tier=r["tier"] if r["tier"] is not None else None,
                    promotion_stage=(
                        str(r["promotion_stage"])
                        if r["promotion_stage"] is not None
                        else None
                    ),
                    validation_passed=bool(r["validation_passed"]),
                    judgment_label=str(r["judgment_label"]),
                    win_rate=float(r["win_rate"]),
                    score=float(r["score"]),
                    exp_decay_percentile=float(r["exp_decay_percentile"]),
                    multiple_vs_uniform=(
                        float(r["multiple_vs_uniform"])
                        if r["multiple_vs_uniform"] is not None
                        else None
                    ),
                    score_delta_vs_baseline=float(r["score"]) - baseline_score,
                    exp_decay_delta_vs_baseline=(
                        float(r["exp_decay_percentile"]) - baseline_exp_decay
                    ),
                    is_baseline=bool(r["is_baseline"]),
                )
            )
        return ComparisonResult(
            baseline_selector=config.baseline,
            comparison_window={
                "start_date": config.start_date,
                "end_date": config.end_date,
                "strict": config.strict,
                "min_win_rate": config.min_win_rate,
            },
            rows=rows,
            run_id="stub-run",
            config_hash="stub",
            artifact_path=None,
        )

    monkeypatch.setattr("stacksats.runner.core.StrategyRunner.compare", stub_compare)


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
        self.validation_kwargs: list[dict[str, object]] = []
        self.backtest_kwargs: list[dict[str, object]] = []

    def metadata(self):
        return SimpleNamespace(strategy_id=self._strategy_id)

    def spec(self):
        return SimpleNamespace(intent_mode=self._intent_mode)

    def validate(self, config: ValidationConfig, **kwargs):
        self.validation_calls.append(config)
        self.validation_kwargs.append(dict(kwargs))
        return self._validation_result

    def backtest(self, config: BacktestConfig, **kwargs):
        self.backtest_calls.append(config)
        self.backtest_kwargs.append(dict(kwargs))
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

    monkeypatch.setattr(
        compare_strategies,
        "load_strategy",
        lambda selector, config_path=None: strategies[selector],
    )
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
    _patch_runner_compare_stub(monkeypatch, compare_strategies)

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
    monkeypatch.setattr(
        compare_strategies,
        "load_strategy",
        lambda selector, config_path=None: strategies[selector],
    )
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
    _patch_runner_compare_stub(monkeypatch, compare_strategies)

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


def test_compare_strategies_supports_selector_config_paths_and_btc_df(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uniform_strategy = _FakeStrategy(
        strategy_id="uniform",
        intent_mode="propose",
        win_rate=50.0,
        score=50.0,
        exp_decay_percentile=50.0,
    )
    strategy = _FakeStrategy(
        strategy_id="custom-strategy",
        intent_mode="profile",
        win_rate=52.0,
        score=53.0,
        exp_decay_percentile=54.0,
    )
    loaded: list[tuple[str, str | None]] = []
    btc_df = pl.DataFrame({"date": [], "price_usd": []})

    def _fake_load(selector: str, *, config_path: str | None = None):
        loaded.append((selector, config_path))
        return uniform_strategy if selector == "uniform" else strategy

    monkeypatch.setattr(compare_strategies, "load_strategy", _fake_load)
    monkeypatch.setattr(compare_strategies, "find_strategy_catalog_entry", lambda selector: None)
    _patch_runner_compare_stub(monkeypatch, compare_strategies)

    payload = compare_strategies.compare_strategies(
        selectors=["custom.py:CustomStrategy"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        strict=True,
        min_win_rate=0.0,
        output_path=None,
        btc_df=btc_df,
        selector_config_paths={"custom.py:CustomStrategy": str(tmp_path / "strategy.json")},
    )

    assert payload["rows"][1]["selector"] == "custom.py:CustomStrategy"
    assert loaded == [
        ("uniform", None),
        ("custom.py:CustomStrategy", str(tmp_path / "strategy.json")),
    ]
    assert uniform_strategy.validation_kwargs[-1]["btc_df"] is btc_df
    assert strategy.validation_kwargs[-1]["btc_df"] is btc_df
    assert uniform_strategy.backtest_kwargs[-1]["btc_df"] is btc_df
    assert strategy.backtest_kwargs[-1]["btc_df"] is btc_df
