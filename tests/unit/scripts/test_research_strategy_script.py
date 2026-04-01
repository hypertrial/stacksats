from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import polars as pl
import pytest

from stacksats.api import BacktestResult, ValidationResult


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


research_strategy = _load_script_module("research_strategy")


class _FakeStrategy:
    def __init__(self) -> None:
        self.validation_calls: list[tuple[object, dict[str, object]]] = []
        self.backtest_calls: list[tuple[object, dict[str, object]]] = []

    def metadata(self):
        return SimpleNamespace(strategy_id="my-strategy", version="1.2.3")

    def spec(self):
        return SimpleNamespace(intent_mode="profile")

    def validate(self, config, **kwargs):
        self.validation_calls.append((config, dict(kwargs)))
        return ValidationResult(
            passed=True,
            forward_leakage_ok=True,
            weight_constraints_ok=True,
            win_rate=61.0,
            win_rate_ok=True,
            messages=[],
            strategy_id="my-strategy",
            min_win_rate=float(config.min_win_rate),
            diagnostics={"judgment": "validation-passed"},
        )

    def backtest(self, config, **kwargs):
        self.backtest_calls.append((config, dict(kwargs)))
        return BacktestResult(
            spd_table=pl.DataFrame({"window_start": [], "window_end": []}),
            exp_decay_percentile=62.0,
            win_rate=61.0,
            score=63.0,
            uniform_exp_decay_percentile=50.0,
            strategy_id="my-strategy",
            strategy_version="1.2.3",
            config_hash="cfg123",
            run_id="run123",
        )


def test_research_strategy_main_runs_canonical_flow_and_writes_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    strategy = _FakeStrategy()
    loaded: list[tuple[str, str | None]] = []

    def _fake_load(selector: str, *, config_path: str | None = None):
        loaded.append((selector, config_path))
        return strategy

    config_path = tmp_path / "strategy.json"
    config_path.write_text('{"value_weight": 0.65}', encoding="utf-8")
    output_path = tmp_path / "research.json"

    monkeypatch.setattr(research_strategy, "load_strategy", _fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_strategy.py",
            "--strategy",
            "my_strategy.py:MyStrategy",
            "--strategy-config",
            str(config_path),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-12-31",
            "--output-path",
            str(output_path),
        ],
    )

    exit_code = research_strategy.main()

    assert exit_code == 0
    assert loaded == [("my_strategy.py:MyStrategy", str(config_path))]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["resolved_strategy_id"] == "my-strategy"
    assert payload["window"]["strict"] is True
    assert payload["comparison"] is None
    stdout = capsys.readouterr().out
    assert "Validation PASSED" in stdout
    assert "Saved" in stdout


def test_research_strategy_uses_from_dataframe_and_column_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategy = _FakeStrategy()
    source_df = pl.DataFrame(
        {
            "date": [pl.datetime(2024, 1, 1, 0, 0, 0, time_unit="us")],
            "Close": [42000.0],
            "MVRV_Ratio": [1.2],
        }
    )
    column_map_path = tmp_path / "column_map.json"
    column_map_path.write_text(
        json.dumps({"price_usd": "Close", "mvrv": "MVRV_Ratio"}),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    class _FakeRunner:
        def validate(self, strategy_arg, config):
            observed["validated_with"] = strategy_arg
            observed["validation_strict"] = config.strict
            return strategy.validate(config)

        def backtest(self, strategy_arg, config):
            observed["backtested_with"] = strategy_arg
            return strategy.backtest(config)

    def _fake_from_dataframe(df, *, column_map=None):
        observed["dataframe"] = df
        observed["column_map"] = dict(column_map or {})
        return _FakeRunner()

    monkeypatch.setattr(research_strategy, "load_strategy", lambda selector, config_path=None: strategy)
    monkeypatch.setattr(research_strategy.pl, "read_parquet", lambda path: source_df)
    monkeypatch.setattr(research_strategy.StrategyRunner, "from_dataframe", _fake_from_dataframe)

    payload = research_strategy.build_research_payload(
        strategy_selector="my_strategy.py:MyStrategy",
        strategy_config_path=None,
        dataframe_parquet=str(tmp_path / "custom.parquet"),
        column_map_config=str(column_map_path),
        start_date="2024-01-01",
        end_date="2024-12-31",
        min_win_rate=0.0,
        strict=True,
        baseline="uniform",
        compare_strategies_selectors=[],
    )

    assert observed["dataframe"] is source_df
    assert observed["column_map"] == {"price_usd": "Close", "mvrv": "MVRV_Ratio"}
    assert observed["validated_with"] is strategy
    assert payload["data_source"]["mode"] == "dataframe-parquet"


def test_research_strategy_compare_mode_uses_explicit_shared_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategy = _FakeStrategy()
    config_path = tmp_path / "strategy.json"
    config_path.write_text('{"value_weight": 0.65}', encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_compare(**kwargs):
        captured.update(kwargs)
        return {
            "baseline_selector": kwargs["baseline"],
            "comparison_window": {
                "start_date": kwargs["start_date"],
                "end_date": kwargs["end_date"],
                "strict": kwargs["strict"],
                "min_win_rate": kwargs["min_win_rate"],
            },
            "rows": [],
        }

    monkeypatch.setattr(research_strategy, "load_strategy", lambda selector, config_path=None: strategy)
    monkeypatch.setattr(research_strategy.compare_strategies, "compare_strategies", _fake_compare)

    payload = research_strategy.build_research_payload(
        strategy_selector="my_strategy.py:MyStrategy",
        strategy_config_path=str(config_path),
        dataframe_parquet=None,
        column_map_config=None,
        start_date="2024-01-01",
        end_date="2024-12-31",
        min_win_rate=0.0,
        strict=True,
        baseline="uniform",
        compare_strategies_selectors=["simple-zscore", "mvrv"],
    )

    assert captured["selectors"] == ["my_strategy.py:MyStrategy", "simple-zscore", "mvrv"]
    assert captured["start_date"] == "2024-01-01"
    assert captured["end_date"] == "2024-12-31"
    assert captured["strict"] is True
    assert captured["min_win_rate"] == 0.0
    assert captured["output_path"] is None
    assert captured["selector_config_paths"] == {
        "my_strategy.py:MyStrategy": str(config_path),
    }
    assert payload["comparison"]["comparison_window"]["strict"] is True


def test_research_strategy_dataframe_compare_passes_canonical_btc_df(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategy = _FakeStrategy()
    source_df = pl.DataFrame(
        {
            "date": [pl.datetime(2024, 1, 1, 0, 0, 0, time_unit="us")],
            "Close": [42000.0],
            "MVRV_Ratio": [1.2],
        }
    )
    column_map_path = tmp_path / "column_map.json"
    column_map_path.write_text(
        json.dumps({"price_usd": "Close", "mvrv": "MVRV_Ratio"}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class _FakeRunner:
        def validate(self, strategy_arg, config):
            return strategy.validate(config)

        def backtest(self, strategy_arg, config):
            return strategy.backtest(config)

    monkeypatch.setattr(research_strategy, "load_strategy", lambda selector, config_path=None: strategy)
    monkeypatch.setattr(research_strategy.pl, "read_parquet", lambda path: source_df)
    monkeypatch.setattr(
        research_strategy.StrategyRunner,
        "from_dataframe",
        lambda df, column_map=None: _FakeRunner(),
    )

    def _fake_compare(**kwargs):
        captured.update(kwargs)
        return {"baseline_selector": "uniform", "comparison_window": {}, "rows": []}

    monkeypatch.setattr(research_strategy.compare_strategies, "compare_strategies", _fake_compare)

    research_strategy.build_research_payload(
        strategy_selector="my_strategy.py:MyStrategy",
        strategy_config_path=None,
        dataframe_parquet=str(tmp_path / "custom.parquet"),
        column_map_config=str(column_map_path),
        start_date="2024-01-01",
        end_date="2024-12-31",
        min_win_rate=0.0,
        strict=True,
        baseline="uniform",
        compare_strategies_selectors=["simple-zscore"],
    )

    assert captured["btc_df"].columns == ["date", "price_usd", "mvrv"]


def test_research_strategy_rejects_column_map_without_dataframe() -> None:
    with pytest.raises(ValueError, match="--column-map-config requires --dataframe-parquet"):
        research_strategy.build_research_payload(
            strategy_selector="my_strategy.py:MyStrategy",
            strategy_config_path=None,
            dataframe_parquet=None,
            column_map_config="column_map.json",
            start_date="2024-01-01",
            end_date="2024-12-31",
            min_win_rate=0.0,
            strict=True,
            baseline="uniform",
            compare_strategies_selectors=[],
        )
