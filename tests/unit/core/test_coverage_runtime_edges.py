from __future__ import annotations

import builtins
import datetime as dt
import importlib
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl
import pytest

from stacksats.api import BacktestResult, DailyRunResult, ValidationResult
from stacksats.execution_adapters import load_execution_adapter
from stacksats.export_weights_runtime import insert_all_data
from stacksats.execution_state import IdempotencyConflictError, SQLiteExecutionStateStore
from stacksats.prelude import compute_cycle_spd
from stacksats.runner import StrategyRunner
from stacksats.runner_validation import _ValidationState
from stacksats.strategy_types import BaseStrategy, RunDailyConfig, ValidationConfig


class _UniformStrategy(BaseStrategy):
    strategy_id = "coverage-uniform"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


def _btc_df(days: int = 900) -> pl.DataFrame:
    dates = pl.datetime_range(
        dt.datetime(2023, 1, 1),
        dt.datetime(2023, 1, 1) + dt.timedelta(days=days - 1),
        interval="1d",
        eager=True,
    ).to_list()
    return pl.DataFrame(
        {
            "date": dates,
            "price_usd": np.linspace(20000.0, 80000.0, len(dates)),
            "mvrv": np.linspace(1.0, 2.0, len(dates)),
        }
    )


def _allow_validation(runner: StrategyRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "validate",
        lambda *args, **kwargs: ValidationResult(
            passed=True,
            forward_leakage_ok=True,
            weight_constraints_ok=True,
            win_rate=100.0,
            win_rate_ok=True,
            messages=["ok"],
            diagnostics={},
        ),
    )


def test_api_and_backtest_uncovered_branches(tmp_path: Path, monkeypatch) -> None:
    result = BacktestResult(
        spd_table=pl.DataFrame({"dynamic_percentile": [1.0], "uniform_percentile": [1.0]}),
        exp_decay_percentile=float("inf"),
        uniform_exp_decay_percentile=1.0,
        win_rate=0.0,
        score=0.0,
    )
    assert result.exp_decay_multiple_vs_uniform is None

    daily = DailyRunResult(
        status="executed",
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-01",
        run_key="rk",
        mode="paper",
        idempotency_hit=False,
        forced_rerun=False,
        weight_today=0.1,
        order_notional_usd=100.0,
        btc_quantity=0.001,
        price_usd=50000.0,
        adapter_name="paper",
        state_db_path=".stacksats/run_state.sqlite3",
        artifact_path=None,
        message="ok",
    )
    assert "Daily Run EXECUTED" in daily.summary()

    from stacksats.backtest import export_metrics_json

    export_metrics_json(
        pl.DataFrame({"window": ["2024-01-01 → 2024-12-31"], "dynamic_percentile": [1.0], "uniform_percentile": [1.0], "excess_percentile": [0.0], "dynamic_sats_per_dollar": [1e6], "uniform_sats_per_dollar": [1e6], "min_sats_per_dollar": [1e6], "max_sats_per_dollar": [1e6]}),
        {
            "score": 1.0,
            "win_rate": 1.0,
            "exp_decay_percentile": 1.0,
            "uniform_exp_decay_percentile": 1.0,
            "exp_decay_multiple_vs_uniform": None,
            "mean_excess": 0.0,
            "median_excess": 0.0,
            "relative_improvement_pct_mean": 0.0,
            "relative_improvement_pct_median": 0.0,
            "mean_ratio": 1.0,
            "median_ratio": 1.0,
            "total_windows": 1,
            "wins": 0,
            "losses": 1,
        },
        output_dir=str(tmp_path),
    )


def test_execution_adapter_error_branches(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="Invalid adapter spec"):
        load_execution_adapter("bad-spec")

    with pytest.raises(FileNotFoundError):
        load_execution_adapter(f"{tmp_path / 'missing.py'}:Adapter")

    no_class = tmp_path / "no_class.py"
    no_class.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="not found"):
        load_execution_adapter(f"{no_class}:MissingAdapter")

    bad_impl = tmp_path / "bad_impl.py"
    bad_impl.write_text("class Bad:\n    submit_order = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="callable submit_order"):
        load_execution_adapter(f"{bad_impl}:Bad")

    boom_module = tmp_path / "boom.py"
    boom_module.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        load_execution_adapter(f"{boom_module}:Any")

    monkeypatch.setattr(
        "stacksats.execution_adapters.importlib.util.spec_from_file_location",
        lambda *a, **k: type("Spec", (), {"loader": None})(),
    )
    fake = tmp_path / "fake.py"
    fake.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ImportError, match="Could not load adapter module spec"):
        load_execution_adapter(f"{fake}:Adapter")


def test_insert_all_data_missing_execute_values_raises_import_error() -> None:
    with pytest.raises(ImportError, match="psycopg2-binary"):
        insert_all_data(
            conn=None,
            df=pl.DataFrame(),
            execute_values=None,
            prepare_copy_dataframe_fn=lambda df: df,
            build_insert_rows_fn=lambda df: [],
        )


def test_execution_state_error_branches(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))

    with pytest.raises(RuntimeError, match="before claiming run"):
        store.mark_run_success_with_snapshot(
            strategy_id="s",
            strategy_version="1.0.0",
            run_date="2024-01-03",
            mode="paper",
            payload={},
            order_summary=None,
            force_flag=False,
            snapshot_date="2024-01-03",
            weights=pl.DataFrame({"date": [dt.datetime(2024, 1, 3)], "weight": [1.0]}),
        )
    with pytest.raises(RuntimeError, match="before claiming run"):
        store.mark_run_failure(
            strategy_id="s",
            strategy_version="1.0.0",
            run_date="2024-01-03",
            mode="paper",
            payload={},
            force_flag=False,
        )

    # expected_index empty path
    locked = store.load_locked_prefix(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        run_date="2024-01-03",
        window_start=dt.datetime(2024, 1, 3),
    )
    assert isinstance(locked, np.ndarray)
    assert locked.size == 0

    # incomplete snapshot length mismatch
    store.write_weight_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        snapshot_date="2024-01-03",
        weights=pl.DataFrame({"date": [dt.datetime(2024, 1, 2)], "weight": [0.5]}),
    )
    with pytest.raises(ValueError, match="incomplete"):
        store.load_locked_prefix(
            strategy_id="s",
            strategy_version="1.0.0",
            mode="paper",
            run_date="2024-01-04",
            window_start=dt.datetime(2024, 1, 2),
        )

    # date mismatch branch with same row count
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("DELETE FROM weight_snapshots")
        conn.execute(
            """
            INSERT INTO weight_snapshots(strategy_id, strategy_version, mode, snapshot_date, date, weight)
            VALUES ('s', '1.0.0', 'paper', '2024-01-03', '2024-01-02', 0.2),
                   ('s', '1.0.0', 'paper', '2024-01-03', '2024-01-02T12:00:00', 0.3)
            """
        )
        conn.commit()
    with pytest.raises(ValueError, match="do not match expected"):
        store.load_locked_prefix(
            strategy_id="s",
            strategy_version="1.0.0",
            mode="paper",
            run_date="2024-01-04",
            window_start=dt.datetime(2024, 1, 2),
        )


def test_runner_run_daily_uncovered_paths(tmp_path: Path, monkeypatch) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    btc = _btc_df()
    _allow_validation(runner, monkeypatch)

    with pytest.raises(ValueError, match="greater than 0"):
        runner.run_daily(
            strategy,
            RunDailyConfig(
                run_date="2024-12-31",
                total_window_budget_usd=0.0,
                state_db_path=str(tmp_path / "state1.sqlite3"),
            ),
            btc_df=btc,
        )

    with pytest.raises(ValueError, match="either 'paper' or 'live'"):
        runner.run_daily(
            strategy,
            RunDailyConfig(
                run_date="2024-12-31",
                total_window_budget_usd=1000.0,
                mode="invalid",  # type: ignore[arg-type]
                state_db_path=str(tmp_path / "state2.sqlite3"),
                output_dir=str(tmp_path),
            ),
            btc_df=btc,
        )

    bad_col = runner.run_daily(
        strategy,
        RunDailyConfig(
            run_date="2024-12-31",
            total_window_budget_usd=1000.0,
            btc_price_col="MISSING_COL",
            state_db_path=str(tmp_path / "state3.sqlite3"),
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
    )
    assert bad_col.status == "failed"
    assert "Missing required BTC price column" in bad_col.message

    # missing run-date weight allocation branch
    strategy_missing_weight = _UniformStrategy()
    strategy_missing_weight.compute_weights = lambda ctx: pl.DataFrame(
        {"date": [ctx.start_date], "weight": [1.0]}
    )
    missing_weight = runner.run_daily(
        strategy_missing_weight,
        RunDailyConfig(
            run_date="2024-12-31",
            total_window_budget_usd=1000.0,
            state_db_path=str(tmp_path / "state4.sqlite3"),
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
    )
    assert missing_weight.status == "failed"
    assert "missing run_date allocation" in missing_weight.message

    run_date_str = "2024-12-31"
    btc_zero = btc.with_columns(
        pl.when(pl.col("date").dt.strftime("%Y-%m-%d") == run_date_str)
        .then(0.0)
        .otherwise(pl.col("price_usd"))
        .alias("price_usd")
    )
    zero_price = runner.run_daily(
        _UniformStrategy(),
        RunDailyConfig(
            run_date="2024-12-31",
            total_window_budget_usd=1000.0,
            state_db_path=str(tmp_path / "state5.sqlite3"),
            output_dir=str(tmp_path),
        ),
        btc_df=btc_zero,
    )
    assert zero_price.status == "failed"
    assert "greater than 0" in zero_price.message

    adapter_file = tmp_path / "adapter.py"
    adapter_file.write_text(
        "\n".join(
            [
                "class LiveAdapter:",
                "    def submit_order(self, request, *, idempotency_key):",
                "        return {",
                "            'status': 'filled',",
                "            'external_order_id': 'x-' + idempotency_key,",
                "            'filled_notional_usd': request.notional_usd,",
                "            'filled_quantity_btc': request.quantity_btc,",
                "            'fill_price_usd': request.price_usd,",
                "            'metadata': {'source': 'dict'}",
                "        }",
            ]
        ),
        encoding="utf-8",
    )
    live_result = runner.run_daily(
        _UniformStrategy(),
        RunDailyConfig(
            run_date="2024-12-31",
            total_window_budget_usd=1000.0,
            mode="live",
            adapter_spec=f"{adapter_file}:LiveAdapter",
            state_db_path=str(tmp_path / "state6.sqlite3"),
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
    )
    assert live_result.status == "executed"
    assert live_result.order_receipt is not None
    assert live_result.order_receipt.external_order_id is not None

    bad_adapter_file = tmp_path / "bad_adapter.py"
    bad_adapter_file.write_text(
        "\n".join(
            [
                "class BadAdapter:",
                "    def submit_order(self, request, *, idempotency_key):",
                "        return 'bad'",
            ]
        ),
        encoding="utf-8",
    )
    bad_receipt = runner.run_daily(
        _UniformStrategy(),
        RunDailyConfig(
            run_date="2024-12-31",
            total_window_budget_usd=1000.0,
            mode="live",
            adapter_spec=f"{bad_adapter_file}:BadAdapter",
            state_db_path=str(tmp_path / "state7.sqlite3"),
            output_dir=str(tmp_path),
        ),
        btc_df=btc,
    )
    assert bad_receipt.status == "failed"
    assert "DailyOrderReceipt" in bad_receipt.message

    monkeypatch.setattr(
        "stacksats.execution_adapters.PaperExecutionAdapter.submit_order",
        lambda self, request, idempotency_key: (_ for _ in ()).throw(
            IdempotencyConflictError("conflict")
        ),
    )
    with pytest.raises(IdempotencyConflictError):
        runner.run_daily(
            _UniformStrategy(),
            RunDailyConfig(
                run_date="2024-12-31",
                total_window_budget_usd=1000.0,
                state_db_path=str(tmp_path / "state8.sqlite3"),
                output_dir=str(tmp_path),
            ),
            btc_df=btc,
        )


def test_compute_cycle_spd_source_mask_branches() -> None:
    dates = pl.datetime_range(
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 1) + dt.timedelta(days=369),
        interval="1d",
        eager=True,
    ).to_list()
    n = len(dates)
    prices = np.linspace(10000.0, 20000.0, n)
    # Set source_exists False for all rows after index 100 so end truncates
    # and no full 365-day windows fit in the truncated range
    source_exists = [True] * 101 + [False] * (n - 101)
    df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": prices,
            "price_usd_source_exists": source_exists,
            "mvrv": np.linspace(1.0, 2.0, n),
        }
    )
    feats = df.select(["date", "price_usd", "mvrv"])
    out = compute_cycle_spd(
        df,
        lambda window: pl.DataFrame({
            "date": window["date"],
            "weight": [1.0 / window.height] * window.height,
        }),
        features_df=feats,
        start_date="2020-01-01",
        end_date="2021-01-10",
    )
    assert out.is_empty()


def test_strategy_types_helpers_and_runner_validation_edge(monkeypatch) -> None:
    strategy = _UniformStrategy()
    assert RunDailyConfig().state_db_path.endswith("run_state.sqlite3")

    monkeypatch.setattr(
        "stacksats.runner.StrategyRunner.validate",
        lambda self, strategy, config, **kwargs: type("V", (), {"passed": True, "messages": [], "summary": lambda self: "ok"})(),
    )
    monkeypatch.setattr(
        "stacksats.runner.StrategyRunner.backtest",
        lambda self, strategy, config, **kwargs: type(
            "B",
            (),
            {
                "strategy_id": "s",
                "strategy_version": "1.0.0",
                "run_id": "r",
                "summary": lambda self: "ok",
            },
        )(),
    )
    run_result = strategy.run(
        validation_config=ValidationConfig(),
        save_backtest_artifacts=False,
    )
    assert run_result.output_dir is None

    monkeypatch.setattr(
        "stacksats.runner.StrategyRunner.run_daily",
        lambda self, strategy, config, **kwargs: "daily-ok",
    )
    assert strategy.run_daily() == "daily-ok"

    runner = StrategyRunner()
    assert runner._compute_win_rate(pl.DataFrame()) == 0.0
    nan_win_rate = runner._compute_win_rate(
        pl.DataFrame({"dynamic_percentile": [np.nan], "uniform_percentile": [np.nan]})
    )
    assert nan_win_rate == 0.0

    state = _ValidationState(messages=[])
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2),
        interval="1d", eager=True
    ).to_list()
    runner._run_locked_prefix_check(
        strategy=_UniformStrategy(),
        features_df=pl.DataFrame(
            {
                "date": dates,
                "price_usd": [100.0, 101.0],
                "mvrv": [1.0, 1.1],
            }
        ),
        start_ts=dt.datetime(2024, 1, 1),
        max_window_start=dt.datetime(2024, 1, 1),
        strict_mode=True,
        state=state,
    )


def test_runner_locked_prefix_check_empty_weights_early_return(monkeypatch) -> None:
    runner = StrategyRunner()
    state = _ValidationState(messages=[])
    monkeypatch.setattr(
        runner,
        "_compute_with_mutation_guard",
        lambda strategy, ctx, strict_mode: (pl.DataFrame(schema={"date": pl.Datetime, "weight": pl.Float64}), False),
    )
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2),
        interval="1d", eager=True
    ).to_list()
    runner._run_locked_prefix_check(
        strategy=_UniformStrategy(),
        features_df=pl.DataFrame(
            {
                "date": dates,
                "price_usd": [100.0, 101.0],
                "mvrv": [1.0, 1.1],
            }
        ),
        start_ts=dt.datetime(2024, 1, 1),
        max_window_start=dt.datetime(2024, 1, 1),
        strict_mode=True,
        state=state,
    )
    assert state.strict_checks_ok is True


def test_plot_and_export_importerror_fallbacks(monkeypatch) -> None:
    import stacksats.export_weights as export_weights
    import stacksats.plot_mvrv as plot_mvrv
    import stacksats.plot_weights as plot_weights
    from stacksats.export_weights_db import sql_quote

    # plot_mvrv backend fallback
    monkeypatch.setattr(plot_mvrv.plt, "switch_backend", lambda *_: (_ for _ in ()).throw(RuntimeError("no backend")))
    plot_mvrv._init_plot_env()
    monkeypatch.setattr(plot_weights.plt, "switch_backend", lambda *_: (_ for _ in ()).throw(RuntimeError("no backend")))
    plot_weights._init_plot_env()

    # dotenv import fallback in plot_weights/export_weights
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    plot_weights._load_dotenv_if_available()
    export_weights._load_dotenv_if_available()

    # export_weights module-level psycopg2.connect fallback branch
    original_psycopg2 = sys.modules.get("psycopg2")
    stub = ModuleType("psycopg2")
    monkeypatch.setitem(sys.modules, "psycopg2", stub)
    reloaded = importlib.reload(export_weights)
    assert hasattr(reloaded.psycopg2, "connect")
    with pytest.raises(ImportError, match="psycopg2-binary"):
        reloaded.psycopg2.connect()
    if original_psycopg2 is not None:
        monkeypatch.setitem(sys.modules, "psycopg2", original_psycopg2)
    reloaded = importlib.reload(reloaded)
    assert reloaded is not None

    assert sql_quote(None) == "NULL"
    assert sql_quote(np.int64(7)) == "7"
