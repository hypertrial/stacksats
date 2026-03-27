from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from stacksats.execution_state import (
    IdempotencyConflictError,
    SQLiteExecutionStateStore,
    _norm_dt_str,
    _parse_date_like,
)


def test_state_store_initializes_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    SQLiteExecutionStateStore(str(db_path))
    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "daily_runs" in tables
    assert "weight_snapshots" in tables


def test_state_store_memory_path_uses_true_sqlite_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    store = SQLiteExecutionStateStore(":memory:")
    claim = store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )

    assert claim.status == "claimed"
    assert store.db_path == Path(":memory:")
    assert not (tmp_path / ":memory:").exists()


def test_claim_or_noop_behavior_for_same_fingerprint(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    claim = store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )
    assert claim.status == "claimed"
    store.mark_run_success(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "executed"},
        order_summary={"status": "filled"},
        force_flag=False,
    )

    noop_claim = store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-2",
        fingerprint="fingerprint-a",
        force=False,
    )
    assert noop_claim.status == "noop"
    assert noop_claim.existing_run is not None
    assert noop_claim.existing_run.run_key == "run-1"


def test_claim_conflict_for_mismatched_fingerprint(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )
    store.mark_run_success(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "executed"},
        order_summary={"status": "filled"},
        force_flag=False,
    )

    with pytest.raises(IdempotencyConflictError):
        store.claim_run(
            strategy_id="s",
            strategy_version="1.0.0",
            run_date="2025-01-01",
            mode="paper",
            run_key="run-2",
            fingerprint="fingerprint-b",
            force=False,
        )


def test_snapshot_round_trip_locked_prefix(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1) + dt.timedelta(days=364),
        interval="1d", eager=True
    ).to_list()
    weights = pl.DataFrame({
        "date": dates,
        "weight": [1.0 / 365.0] * 365,
    })
    store.write_weight_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        snapshot_date="2024-12-30",
        weights=weights,
    )
    locked_prefix = store.load_locked_prefix(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        run_date="2024-12-31",
        window_start=dt.datetime(2024, 1, 2),
    )
    assert locked_prefix is not None
    assert len(locked_prefix) == 364


def test_claim_force_overwrites_running_run_and_missing_getters_return_none(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )

    overwrite = store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-2",
        fingerprint="fingerprint-b",
        force=True,
    )

    assert overwrite.status == "claimed"
    assert overwrite.forced_overwrite is True
    assert store.get_run(
        strategy_id="missing",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
    ) is None
    assert store.get_validation_receipt(9999) is None


def test_validation_receipt_and_mark_run_failure_round_trip(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )
    receipt = store.create_validation_receipt(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        fingerprint="fp",
        data_hash="dh",
        provider_hash="ph",
        feature_snapshot_hash="fh",
        config_hash="cfg",
        passed=False,
        diagnostics={"reason": "bad"},
    )
    fetched = store.get_validation_receipt(receipt.receipt_id)
    assert fetched is not None
    assert fetched.diagnostics == {"reason": "bad"}

    store.mark_run_failure(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "failed"},
        force_flag=True,
        validation_receipt_id=receipt.receipt_id,
        data_hash="dh",
        feature_snapshot_hash="fh",
    )
    stored = store.get_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
    )
    assert stored is not None
    assert stored.status == "failed"
    assert stored.force_flag is True


def test_load_locked_prefix_empty_and_incomplete_snapshot_paths(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    weights = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 2)],
            "weight": [1.0],
        }
    )
    store.write_weight_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        snapshot_date="2024-01-02",
        weights=weights,
    )

    empty_locked = store.load_locked_prefix(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        run_date="2024-01-01",
        window_start=dt.datetime(2024, 1, 1),
    )
    assert empty_locked is not None
    assert empty_locked.size == 0

    with pytest.raises(ValueError, match="incomplete"):
        store.load_locked_prefix(
            strategy_id="s",
            strategy_version="1.0.0",
            mode="paper",
            run_date="2024-01-03",
            window_start=dt.datetime(2024, 1, 1),
        )


def test_execution_state_datetime_normalizers_handle_datetime_inputs() -> None:
    aware = dt.datetime(2024, 1, 3, 12, 30, tzinfo=dt.timezone.utc)
    parsed = _parse_date_like(aware)
    assert parsed == dt.datetime(2024, 1, 3, 12, 30)
    assert _parse_date_like(dt.datetime(2024, 1, 3, 12, 30)) == dt.datetime(2024, 1, 3)
    assert _norm_dt_str(aware) == "2024-01-03"
    assert _norm_dt_str("2024-01-04T15:30:00") == "2024-01-04"


def test_write_weight_snapshot_allows_empty_rows(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))

    store.write_weight_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        snapshot_date="2024-01-02",
        weights=pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64}),
    )

    loaded = store.load_locked_prefix(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        run_date="2024-01-03",
        window_start=dt.datetime(2024, 1, 1),
    )
    assert loaded is None


def test_mark_run_success_with_snapshot_allows_empty_rows_after_claim(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-03",
        mode="paper",
        run_key="run-1",
        fingerprint="fp-1",
        force=False,
    )

    store.mark_run_success_with_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-03",
        mode="paper",
        payload={"status": "executed"},
        order_summary=None,
        force_flag=False,
        snapshot_date="2024-01-03",
        weights=pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64}),
    )

    loaded = store.load_locked_prefix(
        strategy_id="s",
        strategy_version="1.0.0",
        mode="paper",
        run_date="2024-01-04",
        window_start=dt.datetime(2024, 1, 1),
    )
    assert loaded is None


def test_ensure_column_adds_missing_column(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY)")
        SQLiteExecutionStateStore._ensure_column(
            conn,
            table="demo",
            column="name",
            ddl="TEXT",
        )
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(demo)").fetchall()
        }
    assert "name" in cols
