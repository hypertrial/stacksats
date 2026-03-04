from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from stacksats.execution_state import (
    IdempotencyConflictError,
    SQLiteExecutionStateStore,
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
    index = pd.date_range("2024-01-01", periods=365, freq="D")
    weights = pd.Series(1.0 / 365.0, index=index, dtype=float)
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
        window_start=pd.Timestamp("2024-01-02"),
    )
    assert locked_prefix is not None
    assert len(locked_prefix) == 364
