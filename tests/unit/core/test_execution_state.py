from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from stacksats.api import ExecutionReceiptEvent
import stacksats.execution_state as execution_state_module
from stacksats.execution_state import (
    IdempotencyConflictError,
    ReceiptConflictError,
    SQLiteExecutionStateStore,
    ValidationReceipt,
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


def test_multi_handle_completed_run_is_noop_for_same_fingerprint(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    first = SQLiteExecutionStateStore(str(db_path))
    second = SQLiteExecutionStateStore(str(db_path))

    claim = first.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )
    assert claim.status == "claimed"
    first.mark_run_success(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "executed", "weight_today": 0.1},
        order_summary={"status": "filled"},
        force_flag=False,
    )

    noop_claim = second.claim_run(
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
    assert noop_claim.existing_run.payload["weight_today"] == 0.1


def test_multi_handle_completed_run_conflict_uses_latest_persisted_state(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    first = SQLiteExecutionStateStore(str(db_path))
    second = SQLiteExecutionStateStore(str(db_path))

    first.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )
    second.mark_run_success(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "executed"},
        order_summary={"status": "filled"},
        force_flag=False,
    )

    with pytest.raises(IdempotencyConflictError):
        first.claim_run(
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


def test_multi_handle_force_overwrite_is_visible_to_other_store(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    first = SQLiteExecutionStateStore(str(db_path))
    second = SQLiteExecutionStateStore(str(db_path))

    first.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        run_key="run-1",
        fingerprint="fingerprint-a",
        force=False,
    )
    first.mark_run_success(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "executed", "marker": "before"},
        order_summary={"status": "filled"},
        force_flag=False,
    )

    overwrite = second.claim_run(
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

    second.mark_run_success(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        payload={"status": "executed", "marker": "after"},
        order_summary={"status": "filled"},
        force_flag=True,
    )
    stored = first.get_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
    )
    assert stored is not None
    assert stored.run_key == "run-2"
    assert stored.fingerprint == "fingerprint-b"
    assert stored.force_flag is True
    assert stored.payload["marker"] == "after"


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


def test_validation_receipts_persist_multiple_receipts_for_same_run(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))

    first = store.create_validation_receipt(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        fingerprint="fp-a",
        data_hash="dh-a",
        provider_hash="ph-a",
        feature_snapshot_hash="fh-a",
        config_hash="cfg-a",
        passed=False,
        diagnostics={"reason": "bad-a"},
    )
    second = store.create_validation_receipt(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        fingerprint="fp-b",
        data_hash="dh-b",
        provider_hash="ph-b",
        feature_snapshot_hash="fh-b",
        config_hash="cfg-b",
        passed=True,
        diagnostics={"reason": "bad-b"},
    )

    fetched_first = store.get_validation_receipt(first.receipt_id)
    fetched_second = store.get_validation_receipt(second.receipt_id)
    assert fetched_first == ValidationReceipt(
        receipt_id=first.receipt_id,
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        fingerprint="fp-a",
        data_hash="dh-a",
        provider_hash="ph-a",
        feature_snapshot_hash="fh-a",
        config_hash="cfg-a",
        passed=False,
        diagnostics={"reason": "bad-a"},
    )
    assert fetched_second == ValidationReceipt(
        receipt_id=second.receipt_id,
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        fingerprint="fp-b",
        data_hash="dh-b",
        provider_hash="ph-b",
        feature_snapshot_hash="fh-b",
        config_hash="cfg-b",
        passed=True,
        diagnostics={"reason": "bad-b"},
    )


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


def test_mark_run_success_requires_prior_claim(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))

    with pytest.raises(RuntimeError, match="claiming run"):
        store.mark_run_success(
            strategy_id="s",
            strategy_version="1.0.0",
            run_date="2024-01-03",
            mode="paper",
            payload={"status": "executed"},
            order_summary=None,
            force_flag=False,
        )


def test_mark_run_failure_requires_prior_claim(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))

    with pytest.raises(RuntimeError, match="claiming run"):
        store.mark_run_failure(
            strategy_id="s",
            strategy_version="1.0.0",
            run_date="2024-01-03",
            mode="paper",
            payload={"status": "failed"},
            force_flag=False,
        )


def test_mark_run_success_with_snapshot_replaces_existing_snapshot_rows(tmp_path: Path) -> None:
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
    first_weights = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "weight": [0.1, 0.2],
        }
    )
    store.mark_run_success_with_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-03",
        mode="paper",
        payload={"status": "executed", "round": "first"},
        order_summary=None,
        force_flag=False,
        snapshot_date="2024-01-02",
        weights=first_weights,
    )

    store.claim_run(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-03",
        mode="paper",
        run_key="run-2",
        fingerprint="fp-2",
        force=True,
    )
    replacement_weights = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1)],
            "weight": [0.9],
        }
    )
    store.mark_run_success_with_snapshot(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-03",
        mode="paper",
        payload={"status": "executed", "round": "second"},
        order_summary=None,
        force_flag=True,
        snapshot_date="2024-01-02",
        weights=replacement_weights,
    )

    with sqlite3.connect(str(tmp_path / "state.sqlite3")) as conn:
        rows = conn.execute(
            """
            SELECT date, weight
            FROM weight_snapshots
            WHERE strategy_id = ? AND strategy_version = ? AND mode = ? AND snapshot_date = ?
            ORDER BY date ASC
            """,
            ("s", "1.0.0", "paper", "2024-01-02"),
        ).fetchall()

    assert rows == [("2024-01-01", 0.9)]


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


def _claim_decision(store: SQLiteExecutionStateStore) -> None:
    store.claim_run(
        strategy_id="decision-strategy",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="decision",
        run_key="decision-1",
        fingerprint="decision-fp",
        force=False,
    )
    store.mark_run_success(
        strategy_id="decision-strategy",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="decision",
        payload={
            "status": "decided",
            "decision_key": "decision-1",
            "strategy_id": "decision-strategy",
            "strategy_version": "1.0.0",
            "run_date": "2025-01-01",
            "recommended_notional_usd": 10.0,
            "recommended_quantity_btc": 0.0001,
        },
        order_summary=None,
        force_flag=False,
    )


def test_get_run_by_run_key_returns_decision_record(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    _claim_decision(store)

    stored = store.get_run_by_run_key(run_key="decision-1", mode="decision")

    assert stored is not None
    assert stored.run_key == "decision-1"
    assert stored.payload["status"] == "decided"


def test_ingest_execution_receipt_accumulates_and_matches(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    _claim_decision(store)

    submitted, submitted_idempotent = store.ingest_execution_receipt(
        decision_key="decision-1",
        event=ExecutionReceiptEvent(
            decision_key="decision-1",
            event_id="evt-1",
            event_type="submitted",
            event_time="2025-01-01T10:00:00Z",
            external_order_id="ord-1",
        ),
    )
    partial, partial_idempotent = store.ingest_execution_receipt(
        decision_key="decision-1",
        event=ExecutionReceiptEvent(
            decision_key="decision-1",
            event_id="evt-2",
            event_type="partially_filled",
            event_time="2025-01-01T10:01:00Z",
            external_order_id="ord-1",
            filled_notional_usd=4.0,
            filled_quantity_btc=0.00004,
            fill_price_usd=100000.0,
        ),
    )
    matched, matched_idempotent = store.ingest_execution_receipt(
        decision_key="decision-1",
        event=ExecutionReceiptEvent(
            decision_key="decision-1",
            event_id="evt-3",
            event_type="filled",
            event_time="2025-01-01T10:02:00Z",
            external_order_id="ord-1",
            filled_notional_usd=6.0,
            filled_quantity_btc=0.00006,
            fill_price_usd=100000.0,
        ),
    )

    assert submitted_idempotent is False
    assert partial_idempotent is False
    assert matched_idempotent is False
    assert submitted.execution_status == "submitted"
    assert submitted.reconciliation_status == "pending"
    assert partial.filled_notional_usd == pytest.approx(4.0)
    assert partial.receipt_count == 2
    assert matched.execution_status == "filled"
    assert matched.reconciliation_status == "matched"
    assert matched.filled_notional_usd == pytest.approx(10.0)
    assert matched.filled_quantity_btc == pytest.approx(0.0001)
    assert matched.average_fill_price_usd == pytest.approx(100000.0)


def test_ingest_execution_receipt_duplicate_same_body_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    _claim_decision(store)
    event = ExecutionReceiptEvent(
        decision_key="decision-1",
        event_id="evt-1",
        event_type="submitted",
        event_time="2025-01-01T10:00:00Z",
        external_order_id="ord-1",
    )

    first, first_idempotent = store.ingest_execution_receipt(
        decision_key="decision-1",
        event=event,
    )
    second, second_idempotent = store.ingest_execution_receipt(
        decision_key="decision-1",
        event=event,
    )

    assert first_idempotent is False
    assert second_idempotent is True
    assert second.receipt_count == 1
    assert second.execution_status == first.execution_status


def test_ingest_execution_receipt_duplicate_conflict_raises(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    _claim_decision(store)
    store.ingest_execution_receipt(
        decision_key="decision-1",
        event=ExecutionReceiptEvent(
            decision_key="decision-1",
            event_id="evt-1",
            event_type="submitted",
            event_time="2025-01-01T10:00:00Z",
        ),
    )

    with pytest.raises(ReceiptConflictError):
        store.ingest_execution_receipt(
            decision_key="decision-1",
            event=ExecutionReceiptEvent(
                decision_key="decision-1",
                event_id="evt-1",
                event_type="failed",
                event_time="2025-01-01T10:05:00Z",
                message="broker rejected order",
            ),
        )


def test_execution_status_and_history_retrieval_cover_terminal_states(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    _claim_decision(store)
    store.ingest_execution_receipt(
        decision_key="decision-1",
        event=ExecutionReceiptEvent(
            decision_key="decision-1",
            event_id="evt-1",
            event_type="partially_filled",
            event_time="2025-01-01T10:01:00Z",
            filled_notional_usd=12.0,
            filled_quantity_btc=0.00012,
        ),
    )
    status_result = store.get_execution_status(decision_key="decision-1")
    history_result = store.get_execution_receipts(decision_key="decision-1")

    assert status_result is not None
    assert status_result.reconciliation_status == "overfilled"
    assert history_result is not None
    assert len(history_result.receipts) == 1
    assert history_result.receipts[0].event_type == "partially_filled"


def test_execution_state_returns_none_for_missing_lookup_paths(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))

    assert store.get_run_by_run_key(run_key="missing") is None
    assert store.get_execution_status(decision_key="missing") is None
    assert store.get_execution_receipts(decision_key="missing") is None


def test_ingest_execution_receipt_validates_decision_key_and_status(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(str(tmp_path / "state.sqlite3"))
    _claim_decision(store)

    with pytest.raises(ValueError, match="must match"):
        store.ingest_execution_receipt(
            decision_key="decision-1",
            event=ExecutionReceiptEvent(
                decision_key="other-decision",
                event_id="evt-1",
                event_type="submitted",
                event_time="2025-01-01T10:00:00Z",
            ),
        )

    store.claim_run(
        strategy_id="failed-strategy",
        strategy_version="1.0.0",
        run_date="2025-01-02",
        mode="decision",
        run_key="failed-decision",
        fingerprint="fp-failed",
        force=False,
    )
    store.mark_run_failure(
        strategy_id="failed-strategy",
        strategy_version="1.0.0",
        run_date="2025-01-02",
        mode="decision",
        payload={"status": "failed", "decision_key": "failed-decision", "message": "bad"},
        force_flag=False,
    )

    with pytest.raises(ValueError, match="decided daily decisions"):
        store.ingest_execution_receipt(
            decision_key="failed-decision",
            event=ExecutionReceiptEvent(
                decision_key="failed-decision",
                event_id="evt-2",
                event_type="submitted",
                event_time="2025-01-02T10:00:00Z",
            ),
        )


def test_execution_state_internal_datetime_and_status_helpers() -> None:
    naive_dt = dt.datetime(2025, 1, 1, 10, 0, 0)
    aware_dt = dt.datetime(2025, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)

    assert execution_state_module._parse_datetime_like(naive_dt).tzinfo is not None
    assert execution_state_module._parse_datetime_like("2025-01-01T10:00:00Z").tzinfo is not None
    assert execution_state_module._normalize_event_time(aware_dt).endswith("Z")

    failed_decision = execution_state_module.StoredRun(
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="decision",
        run_key="decision-1",
        fingerprint="fp",
        status="failed",
        payload={"status": "failed", "message": "decision failed"},
        order_summary=None,
        force_flag=False,
    )
    failed_status = execution_state_module._default_execution_status(
        decision=failed_decision,
        decision_status="failed",
        recommended_notional=1.0,
        recommended_quantity=0.1,
    )

    assert failed_status.execution_status == "failed"
    assert failed_status.reconciliation_status == "failed"
    assert execution_state_module._classify_execution_reconciliation_status(
        decision_status="failed",
        execution_status="filled",
        recommended_notional_usd=1.0,
        recommended_quantity_btc=0.1,
        filled_notional_usd=1.0,
        filled_quantity_btc=0.1,
    ) == "failed"
    assert execution_state_module._classify_execution_reconciliation_status(
        decision_status="decided",
        execution_status="canceled",
        recommended_notional_usd=10.0,
        recommended_quantity_btc=0.1,
        filled_notional_usd=5.0,
        filled_quantity_btc=0.05,
    ) == "underfilled"
    assert execution_state_module._classify_execution_reconciliation_status(
        decision_status="decided",
        execution_status="rejected",
        recommended_notional_usd=10.0,
        recommended_quantity_btc=0.1,
        filled_notional_usd=0.0,
        filled_quantity_btc=0.0,
    ) == "failed"
    assert execution_state_module._classify_execution_reconciliation_status(
        decision_status="decided",
        execution_status="mystery",
        recommended_notional_usd=10.0,
        recommended_quantity_btc=0.1,
        filled_notional_usd=0.0,
        filled_quantity_btc=0.0,
    ) == "pending"
