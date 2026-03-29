"""SQLite-backed state store for idempotent daily strategy execution."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .api import ExecutionReceiptEvent, ExecutionReceiptHistoryResult, ExecutionStatusResult


class IdempotencyConflictError(ValueError):
    """Raised when an existing run conflicts with new run inputs."""


class ReceiptConflictError(ValueError):
    """Raised when a duplicate execution receipt conflicts with stored data."""


@dataclass(frozen=True, slots=True)
class StoredRun:
    strategy_id: str
    strategy_version: str
    run_date: str
    mode: str
    run_key: str
    fingerprint: str
    status: str
    payload: dict
    order_summary: dict | None
    force_flag: bool


@dataclass(frozen=True, slots=True)
class RunClaim:
    status: str
    existing_run: StoredRun | None = None
    forced_overwrite: bool = False


@dataclass(frozen=True, slots=True)
class ValidationReceipt:
    receipt_id: int
    strategy_id: str
    strategy_version: str
    run_date: str
    fingerprint: str
    data_hash: str
    provider_hash: str
    feature_snapshot_hash: str
    config_hash: str
    passed: bool
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class StoredReceiptEvent:
    decision_key: str
    strategy_id: str
    strategy_version: str
    run_date: str
    event_id: str
    event_type: str
    event_time: str
    broker_name: str | None
    broker_account_ref: str | None
    external_order_id: str | None
    filled_notional_usd: float | None
    filled_quantity_btc: float | None
    fill_price_usd: float | None
    message: str | None
    metadata: dict[str, object]
    payload: dict[str, object]


class SQLiteExecutionStateStore:
    """Durable idempotency and snapshot state for daily strategy runs."""

    def __init__(self, db_path: str):
        self._in_memory = db_path == ":memory:"
        if self._in_memory:
            self.db_path = Path(":memory:")
            self._memory_connection = sqlite3.connect(":memory:", timeout=30.0)
            self._memory_connection.row_factory = sqlite3.Row
        else:
            path = Path(db_path).expanduser()
            self.db_path = path if path.is_absolute() else Path.cwd() / path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_connection = None
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_runs (
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    run_key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    order_json TEXT,
                    validation_receipt_id INTEGER,
                    data_hash TEXT NOT NULL DEFAULT '',
                    feature_snapshot_hash TEXT NOT NULL DEFAULT '',
                    force_flag INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (strategy_id, strategy_version, run_date, mode)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weight_snapshots (
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    date TEXT NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (strategy_id, strategy_version, mode, snapshot_date, date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_receipts (
                    receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    provider_hash TEXT NOT NULL,
                    feature_snapshot_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_receipt_events (
                    event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_key TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    broker_name TEXT,
                    broker_account_ref TEXT,
                    external_order_id TEXT,
                    filled_notional_usd REAL,
                    filled_quantity_btc REAL,
                    fill_price_usd REAL,
                    message TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE (decision_key, event_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_status (
                    decision_key TEXT PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    decision_status TEXT NOT NULL,
                    execution_status TEXT NOT NULL,
                    reconciliation_status TEXT NOT NULL,
                    recommended_notional_usd REAL,
                    recommended_quantity_btc REAL,
                    filled_notional_usd REAL NOT NULL DEFAULT 0.0,
                    filled_quantity_btc REAL NOT NULL DEFAULT 0.0,
                    average_fill_price_usd REAL,
                    receipt_count INTEGER NOT NULL DEFAULT 0,
                    latest_event_type TEXT,
                    latest_event_time TEXT,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            self._ensure_column(
                conn,
                table="daily_runs",
                column="validation_receipt_id",
                ddl="INTEGER",
            )
            self._ensure_column(
                conn,
                table="daily_runs",
                column="data_hash",
                ddl="TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                table="daily_runs",
                column="feature_snapshot_hash",
                ddl="TEXT NOT NULL DEFAULT ''",
            )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, *, table: str, column: str, ddl: str) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def claim_run(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        mode: str,
        run_key: str,
        fingerprint: str,
        force: bool,
    ) -> RunClaim:
        normalized_run_date = _normalize_date_str(run_date)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM daily_runs
                WHERE strategy_id = ? AND strategy_version = ? AND run_date = ? AND mode = ?
                """,
                (strategy_id, strategy_version, normalized_run_date, mode),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO daily_runs (
                        strategy_id,
                        strategy_version,
                        run_date,
                        mode,
                        run_key,
                        fingerprint,
                        status,
                        payload_json,
                        order_json,
                        validation_receipt_id,
                        data_hash,
                        feature_snapshot_hash,
                        force_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, 'running', '{}', NULL, NULL, '', '', ?)
                    """,
                    (
                        strategy_id,
                        strategy_version,
                        normalized_run_date,
                        mode,
                        run_key,
                        fingerprint,
                        int(bool(force)),
                    ),
                )
                return RunClaim(status="claimed")

            existing = _row_to_stored_run(row)
            if existing.status == "executed" and not force:
                if existing.fingerprint == fingerprint:
                    return RunClaim(status="noop", existing_run=existing)
                raise IdempotencyConflictError(
                    "Existing completed daily run has different inputs. "
                    "Use --force to rerun with new parameters."
                )

            conn.execute(
                """
                UPDATE daily_runs
                SET run_key = ?,
                    fingerprint = ?,
                    status = 'running',
                    payload_json = '{}',
                    order_json = NULL,
                    validation_receipt_id = NULL,
                    data_hash = '',
                    feature_snapshot_hash = '',
                    force_flag = ?,
                    updated_at = datetime('now')
                WHERE strategy_id = ? AND strategy_version = ? AND run_date = ? AND mode = ?
                """,
                (
                    run_key,
                    fingerprint,
                    int(bool(force)),
                    strategy_id,
                    strategy_version,
                    normalized_run_date,
                    mode,
                ),
            )
            return RunClaim(status="claimed", forced_overwrite=bool(force))

    def mark_run_success(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        mode: str,
        payload: dict,
        order_summary: dict | None,
        force_flag: bool,
        validation_receipt_id: int | None = None,
        data_hash: str = "",
        feature_snapshot_hash: str = "",
    ) -> None:
        self._update_run(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_date=run_date,
            mode=mode,
            status="executed",
            payload=payload,
            order_summary=order_summary,
            force_flag=force_flag,
            validation_receipt_id=validation_receipt_id,
            data_hash=data_hash,
            feature_snapshot_hash=feature_snapshot_hash,
        )

    def mark_run_success_with_snapshot(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        mode: str,
        payload: dict,
        order_summary: dict | None,
        force_flag: bool,
        snapshot_date: str,
        weights: pl.DataFrame,
        validation_receipt_id: int | None = None,
        data_hash: str = "",
        feature_snapshot_hash: str = "",
    ) -> None:
        normalized_run_date = _normalize_date_str(run_date)
        normalized_snapshot_date = _normalize_date_str(snapshot_date)
        ordered = weights.sort("date")
        date_strs = (
            ordered["date"].cast(pl.Datetime, strict=False).dt.strftime("%Y-%m-%d").to_list()
        )
        rows = list(
            zip(
                [strategy_id] * ordered.height,
                [strategy_version] * ordered.height,
                [mode] * ordered.height,
                [normalized_snapshot_date] * ordered.height,
                date_strs,
                ordered["weight"].cast(pl.Float64, strict=False).to_list(),
                strict=True,
            )
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE daily_runs
                SET status = 'executed',
                    payload_json = ?,
                    order_json = ?,
                    validation_receipt_id = ?,
                    data_hash = ?,
                    feature_snapshot_hash = ?,
                    force_flag = ?,
                    updated_at = datetime('now')
                WHERE strategy_id = ? AND strategy_version = ? AND run_date = ? AND mode = ?
                """,
                (
                    json.dumps(payload, sort_keys=True, default=str),
                    (
                        json.dumps(order_summary, sort_keys=True, default=str)
                        if order_summary is not None
                        else None
                    ),
                    validation_receipt_id,
                    data_hash,
                    feature_snapshot_hash,
                    int(bool(force_flag)),
                    strategy_id,
                    strategy_version,
                    normalized_run_date,
                    mode,
                ),
            )
            if cursor.rowcount == 0:
                raise RuntimeError("Cannot update run state before claiming run.")
            conn.execute(
                """
                DELETE FROM weight_snapshots
                WHERE strategy_id = ? AND strategy_version = ? AND mode = ? AND snapshot_date = ?
                """,
                (strategy_id, strategy_version, mode, normalized_snapshot_date),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO weight_snapshots (
                        strategy_id,
                        strategy_version,
                        mode,
                        snapshot_date,
                        date,
                        weight
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def mark_run_failure(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        mode: str,
        payload: dict,
        force_flag: bool,
        validation_receipt_id: int | None = None,
        data_hash: str = "",
        feature_snapshot_hash: str = "",
    ) -> None:
        self._update_run(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_date=run_date,
            mode=mode,
            status="failed",
            payload=payload,
            order_summary=None,
            force_flag=force_flag,
            validation_receipt_id=validation_receipt_id,
            data_hash=data_hash,
            feature_snapshot_hash=feature_snapshot_hash,
        )

    def _update_run(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        mode: str,
        status: str,
        payload: dict,
        order_summary: dict | None,
        force_flag: bool,
        validation_receipt_id: int | None,
        data_hash: str,
        feature_snapshot_hash: str,
    ) -> None:
        normalized_run_date = _normalize_date_str(run_date)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE daily_runs
                SET status = ?,
                    payload_json = ?,
                    order_json = ?,
                    validation_receipt_id = ?,
                    data_hash = ?,
                    feature_snapshot_hash = ?,
                    force_flag = ?,
                    updated_at = datetime('now')
                WHERE strategy_id = ? AND strategy_version = ? AND run_date = ? AND mode = ?
                """,
                (
                    status,
                    json.dumps(payload, sort_keys=True, default=str),
                    (
                        json.dumps(order_summary, sort_keys=True, default=str)
                        if order_summary is not None
                        else None
                    ),
                    validation_receipt_id,
                    data_hash,
                    feature_snapshot_hash,
                    int(bool(force_flag)),
                    strategy_id,
                    strategy_version,
                    normalized_run_date,
                    mode,
                ),
            )
            if cursor.rowcount == 0:
                raise RuntimeError("Cannot update run state before claiming run.")

    def write_weight_snapshot(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        mode: str,
        snapshot_date: str,
        weights: pl.DataFrame,
    ) -> None:
        normalized_snapshot_date = _normalize_date_str(snapshot_date)
        ordered = weights.sort("date")
        date_strs = (
            ordered["date"].cast(pl.Datetime, strict=False).dt.strftime("%Y-%m-%d").to_list()
        )
        rows = list(
            zip(
                [strategy_id] * ordered.height,
                [strategy_version] * ordered.height,
                [mode] * ordered.height,
                [normalized_snapshot_date] * ordered.height,
                date_strs,
                ordered["weight"].cast(pl.Float64, strict=False).to_list(),
                strict=True,
            )
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                DELETE FROM weight_snapshots
                WHERE strategy_id = ? AND strategy_version = ? AND mode = ? AND snapshot_date = ?
                """,
                (strategy_id, strategy_version, mode, normalized_snapshot_date),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO weight_snapshots (
                        strategy_id,
                        strategy_version,
                        mode,
                        snapshot_date,
                        date,
                        weight
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def create_validation_receipt(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        fingerprint: str,
        data_hash: str,
        provider_hash: str,
        feature_snapshot_hash: str,
        config_hash: str,
        passed: bool,
        diagnostics: dict,
    ) -> ValidationReceipt:
        normalized_run_date = _normalize_date_str(run_date)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO validation_receipts (
                    strategy_id,
                    strategy_version,
                    run_date,
                    fingerprint,
                    data_hash,
                    provider_hash,
                    feature_snapshot_hash,
                    config_hash,
                    passed,
                    diagnostics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    strategy_version,
                    normalized_run_date,
                    fingerprint,
                    data_hash,
                    provider_hash,
                    feature_snapshot_hash,
                    config_hash,
                    int(bool(passed)),
                    json.dumps(diagnostics, sort_keys=True, default=str),
                ),
            )
            receipt_id = int(cursor.lastrowid)
        return ValidationReceipt(
            receipt_id=receipt_id,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_date=normalized_run_date,
            fingerprint=fingerprint,
            data_hash=data_hash,
            provider_hash=provider_hash,
            feature_snapshot_hash=feature_snapshot_hash,
            config_hash=config_hash,
            passed=bool(passed),
            diagnostics=diagnostics,
        )

    def get_validation_receipt(self, receipt_id: int) -> ValidationReceipt | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM validation_receipts
                WHERE receipt_id = ?
                """,
                (int(receipt_id),),
            ).fetchone()
        if row is None:
            return None
        return ValidationReceipt(
            receipt_id=int(row["receipt_id"]),
            strategy_id=row["strategy_id"],
            strategy_version=row["strategy_version"],
            run_date=row["run_date"],
            fingerprint=row["fingerprint"],
            data_hash=row["data_hash"],
            provider_hash=row["provider_hash"],
            feature_snapshot_hash=row["feature_snapshot_hash"],
            config_hash=row["config_hash"],
            passed=bool(row["passed"]),
            diagnostics=json.loads(row["diagnostics_json"] or "{}"),
        )

    def get_run(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        run_date: str,
        mode: str,
    ) -> StoredRun | None:
        normalized_run_date = _normalize_date_str(run_date)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM daily_runs
                WHERE strategy_id = ? AND strategy_version = ? AND run_date = ? AND mode = ?
                """,
                (strategy_id, strategy_version, normalized_run_date, mode),
            ).fetchone()
        if row is None:
            return None
        return _row_to_stored_run(row)

    def get_run_by_run_key(
        self,
        *,
        run_key: str,
        mode: str | None = None,
    ) -> StoredRun | None:
        with self._connect() as conn:
            if mode is None:
                row = conn.execute(
                    """
                    SELECT *
                    FROM daily_runs
                    WHERE run_key = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (run_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT *
                    FROM daily_runs
                    WHERE run_key = ? AND mode = ?
                    LIMIT 1
                    """,
                    (run_key, mode),
                ).fetchone()
        if row is None:
            return None
        return _row_to_stored_run(row)

    def ingest_execution_receipt(
        self,
        *,
        decision_key: str,
        event: ExecutionReceiptEvent,
    ) -> tuple[ExecutionStatusResult, bool]:
        normalized_event = _normalize_execution_receipt_event(event)
        if normalized_event.decision_key != decision_key:
            raise ValueError("Receipt event decision_key must match the request decision_key.")
        payload_json = json.dumps(normalized_event.to_json(), sort_keys=True, default=str)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            decision = self._get_decision_run(conn, decision_key)
            if decision is None:
                raise ValueError(
                    "No stored daily decision exists for the requested decision_key."
                )
            if str(decision.payload.get("status", "")) != "decided":
                raise ValueError(
                    "Execution receipts can only be ingested for decided daily decisions."
                )

            existing = conn.execute(
                """
                SELECT payload_json
                FROM execution_receipt_events
                WHERE decision_key = ? AND event_id = ?
                """,
                (decision_key, normalized_event.event_id),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload_json:
                    raise ReceiptConflictError(
                        "Existing execution receipt has different inputs for the same event_id."
                    )
                return self._load_execution_status_for_decision(conn, decision), True

            conn.execute(
                """
                INSERT INTO execution_receipt_events (
                    decision_key,
                    strategy_id,
                    strategy_version,
                    run_date,
                    event_id,
                    event_type,
                    event_time,
                    broker_name,
                    broker_account_ref,
                    external_order_id,
                    filled_notional_usd,
                    filled_quantity_btc,
                    fill_price_usd,
                    message,
                    metadata_json,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_key,
                    decision.strategy_id,
                    decision.strategy_version,
                    decision.run_date,
                    normalized_event.event_id,
                    normalized_event.event_type,
                    normalized_event.event_time,
                    normalized_event.broker_name,
                    normalized_event.broker_account_ref,
                    normalized_event.external_order_id,
                    normalized_event.filled_notional_usd,
                    normalized_event.filled_quantity_btc,
                    normalized_event.fill_price_usd,
                    normalized_event.message,
                    json.dumps(normalized_event.metadata, sort_keys=True, default=str),
                    payload_json,
                ),
            )
            status = _build_execution_status(
                decision=decision,
                receipts=self._load_receipt_events(conn, decision_key),
            )
            self._write_execution_status(conn, decision=decision, status=status)
            return status, False

    def get_execution_status(self, *, decision_key: str) -> ExecutionStatusResult | None:
        with self._connect() as conn:
            decision = self._get_decision_run(conn, decision_key)
            if decision is None:
                return None
            return self._load_execution_status_for_decision(conn, decision)

    def get_execution_receipts(
        self,
        *,
        decision_key: str,
    ) -> ExecutionReceiptHistoryResult | None:
        with self._connect() as conn:
            decision = self._get_decision_run(conn, decision_key)
            if decision is None:
                return None
            receipts = [
                _stored_receipt_event_to_public(event)
                for event in self._load_receipt_events(conn, decision_key)
            ]
        return ExecutionReceiptHistoryResult(
            decision_key=decision_key,
            receipts=receipts,
        )

    def _get_decision_run(
        self,
        conn: sqlite3.Connection,
        decision_key: str,
    ) -> StoredRun | None:
        row = conn.execute(
            """
            SELECT *
            FROM daily_runs
            WHERE run_key = ? AND mode = 'decision'
            LIMIT 1
            """,
            (decision_key,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_stored_run(row)

    def _load_receipt_events(
        self,
        conn: sqlite3.Connection,
        decision_key: str,
    ) -> list[StoredReceiptEvent]:
        rows = conn.execute(
            """
            SELECT *
            FROM execution_receipt_events
            WHERE decision_key = ?
            ORDER BY event_sequence ASC
            """,
            (decision_key,),
        ).fetchall()
        return [_row_to_stored_receipt_event(row) for row in rows]

    def _load_execution_status_for_decision(
        self,
        conn: sqlite3.Connection,
        decision: StoredRun,
    ) -> ExecutionStatusResult:
        row = conn.execute(
            """
            SELECT payload_json
            FROM execution_status
            WHERE decision_key = ?
            """,
            (decision.run_key,),
        ).fetchone()
        if row is not None:
            return _payload_to_execution_status(json.loads(row["payload_json"]))
        return _build_execution_status(
            decision=decision,
            receipts=self._load_receipt_events(conn, decision.run_key),
        )

    def _write_execution_status(
        self,
        conn: sqlite3.Connection,
        *,
        decision: StoredRun,
        status: ExecutionStatusResult,
    ) -> None:
        payload = status.to_json()
        conn.execute(
            """
            INSERT INTO execution_status (
                decision_key,
                strategy_id,
                strategy_version,
                run_date,
                decision_status,
                execution_status,
                reconciliation_status,
                recommended_notional_usd,
                recommended_quantity_btc,
                filled_notional_usd,
                filled_quantity_btc,
                average_fill_price_usd,
                receipt_count,
                latest_event_type,
                latest_event_time,
                message,
                payload_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(decision_key) DO UPDATE SET
                decision_status = excluded.decision_status,
                execution_status = excluded.execution_status,
                reconciliation_status = excluded.reconciliation_status,
                recommended_notional_usd = excluded.recommended_notional_usd,
                recommended_quantity_btc = excluded.recommended_quantity_btc,
                filled_notional_usd = excluded.filled_notional_usd,
                filled_quantity_btc = excluded.filled_quantity_btc,
                average_fill_price_usd = excluded.average_fill_price_usd,
                receipt_count = excluded.receipt_count,
                latest_event_type = excluded.latest_event_type,
                latest_event_time = excluded.latest_event_time,
                message = excluded.message,
                payload_json = excluded.payload_json,
                updated_at = datetime('now')
            """,
            (
                decision.run_key,
                decision.strategy_id,
                decision.strategy_version,
                decision.run_date,
                status.decision_status,
                status.execution_status,
                status.reconciliation_status,
                status.recommended_notional_usd,
                status.recommended_quantity_btc,
                status.filled_notional_usd,
                status.filled_quantity_btc,
                status.average_fill_price_usd,
                status.receipt_count,
                status.latest_event_type,
                status.latest_event_time,
                status.message,
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )

    def load_locked_prefix(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        mode: str,
        run_date: str,
        window_start: dt.datetime,
    ) -> np.ndarray | None:
        run_ts = _parse_date_like(run_date)
        previous_snapshot_date = (run_ts - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        expected_dates = pl.datetime_range(
            window_start, run_ts - dt.timedelta(days=1), interval="1d", eager=True
        ).to_list()
        if len(expected_dates) == 0:
            return np.array([], dtype=float)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, weight
                FROM weight_snapshots
                WHERE strategy_id = ?
                  AND strategy_version = ?
                  AND mode = ?
                  AND snapshot_date = ?
                  AND date >= ?
                  AND date <= ?
                ORDER BY date ASC
                """,
                (
                    strategy_id,
                    strategy_version,
                    mode,
                    previous_snapshot_date,
                    min(d.strftime("%Y-%m-%d") for d in expected_dates),
                    max(d.strftime("%Y-%m-%d") for d in expected_dates),
                ),
            ).fetchall()

        if not rows:
            return None
        if len(rows) != len(expected_dates):
            raise ValueError(
                "Existing weight snapshot is incomplete for locked-prefix reuse."
            )

        observed_dates = [_parse_date_like(row["date"]) for row in rows]
        expected_set = {d.strftime("%Y-%m-%d") for d in expected_dates}
        observed_set = {d.strftime("%Y-%m-%d") for d in observed_dates}
        if expected_set != observed_set:
            raise ValueError(
                "Existing weight snapshot dates do not match expected locked-prefix range."
            )
        return np.asarray([float(row["weight"]) for row in rows], dtype=float)


USD_RECONCILIATION_TOLERANCE = 0.01
BTC_RECONCILIATION_TOLERANCE = 1e-10


def _normalize_execution_receipt_event(event: ExecutionReceiptEvent) -> ExecutionReceiptEvent:
    return ExecutionReceiptEvent(
        decision_key=event.decision_key,
        event_id=event.event_id,
        event_type=event.event_type,
        event_time=_normalize_event_time(event.event_time),
        broker_name=event.broker_name,
        broker_account_ref=event.broker_account_ref,
        external_order_id=event.external_order_id,
        filled_notional_usd=_optional_float(event.filled_notional_usd),
        filled_quantity_btc=_optional_float(event.filled_quantity_btc),
        fill_price_usd=_optional_float(event.fill_price_usd),
        message=event.message,
        metadata=dict(event.metadata),
    )


def _normalize_event_time(value: str | dt.datetime) -> str:
    parsed = _parse_datetime_like(value)
    return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime_like(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = str(value).strip()
        normalized = text.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _row_to_stored_receipt_event(row: sqlite3.Row) -> StoredReceiptEvent:
    return StoredReceiptEvent(
        decision_key=row["decision_key"],
        strategy_id=row["strategy_id"],
        strategy_version=row["strategy_version"],
        run_date=row["run_date"],
        event_id=row["event_id"],
        event_type=row["event_type"],
        event_time=row["event_time"],
        broker_name=row["broker_name"],
        broker_account_ref=row["broker_account_ref"],
        external_order_id=row["external_order_id"],
        filled_notional_usd=_optional_float(row["filled_notional_usd"]),
        filled_quantity_btc=_optional_float(row["filled_quantity_btc"]),
        fill_price_usd=_optional_float(row["fill_price_usd"]),
        message=row["message"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        payload=json.loads(row["payload_json"] or "{}"),
    )


def _stored_receipt_event_to_public(event: StoredReceiptEvent) -> ExecutionReceiptEvent:
    return ExecutionReceiptEvent(
        decision_key=event.decision_key,
        event_id=event.event_id,
        event_type=event.event_type,
        event_time=event.event_time,
        broker_name=event.broker_name,
        broker_account_ref=event.broker_account_ref,
        external_order_id=event.external_order_id,
        filled_notional_usd=event.filled_notional_usd,
        filled_quantity_btc=event.filled_quantity_btc,
        fill_price_usd=event.fill_price_usd,
        message=event.message,
        metadata=event.metadata,
    )


def _payload_to_execution_status(payload: dict[str, Any]) -> ExecutionStatusResult:
    return ExecutionStatusResult(
        decision_key=str(payload["decision_key"]),
        strategy_id=str(payload["strategy_id"]),
        run_date=str(payload["run_date"]),
        decision_status=str(payload["decision_status"]),
        execution_status=str(payload["execution_status"]),
        reconciliation_status=str(payload["reconciliation_status"]),
        recommended_notional_usd=_optional_float(payload.get("recommended_notional_usd")),
        recommended_quantity_btc=_optional_float(payload.get("recommended_quantity_btc")),
        filled_notional_usd=float(payload["filled_notional_usd"]),
        filled_quantity_btc=float(payload["filled_quantity_btc"]),
        average_fill_price_usd=_optional_float(payload.get("average_fill_price_usd")),
        receipt_count=int(payload["receipt_count"]),
        latest_event_type=payload.get("latest_event_type"),
        latest_event_time=payload.get("latest_event_time"),
        message=str(payload["message"]),
    )


def _build_execution_status(
    *,
    decision: StoredRun,
    receipts: list[StoredReceiptEvent],
) -> ExecutionStatusResult:
    decision_status = str(decision.payload.get("status") or decision.status)
    recommended_notional = _optional_float(decision.payload.get("recommended_notional_usd"))
    recommended_quantity = _optional_float(decision.payload.get("recommended_quantity_btc"))
    if not receipts:
        return _default_execution_status(
            decision=decision,
            decision_status=decision_status,
            recommended_notional=recommended_notional,
            recommended_quantity=recommended_quantity,
        )

    latest = receipts[-1]
    filled_notional = float(sum(event.filled_notional_usd or 0.0 for event in receipts))
    filled_quantity = float(sum(event.filled_quantity_btc or 0.0 for event in receipts))
    average_fill_price = (
        filled_notional / filled_quantity
        if filled_quantity > BTC_RECONCILIATION_TOLERANCE
        else None
    )
    execution_status = latest.event_type
    reconciliation_status = _classify_execution_reconciliation_status(
        decision_status=decision_status,
        execution_status=execution_status,
        recommended_notional_usd=recommended_notional,
        recommended_quantity_btc=recommended_quantity,
        filled_notional_usd=filled_notional,
        filled_quantity_btc=filled_quantity,
    )
    message = latest.message or (
        f"Execution status is '{execution_status}' with reconciliation "
        f"'{reconciliation_status}'."
    )
    return ExecutionStatusResult(
        decision_key=decision.run_key,
        strategy_id=decision.strategy_id,
        run_date=decision.run_date,
        decision_status=decision_status,
        execution_status=execution_status,
        reconciliation_status=reconciliation_status,
        recommended_notional_usd=recommended_notional,
        recommended_quantity_btc=recommended_quantity,
        filled_notional_usd=filled_notional,
        filled_quantity_btc=filled_quantity,
        average_fill_price_usd=average_fill_price,
        receipt_count=len(receipts),
        latest_event_type=latest.event_type,
        latest_event_time=latest.event_time,
        message=message,
    )


def _default_execution_status(
    *,
    decision: StoredRun,
    decision_status: str,
    recommended_notional: float | None,
    recommended_quantity: float | None,
) -> ExecutionStatusResult:
    if decision_status == "decided":
        execution_status = "pending"
        reconciliation_status = "pending"
        message = "Decision created; waiting for external execution receipts."
    else:
        execution_status = "failed"
        reconciliation_status = "failed"
        message = str(
            decision.payload.get("error")
            or decision.payload.get("message")
            or "Decision failed before external execution."
        )
    return ExecutionStatusResult(
        decision_key=decision.run_key,
        strategy_id=decision.strategy_id,
        run_date=decision.run_date,
        decision_status=decision_status,
        execution_status=execution_status,
        reconciliation_status=reconciliation_status,
        recommended_notional_usd=recommended_notional,
        recommended_quantity_btc=recommended_quantity,
        filled_notional_usd=0.0,
        filled_quantity_btc=0.0,
        average_fill_price_usd=None,
        receipt_count=0,
        latest_event_type=None,
        latest_event_time=None,
        message=message,
    )


def _classify_execution_reconciliation_status(
    *,
    decision_status: str,
    execution_status: str,
    recommended_notional_usd: float | None,
    recommended_quantity_btc: float | None,
    filled_notional_usd: float,
    filled_quantity_btc: float,
) -> str:
    if decision_status != "decided":
        return "failed"
    notional_target = float(recommended_notional_usd or 0.0)
    quantity_target = float(recommended_quantity_btc or 0.0)
    overfilled = (
        filled_notional_usd > (notional_target + USD_RECONCILIATION_TOLERANCE)
        or filled_quantity_btc > (quantity_target + BTC_RECONCILIATION_TOLERANCE)
    )
    matched = (
        abs(filled_notional_usd - notional_target) <= USD_RECONCILIATION_TOLERANCE
        and abs(filled_quantity_btc - quantity_target) <= BTC_RECONCILIATION_TOLERANCE
    )
    if overfilled:
        return "overfilled"
    if execution_status in {"submitted", "partially_filled", "pending"}:
        return "pending"
    if execution_status == "filled":
        return "matched" if matched else "underfilled"
    if execution_status == "canceled":
        return "underfilled"
    if execution_status in {"rejected", "failed"}:
        return "failed"
    return "pending"


def _parse_date_like(date_like: str | dt.datetime) -> dt.datetime:
    if isinstance(date_like, dt.datetime):
        out = date_like.replace(hour=0, minute=0, second=0, microsecond=0)
        if date_like.tzinfo:
            out = date_like.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return out
    return dt.datetime.strptime(str(date_like)[:10], "%Y-%m-%d")


def _norm_dt_str(date_val) -> str:
    if isinstance(date_val, dt.datetime):
        return date_val.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    return str(date_val)[:10]


def _normalize_date_str(date_like: str) -> str:
    return _parse_date_like(date_like).strftime("%Y-%m-%d")


def _row_to_stored_run(row: sqlite3.Row) -> StoredRun:
    return StoredRun(
        strategy_id=row["strategy_id"],
        strategy_version=row["strategy_version"],
        run_date=row["run_date"],
        mode=row["mode"],
        run_key=row["run_key"],
        fingerprint=row["fingerprint"],
        status=row["status"],
        payload=json.loads(row["payload_json"] or "{}"),
        order_summary=(
            json.loads(row["order_json"]) if row["order_json"] is not None else None
        ),
        force_flag=bool(row["force_flag"]),
    )


__all__ = [
    "IdempotencyConflictError",
    "ReceiptConflictError",
    "RunClaim",
    "SQLiteExecutionStateStore",
    "StoredRun",
    "ValidationReceipt",
]
