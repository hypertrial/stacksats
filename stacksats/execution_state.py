"""SQLite-backed state store for idempotent daily strategy execution."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


class IdempotencyConflictError(ValueError):
    """Raised when an existing run conflicts with new run inputs."""


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
        rows = [
            (
                strategy_id,
                strategy_version,
                mode,
                normalized_snapshot_date,
                _norm_dt_str(row["date"]),
                float(row["weight"]),
            )
            for row in ordered.iter_rows(named=True)
        ]
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
        rows = [
            (
                strategy_id,
                strategy_version,
                mode,
                normalized_snapshot_date,
                _norm_dt_str(row["date"]),
                float(row["weight"]),
            )
            for row in ordered.iter_rows(named=True)
        ]
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
    "RunClaim",
    "SQLiteExecutionStateStore",
    "StoredRun",
    "ValidationReceipt",
]
