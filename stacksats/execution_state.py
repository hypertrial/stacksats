"""SQLite-backed state store for idempotent daily strategy execution."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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


class SQLiteExecutionStateStore:
    """Durable idempotency and snapshot state for daily strategy runs."""

    def __init__(self, db_path: str):
        path = Path(db_path).expanduser()
        self.db_path = path if path.is_absolute() else Path.cwd() / path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
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
                        force_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, 'running', '{}', NULL, ?)
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
        weights: pd.Series,
    ) -> None:
        normalized_run_date = _normalize_date_str(run_date)
        normalized_snapshot_date = _normalize_date_str(snapshot_date)
        ordered_weights = weights.sort_index()
        rows = [
            (
                strategy_id,
                strategy_version,
                mode,
                normalized_snapshot_date,
                pd.Timestamp(dt).normalize().strftime("%Y-%m-%d"),
                float(weight),
            )
            for dt, weight in ordered_weights.items()
        ]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE daily_runs
                SET status = 'executed',
                    payload_json = ?,
                    order_json = ?,
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
    ) -> None:
        normalized_run_date = _normalize_date_str(run_date)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE daily_runs
                SET status = ?,
                    payload_json = ?,
                    order_json = ?,
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
        weights: pd.Series,
    ) -> None:
        normalized_snapshot_date = _normalize_date_str(snapshot_date)
        ordered_weights = weights.sort_index()
        rows = [
            (
                strategy_id,
                strategy_version,
                mode,
                normalized_snapshot_date,
                pd.Timestamp(dt).normalize().strftime("%Y-%m-%d"),
                float(weight),
            )
            for dt, weight in ordered_weights.items()
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

    def load_locked_prefix(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        mode: str,
        run_date: str,
        window_start: pd.Timestamp,
    ) -> np.ndarray | None:
        run_ts = pd.Timestamp(run_date).normalize()
        previous_snapshot_date = (run_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        expected_index = pd.date_range(window_start, run_ts - pd.Timedelta(days=1), freq="D")
        if len(expected_index) == 0:
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
                    expected_index.min().strftime("%Y-%m-%d"),
                    expected_index.max().strftime("%Y-%m-%d"),
                ),
            ).fetchall()

        if not rows:
            return None
        if len(rows) != len(expected_index):
            raise ValueError(
                "Existing weight snapshot is incomplete for locked-prefix reuse."
            )

        observed_dates = [pd.Timestamp(row["date"]).normalize() for row in rows]
        if not pd.DatetimeIndex(observed_dates).equals(expected_index):
            raise ValueError(
                "Existing weight snapshot dates do not match expected locked-prefix range."
            )
        return np.asarray([float(row["weight"]) for row in rows], dtype=float)


def _normalize_date_str(date_like: str) -> str:
    return pd.Timestamp(date_like).normalize().strftime("%Y-%m-%d")


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
]
