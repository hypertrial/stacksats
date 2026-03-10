#!/usr/bin/env python3
"""Render DuckDB schema markdown from a local analytics database."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUCKDB_PATH = ROOT / "bitcoin_analytics.duckdb"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "brk_data_manifest.json"
DEFAULT_OUTPUT_PATH = ROOT / "docs" / "reference" / "bitcoin-analytics-duckdb-schema.md"


@dataclass(frozen=True)
class TableColumn:
    name: str
    data_type: str
    nullable: bool
    ordinal_position: int


@dataclass(frozen=True)
class TableStats:
    table_name: str
    rows: int
    min_date: str
    max_date: str
    unique_metrics: int
    null_value_rows: int


def _load_drive_url(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return "n/a"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "n/a"
    if not isinstance(payload, dict):
        return "n/a"
    raw = payload.get("gdrive_folder_url")
    return str(raw).strip() if raw else "n/a"


def _table_names(con) -> list[str]:
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema='main'
        ORDER BY table_name
        """
    ).fetchall()
    return [row[0] for row in rows]


def _table_columns(con) -> dict[str, list[TableColumn]]:
    rows = con.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable, ordinal_position
        FROM information_schema.columns
        WHERE table_schema='main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    out: dict[str, list[TableColumn]] = defaultdict(list)
    for table_name, col_name, dtype, is_nullable, pos in rows:
        out[table_name].append(
            TableColumn(
                name=col_name,
                data_type=dtype,
                nullable=is_nullable == "YES",
                ordinal_position=pos,
            )
        )
    return out


def _primary_keys(con) -> dict[str, list[str]]:
    rows = con.execute(
        """
        SELECT k.table_name, k.column_name, k.ordinal_position
        FROM information_schema.table_constraints t
        JOIN information_schema.key_column_usage k
          ON t.constraint_name = k.constraint_name
         AND t.table_schema = k.table_schema
         AND t.table_name = k.table_name
        WHERE t.table_schema='main'
          AND t.constraint_type='PRIMARY KEY'
        ORDER BY k.table_name, k.ordinal_position
        """
    ).fetchall()
    out: dict[str, list[str]] = defaultdict(list)
    for table_name, col_name, _ in rows:
        out[table_name].append(col_name)
    return out


def _metric_table_stats(con, metric_tables: list[str]) -> list[TableStats]:
    stats: list[TableStats] = []
    for table_name in metric_tables:
        rows, min_date, max_date, uniq_metrics, null_rows = con.execute(
            f"""
            SELECT
              COUNT(*) AS rows,
              MIN(date_day) AS min_date,
              MAX(date_day) AS max_date,
              COUNT(DISTINCT metric) AS unique_metrics,
              SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_value_rows
            FROM "{table_name}"
            """
        ).fetchone()
        stats.append(
            TableStats(
                table_name=table_name,
                rows=int(rows or 0),
                min_date=str(min_date) if min_date is not None else "n/a",
                max_date=str(max_date) if max_date is not None else "n/a",
                unique_metrics=int(uniq_metrics or 0),
                null_value_rows=int(null_rows or 0),
            )
        )
    return stats


def _global_unique_metrics(con, metric_tables: list[str]) -> int:
    if not metric_tables:
        return 0
    union_sql = "\nUNION ALL\n".join(
        [f'SELECT metric FROM "{table_name}"' for table_name in metric_tables]
    )
    query = f"SELECT COUNT(*) FROM (SELECT DISTINCT metric FROM ({union_sql}) m)"
    return int(con.execute(query).fetchone()[0] or 0)


def _long_load_row_count(con, table_names: list[str], table_name: str) -> int:
    if table_name not in table_names:
        return 0
    return int(con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] or 0)


def _render_markdown(
    *,
    duckdb_path: Path,
    drive_url: str,
    table_names: list[str],
    columns: dict[str, list[TableColumn]],
    primary_keys: dict[str, list[str]],
    metric_stats: list[TableStats],
    global_unique_metrics: int,
    long_runs_rows: int,
    long_chunks_rows: int,
) -> str:
    metric_table_count = len(metric_stats)
    total_metric_rows = sum(item.rows for item in metric_stats)
    min_date = min((item.min_date for item in metric_stats), default="n/a")
    max_date = max((item.max_date for item in metric_stats), default="n/a")
    unique_metrics = global_unique_metrics
    null_rows = sum(item.null_value_rows for item in metric_stats)
    file_size = duckdb_path.stat().st_size
    gib = file_size / float(1024**3)

    lines: list[str] = []
    lines.append("# Bitcoin Analytics DuckDB Schema")
    lines.append("")
    lines.append(f"Source database: `{duckdb_path.name}`  ")
    lines.append("Schema scope: all user tables in `main`  ")
    lines.append("Snapshot source: generated from local DuckDB metadata")
    lines.append("")
    lines.append("All metrics in this database were derived via the BRK library from a local Bitcoin node.")
    lines.append("")
    lines.append("Canonical distribution references:")
    lines.append("")
    lines.append(f"- Drive folder: <{drive_url}>")
    lines.append("- Manifest: `data/brk_data_manifest.json`")
    lines.append("- Regenerate this page: `venv/bin/python scripts/render_duckdb_schema_doc.py`")
    lines.append("")
    lines.append("## Basic DB Stats")
    lines.append("")
    lines.append(f"Computed from `{duckdb_path.name}`.")
    lines.append("")
    lines.append("| Stat | Value |")
    lines.append("|---|---|")
    lines.append(f"| File size | `{file_size:,}` bytes (`~{gib:.2f} GiB`) |")
    lines.append(f"| Total user tables (`main`) | `{len(table_names)}` |")
    lines.append(f"| Metric tables (`metrics_*`) | `{metric_table_count}` |")
    lines.append(f"| Total metric rows | `{total_metric_rows:,}` |")
    lines.append(f"| Date coverage (metric tables) | `{min_date}` to `{max_date}` |")
    lines.append(f"| Unique metrics across all metric tables | `{unique_metrics:,}` |")
    lines.append(f"| Null `value` rows across all metric tables | `{null_rows:,}` |")
    lines.append(f"| `_long_load_runs` rows | `{long_runs_rows:,}` |")
    lines.append(f"| `_long_load_chunks` rows | `{long_chunks_rows:,}` |")
    lines.append("")
    lines.append("### Metric Table Summary")
    lines.append("")
    lines.append("| Table | Rows | Date range | Unique metrics | Null `value` rows |")
    lines.append("|---|---:|---|---:|---:|")
    for item in metric_stats:
        lines.append(
            f"| `{item.table_name}` | `{item.rows:,}` | `{item.min_date}` to `{item.max_date}` | "
            f"`{item.unique_metrics:,}` | `{item.null_value_rows:,}` |"
        )
    lines.append("")
    lines.append("## Tables")
    lines.append("")

    for table_name in table_names:
        lines.append(f"### `main.{table_name}`")
        lines.append("")
        lines.append("| Column | Type | Nullable |")
        lines.append("|---|---|---|")
        for col in columns.get(table_name, []):
            nullable = "yes" if col.nullable else "no"
            lines.append(f"| `{col.name}` | `{col.data_type}` | {nullable} |")
        if table_name in primary_keys:
            joined = ", ".join(primary_keys[table_name])
            lines.append("")
            lines.append("Primary key:  ")
            lines.append(f"`({joined})`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_schema_markdown(duckdb_path: Path, manifest_path: Path) -> str:
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        table_names = _table_names(con)
        columns = _table_columns(con)
        primary_keys = _primary_keys(con)
        metric_tables = [name for name in table_names if name.startswith("metrics_")]
        metric_stats = _metric_table_stats(con, metric_tables)
        global_unique_metrics = _global_unique_metrics(con, metric_tables)
        long_runs_rows = _long_load_row_count(con, table_names, "_long_load_runs")
        long_chunks_rows = _long_load_row_count(con, table_names, "_long_load_chunks")
    finally:
        con.close()

    drive_url = _load_drive_url(manifest_path)
    return _render_markdown(
        duckdb_path=duckdb_path,
        drive_url=drive_url,
        table_names=table_names,
        columns=columns,
        primary_keys=primary_keys,
        metric_stats=metric_stats,
        global_unique_metrics=global_unique_metrics,
        long_runs_rows=long_runs_rows,
        long_chunks_rows=long_chunks_rows,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render docs/reference/bitcoin-analytics-duckdb-schema.md from a DuckDB file."
    )
    parser.add_argument("--duckdb-path", default=str(DEFAULT_DUCKDB_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if output file does not match rendered markdown.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    duckdb_path = Path(args.duckdb_path).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rendered = render_schema_markdown(duckdb_path, manifest_path)
    if args.check:
        current = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if current != rendered:
            print(f"DuckDB schema markdown is out of date: {output_path}")
            return 1
        print("DuckDB schema markdown is up to date.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Rendered DuckDB schema markdown: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
