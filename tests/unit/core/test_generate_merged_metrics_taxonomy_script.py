from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl


def _write_synthetic_merged_metrics(path: Path) -> None:
    rows = [
        {"day_utc": "2024-01-01", "metric": "price_usd", "value": 42000.0},
        {"day_utc": "2024-01-01", "metric": "realized_cap", "value": 100.0},
        {"day_utc": "2024-01-01", "metric": "1m_dca_stack", "value": 1.0},
        {"day_utc": "2024-01-01", "metric": "year_2024_mvrv", "value": 1.2},
        {"day_utc": "2024-01-01", "metric": "epoch_4_mvrv", "value": 1.1},
        {"day_utc": "2024-01-01", "metric": "utxos_10y_to_12y_old_mvrv", "value": 0.9},
        {
            "day_utc": "2024-01-01",
            "metric": "addrs_above_100btc_under_1k_btc_mvrv",
            "value": 0.8,
        },
        {"day_utc": "2024-01-01", "metric": "sth_adjusted_sopr_7d_ema", "value": 1.0},
        {"day_utc": "2024-01-01", "metric": "p2wpkh_count_average", "value": 2.0},
        {"day_utc": "2024-01-01", "metric": "unknown_outputs_mvrv", "value": 0.7},
        {"day_utc": "2024-01-01", "metric": "address_activity_both_average", "value": 3.0},
        {"day_utc": "2024-01-01", "metric": "block_weight_sum", "value": 4.0},
        {"day_utc": "2024-01-01", "metric": "ckpool_blocks_mined", "value": 5.0},
        {"day_utc": "2024-01-01", "metric": "ckpool_dominance", "value": 6.0},
        {"day_utc": "2024-01-01", "metric": "ckpool_coinbase", "value": 7.0},
        {"day_utc": "2024-01-01", "metric": "ckpool_fee", "value": 8.0},
        {"day_utc": "2024-01-01", "metric": "ckpool_subsidy", "value": 9.0},
        {"day_utc": "2024-01-02", "metric": "price_usd", "value": 43000.0},
    ]
    pl.DataFrame(rows).with_columns(pl.col("day_utc").str.to_date()).write_parquet(path)


def test_generate_merged_metrics_taxonomy_script_generates_and_checks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "generate_merged_metrics_taxonomy.py"
    parquet_path = tmp_path / "merged_metrics_test.parquet"
    json_output = tmp_path / "taxonomy.json"
    doc_output = tmp_path / "taxonomy.md"
    _write_synthetic_merged_metrics(parquet_path)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    generate = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parquet-path",
            str(parquet_path),
            "--json-output",
            str(json_output),
            "--doc-output",
            str(doc_output),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generate.returncode == 0, generate.stderr or generate.stdout
    assert json_output.exists()
    assert doc_output.exists()

    taxonomy = json.loads(json_output.read_text(encoding="utf-8"))
    assert taxonomy["dataset_snapshot"]["distinct_metrics"] == 17
    assert taxonomy["dataset_snapshot"]["top_level_family_count"] >= 10
    registry = {item["family"]: item for item in taxonomy["namespace_registry"]}
    assert registry["ckpool"]["semantic_class"] == "mining_pool_metrics"
    assert registry["year"]["semantic_class"] == "vintage_year_cohorts"
    assert registry["epoch"]["semantic_class"] == "halving_epoch_cohorts"
    assert "Merged Metrics Taxonomy" in doc_output.read_text(encoding="utf-8")

    check = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parquet-path",
            str(parquet_path),
            "--json-output",
            str(json_output),
            "--doc-output",
            str(doc_output),
            "--check",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr or check.stdout
    assert "outputs are up to date" in check.stdout

    doc_output.write_text(doc_output.read_text(encoding="utf-8") + "\n<!-- stale -->\n", encoding="utf-8")
    stale = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parquet-path",
            str(parquet_path),
            "--json-output",
            str(json_output),
            "--doc-output",
            str(doc_output),
            "--check",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale.returncode == 1
    assert "out of date" in stale.stdout
