#!/usr/bin/env python3
"""Train and freeze linear DuckDB factor coefficients for DuckDBAlphaStrategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats.data_btc import BTCDataProvider
from stacksats.feature_registry import DEFAULT_FEATURE_REGISTRY
from stacksats.strategy_types import BaseStrategy


@dataclass(frozen=True)
class _FoldResult:
    ic: float
    hit_rate: float
    n_obs: int


class _TrainingStrategy(BaseStrategy):
    strategy_id = "duckdb-alpha-training"

    def required_feature_sets(self) -> tuple[str, ...]:
        return (
            "core_model_features_v1",
            "brk_overlay_v1",
            "duckdb_analytics_factors_v1",
        )

    def propose_weight(self, state):
        return state.uniform_weight


def _future_mean_price_label(price: pd.Series, horizon_days: int) -> pd.Series:
    safe_price = pd.to_numeric(price, errors="coerce").replace(0.0, np.nan)
    future_mean = safe_price.iloc[::-1].rolling(
        horizon_days, min_periods=max(90, horizon_days // 4)
    ).mean().iloc[::-1]
    label = np.log(future_mean / safe_price)
    return label.replace([np.inf, -np.inf], np.nan).clip(-3.0, 3.0)


def _purged_folds(
    index: pd.DatetimeIndex,
    *,
    n_folds: int,
    min_train_days: int,
    min_test_days: int,
    embargo_days: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    dates = pd.DatetimeIndex(index).sort_values().unique()
    n = len(dates)
    if n <= (min_train_days + min_test_days):
        return []
    test_span = max(min_test_days, (n - min_train_days) // max(n_folds, 1))
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    cursor = min_train_days
    while cursor + min_test_days <= n and len(folds) < n_folds:
        train_end_idx = max(cursor - embargo_days, min_train_days)
        test_end_idx = min(cursor + test_span, n)
        train_dates = dates[:train_end_idx]
        test_dates = dates[cursor:test_end_idx]
        if len(train_dates) >= min_train_days and len(test_dates) >= min_test_days:
            folds.append((train_dates, test_dates))
        cursor = test_end_idx
    return folds


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_scaled = (x_train - mean) / std
    y_centered = y_train - float(np.mean(y_train))
    gram = x_scaled.T @ x_scaled
    ridge = gram + (float(alpha) * np.eye(gram.shape[0]))
    coef = np.linalg.solve(ridge, x_scaled.T @ y_centered)
    intercept = float(np.mean(y_train))
    return coef, intercept, mean, std


def _predict(
    x: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    x_scaled = (x - mean) / std
    return intercept + (x_scaled @ coef)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _select_features(
    frame: pd.DataFrame,
    label: pd.Series,
    *,
    min_coverage: float,
    max_features: int,
) -> list[str]:
    candidates = [column for column in frame.columns if column.startswith("ddb_")]
    scored: list[tuple[float, str]] = []
    for column in candidates:
        series = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(series.to_numpy(dtype=float))
        coverage = float(finite.mean())
        if coverage < min_coverage:
            continue
        filled = series.fillna(0.0).to_numpy(dtype=float)
        if float(np.std(filled)) < 1e-8:
            continue
        corr = abs(_corr(filled, label.to_numpy(dtype=float)))
        scored.append((corr, column))
    scored.sort(reverse=True)
    return [column for _, column in scored[:max_features]]


def _profile_features(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        profile[column] = {
            "coverage": float(finite.mean()),
            "std": float(np.nanstd(values.to_numpy(dtype=float))),
            "sparsity": float((values.fillna(0.0).abs() <= 1e-12).mean()),
        }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen DuckDB alpha coefficients.")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2025-05-31")
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=6.0)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=720)
    parser.add_argument("--min-test-days", type=int, default=180)
    parser.add_argument("--embargo-days", type=int, default=365)
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument(
        "--output",
        default="stacksats/strategies/duckdb_alpha_v1.json",
    )
    args = parser.parse_args()

    if args.duckdb_path:
        os.environ["STACKSATS_ANALYTICS_DUCKDB"] = str(Path(args.duckdb_path).expanduser())

    start_ts = pd.Timestamp(args.start_date).normalize()
    end_ts = pd.Timestamp(args.end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end-date must be on or after start-date.")

    btc_df = BTCDataProvider().load(backtest_start=args.start_date, end_date=args.end_date)
    strategy = _TrainingStrategy()
    features = DEFAULT_FEATURE_REGISTRY.materialize_for_strategy(
        strategy,
        btc_df,
        start_date=start_ts,
        end_date=end_ts,
        current_date=end_ts,
    )
    prices = pd.to_numeric(
        btc_df["price_usd"].reindex(features.index),
        errors="coerce",
    )
    label = _future_mean_price_label(prices, args.horizon_days)

    dataset = features.copy()
    dataset["label"] = label
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"])
    if dataset.empty:
        raise ValueError("No valid training rows after label construction.")

    selected = _select_features(
        dataset.drop(columns=["label"]),
        dataset["label"],
        min_coverage=float(args.min_coverage),
        max_features=int(args.max_features),
    )
    if not selected:
        raise ValueError("No DuckDB factors passed feature gates.")

    folds = _purged_folds(
        pd.DatetimeIndex(dataset.index),
        n_folds=int(args.folds),
        min_train_days=int(args.min_train_days),
        min_test_days=int(args.min_test_days),
        embargo_days=int(args.embargo_days),
    )
    if not folds:
        raise ValueError("Unable to generate purged walk-forward folds with current settings.")

    fold_results: list[_FoldResult] = []
    for train_dates, test_dates in folds:
        train = dataset.loc[train_dates]
        test = dataset.loc[test_dates]
        x_train = train[selected].fillna(0.0).to_numpy(dtype=float)
        y_train = train["label"].to_numpy(dtype=float)
        x_test = test[selected].fillna(0.0).to_numpy(dtype=float)
        y_test = test["label"].to_numpy(dtype=float)
        coef, intercept, mean, std = _fit_ridge(
            x_train,
            y_train,
            alpha=float(args.ridge_alpha),
        )
        pred = _predict(x_test, coef, intercept, mean, std)
        fold_results.append(
            _FoldResult(
                ic=_corr(pred, y_test),
                hit_rate=float((np.sign(pred) == np.sign(y_test)).mean()),
                n_obs=len(y_test),
            )
        )

    x_all = dataset[selected].fillna(0.0).to_numpy(dtype=float)
    y_all = dataset["label"].to_numpy(dtype=float)
    coef, intercept, mean, std = _fit_ridge(
        x_all,
        y_all,
        alpha=float(args.ridge_alpha),
    )

    artifact: dict[str, object] = {
        "artifact_version": 1,
        "artifact_id": "duckdb_alpha_v1",
        "description": "Frozen linear alpha overlay over MVRVPlus base preference.",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provider_id": "duckdb_analytics_factors_v1",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "label_horizon_days": int(args.horizon_days),
        "intercept": float(intercept),
        "feature_columns": selected,
        "coefficients": {
            column: float(value) for column, value in zip(selected, coef, strict=True)
        },
        "scaler": {
            "mean": {
                column: float(value) for column, value in zip(selected, mean, strict=True)
            },
            "std": {
                column: float(value) for column, value in zip(selected, std, strict=True)
            },
        },
        "training_metadata": {
            "selection_method": "ridge_walk_forward",
            "ridge_alpha": float(args.ridge_alpha),
            "fold_count": len(folds),
            "fold_ic_mean": float(np.mean([fold.ic for fold in fold_results])),
            "fold_hit_rate_mean": float(np.mean([fold.hit_rate for fold in fold_results])),
            "feature_profile": _profile_features(dataset, selected),
        },
    }
    artifact_hash = hashlib.sha256(
        json.dumps(artifact, sort_keys=True).encode("utf-8")
    ).hexdigest()
    artifact["training_metadata"]["artifact_hash"] = artifact_hash

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Selected features: {len(selected)}")
    print(f"Fold IC mean: {artifact['training_metadata']['fold_ic_mean']:.4f}")
    print(f"Fold hit-rate mean: {artifact['training_metadata']['fold_hit_rate_mean']:.4f}")
    print(f"Artifact hash: {artifact_hash}")
    print(f"Saved artifact: {output_path}")


if __name__ == "__main__":
    main()
