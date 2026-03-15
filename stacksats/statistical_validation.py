"""Statistical robustness helpers for strict strategy validation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float
    upper: float
    samples: np.ndarray


def build_purged_walk_forward_folds(
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    *,
    n_folds: int,
    min_train_days: int,
    test_days: int,
    embargo_days: int,
) -> list[tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]]:
    """Build purged walk-forward folds with a fixed embargo."""
    all_days = pl.datetime_range(start_ts, end_ts, interval="1d", eager=True).to_list()
    if len(all_days) < (min_train_days + test_days + embargo_days):
        return []

    folds: list[tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]] = []
    train_start = all_days[0]
    cursor = min_train_days - 1
    while len(folds) < n_folds:
        train_end = train_start + dt.timedelta(days=cursor)
        test_start = train_end + dt.timedelta(days=embargo_days + 1)
        test_end = test_start + dt.timedelta(days=test_days - 1)
        if test_end > end_ts:
            break
        folds.append((train_start, train_end, test_start, test_end))
        cursor += test_days
    return folds


def anchored_window_excess(
    spd_table: pl.DataFrame,
    *,
    step: int,
) -> np.ndarray:
    """Return non-overlapping excess values sampled at a fixed step."""
    if spd_table.is_empty() or "excess_percentile" not in spd_table.columns:
        return np.array([], dtype=float)
    values = spd_table["excess_percentile"].cast(pl.Float64, strict=False).to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=float)
    return values[:: max(int(step), 1)]


def block_bootstrap_confidence_interval(
    values: np.ndarray | pl.Series,
    *,
    block_size: int,
    trials: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Estimate a bootstrap CI for the sample mean using contiguous blocks."""
    arr = _finite_array(values)
    if arr.size == 0:
        return BootstrapInterval(0.0, 0.0, np.array([], dtype=float))
    if arr.size == 1:
        return BootstrapInterval(float(arr[0]), float(arr[0]), arr.copy())

    block = max(1, min(int(block_size), arr.size))
    rng = np.random.default_rng(seed)
    samples = np.empty(int(trials), dtype=float)
    for idx in range(int(trials)):
        sampled = _sample_blocks(arr, block, rng)
        samples[idx] = float(np.mean(sampled))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1.0 - alpha)),
        samples=samples,
    )


def paired_block_permutation_pvalue(
    lhs: np.ndarray | pl.Series,
    rhs: np.ndarray | pl.Series,
    *,
    block_size: int,
    trials: int,
    seed: int,
) -> float:
    """Estimate a one-sided p-value by swapping paired series blockwise."""
    left = _finite_array(lhs)
    right = _finite_array(rhs)
    n = min(left.size, right.size)
    if n == 0:
        return 1.0
    left = left[:n]
    right = right[:n]
    observed = float(np.mean(left - right))
    block = max(1, min(int(block_size), n))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(trials)):
        perm_left = left.copy()
        perm_right = right.copy()
        for start in range(0, n, block):
            if bool(rng.integers(0, 2)):
                stop = min(start + block, n)
                temp = perm_left[start:stop].copy()
                perm_left[start:stop] = perm_right[start:stop]
                perm_right[start:stop] = temp
        trial_mean = float(np.mean(perm_left - perm_right))
        exceed += int(trial_mean >= observed - 1e-12)
    return float((exceed + 1) / (int(trials) + 1))


def population_stability_index(
    baseline: np.ndarray | pl.Series,
    candidate: np.ndarray | pl.Series,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI using equal-frequency bins from the baseline sample."""
    base = _finite_array(baseline)
    cand = _finite_array(candidate)
    if base.size == 0 or cand.size == 0:
        return 0.0
    if np.allclose(base, base[0]) and np.allclose(cand, cand[0]):
        return 0.0
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.quantile(base, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0
    base_hist, _ = np.histogram(base, bins=edges)
    cand_hist, _ = np.histogram(cand, bins=edges)
    base_pct = np.clip(base_hist / max(base_hist.sum(), 1), 1e-6, 1.0)
    cand_pct = np.clip(cand_hist / max(cand_hist.sum(), 1), 1e-6, 1.0)
    return float(np.sum((cand_pct - base_pct) * np.log(cand_pct / base_pct)))


def ks_statistic(
    baseline: np.ndarray | pl.Series,
    candidate: np.ndarray | pl.Series,
) -> float:
    """Compute the two-sample KS statistic."""
    base = _finite_array(baseline)
    cand = _finite_array(candidate)
    if base.size == 0 or cand.size == 0:
        return 0.0
    return float(stats.ks_2samp(base, cand).statistic)


def whites_reality_check(
    candidate_excess: dict[str, np.ndarray | pl.Series],
    *,
    block_size: int,
    trials: int,
    seed: int,
) -> float:
    """Estimate White's Reality Check p-value for the best candidate mean excess."""
    cleaned = {
        name: _finite_array(values)
        for name, values in candidate_excess.items()
        if _finite_array(values).size > 0
    }
    if not cleaned:
        return 1.0
    min_length = min(values.size for values in cleaned.values())
    aligned = {name: values[:min_length] for name, values in cleaned.items()}
    observed = max(float(np.mean(values)) for values in aligned.values())
    centered = {name: values - float(np.mean(values)) for name, values in aligned.items()}
    block = max(1, min(int(block_size), min_length))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(trials)):
        maxima = []
        for values in centered.values():
            sampled = _sample_blocks(values, block, rng)
            maxima.append(float(np.mean(sampled)))
        exceed += int(max(maxima) >= observed - 1e-12)
    return float((exceed + 1) / (int(trials) + 1))


def _sample_blocks(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= block_size:
        return values.copy()
    pieces = []
    while sum(piece.size for piece in pieces) < values.size:
        start = int(rng.integers(0, values.size - block_size + 1))
        pieces.append(values[start : start + block_size])
    sampled = np.concatenate(pieces)
    return sampled[: values.size]


def _finite_array(values: np.ndarray | pl.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]
