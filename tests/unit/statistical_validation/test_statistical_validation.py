from __future__ import annotations

import datetime as dt

import numpy as np

from stacksats.statistical_validation import (
    anchored_window_excess,
    block_bootstrap_confidence_interval,
    build_purged_walk_forward_folds,
    ks_statistic,
    paired_block_permutation_pvalue,
    population_stability_index,
    whites_reality_check,
)


def test_build_purged_walk_forward_folds_returns_empty_for_short_history() -> None:
    folds = build_purged_walk_forward_folds(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 6, 1),
        n_folds=4,
        min_train_days=365,
        test_days=90,
        embargo_days=30,
    )

    assert folds == []


def test_build_purged_walk_forward_folds_returns_ranges() -> None:
    folds = build_purged_walk_forward_folds(
        dt.datetime(2020, 1, 1),
        dt.datetime(2025, 12, 31),
        n_folds=2,
        min_train_days=365,
        test_days=180,
        embargo_days=30,
    )

    assert len(folds) >= 1
    train_start, train_end, test_start, test_end = folds[0]
    assert train_start < train_end < test_start < test_end


def test_anchored_window_excess_samples_non_overlapping_points() -> None:
    import polars as pl
    spd = pl.DataFrame({"excess_percentile": [1.0, 2.0, 3.0, 4.0]})
    sampled = anchored_window_excess(spd, step=2)
    assert np.array_equal(sampled, np.array([1.0, 3.0]))


def test_block_bootstrap_confidence_interval_handles_single_value() -> None:
    interval = block_bootstrap_confidence_interval(
        np.array([5.0]),
        block_size=5,
        trials=10,
        seed=1,
    )
    assert interval.lower == 5.0
    assert interval.upper == 5.0


def test_paired_block_permutation_pvalue_is_bounded() -> None:
    pvalue = paired_block_permutation_pvalue(
        np.array([2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        block_size=2,
        trials=50,
        seed=7,
    )
    assert 0.0 <= pvalue <= 1.0


def test_population_stability_index_and_ks_are_zero_for_identical_samples() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert population_stability_index(values, values) == 0.0
    assert ks_statistic(values, values) == 0.0


def test_whites_reality_check_handles_empty_candidates() -> None:
    assert whites_reality_check({}, block_size=2, trials=10, seed=5) == 1.0


def test_whites_reality_check_returns_probability() -> None:
    pvalue = whites_reality_check(
        {
            "a": np.array([0.1, 0.2, 0.0, 0.1]),
            "b": np.array([0.0, -0.1, 0.0, 0.1]),
        },
        block_size=2,
        trials=20,
        seed=3,
    )
    assert 0.0 <= pvalue <= 1.0
