"""Unit tests for edge cases in model_development.py.

Covers:
1. allocate_sequential_stable edge cases (n_past behavior)
2. Extreme input values for dynamic multiplier
3. Boundary checks for adaptive modifiers
"""

import numpy as np

from stacksats.model_development import (
    allocate_sequential_stable,
    compute_dynamic_multiplier,
    compute_adaptive_trend_modifier,
    compute_asymmetric_extreme_boost
)

# -----------------------------------------------------------------------------
# Allocation Edge Cases
# -----------------------------------------------------------------------------

def test_allocate_sequential_stable_n_past_gt_n():
    """Test that n_past > n is handled by locking all weights."""
    raw = np.array([1.0, 1.0, 1.0])
    n_past = 5 # Greater than len(raw)
    
    weights = allocate_sequential_stable(raw, n_past)
    
    assert len(weights) == 3
    # Should behave as if n_past = 3
    # Stable allocation logic applies to all
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), 1.0)


def test_allocate_sequential_stable_n_past_zero():
    """Test behavior when n_past is 0 (all future)."""
    raw = np.array([1.0, 1.0, 1.0])
    n_past = 0
    
    weights = allocate_sequential_stable(raw, n_past)
    
    # All uniform
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(weights, expected)


def test_allocate_sequential_stable_locked_weights_mismatch():
    """Partial locked prefixes keep shape under past-budget scaling."""
    raw = np.array([1.0, 1.0, 1.0])
    n_past = 2
    locked = np.array([0.5])

    weights = allocate_sequential_stable(raw, n_past, locked_weights=locked)

    # Kernel may scale the past segment, so absolute locked value can change.
    target_budget = n_past / len(raw)
    assert np.isclose(weights[:n_past].sum(), target_budget)
    assert weights[0] > 0.0
    assert np.isclose(weights.sum(), 1.0)
    assert len(weights) == 3


# -----------------------------------------------------------------------------
# Extreme Value Handling
# -----------------------------------------------------------------------------

def test_compute_dynamic_multiplier_extreme_values():
    """Test handling of extreme inputs (inf, nan handling wrapped in array)."""
    # Inputs
    price_vs_ma = np.array([0.0])
    mvrv_zscore = np.array([1000.0]) # Extreme value
    mvrv_gradient = np.array([0.0])
    
    # Should not crash and clip result
    multiplier = compute_dynamic_multiplier(
        price_vs_ma, mvrv_zscore, mvrv_gradient
    )
    
    assert np.all(np.isfinite(multiplier))
    assert multiplier[0] > 0 # Multipliers must be positive


def test_compute_asymmetric_extreme_boost_bounds():
    """Test that boost logic handles extremely deep negative values correctly."""
    z_scores = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    
    boost = compute_asymmetric_extreme_boost(z_scores)
    
    assert np.all(np.isfinite(boost))
    
    # -10 should have massive boost
    assert boost[0] > boost[1] 
    # +10 should have massive penalty (negative boost)
    assert boost[4] < boost[3]


def test_compute_adaptive_trend_modifier_bounds():
    """Test limits of adaptive trend modifier."""
    mvrv_gradient = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    mvrv_zscore = np.array([0.0] * 5) # Neutral z-score
    
    modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    
    # Should be clipped to [0.3, 1.5]
    assert np.all(modifier >= 0.3)
    assert np.all(modifier <= 1.5)
