"""Tests for helper functions in model_development.py."""

import numpy as np
import pandas as pd
from stacksats.model_development import (
    classify_mvrv_zone,
    compute_signal_confidence,
    rolling_percentile,
    softmax,
    zscore,
)


class TestModelDevelopmentHelpers:
    """Tests for model development helper functions."""

    def test_softmax(self):
        """Test softmax function with basic inputs."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        
        assert len(result) == 3
        assert np.isclose(result.sum(), 1.0)
        assert result[2] > result[1] > result[0]
        
    def test_softmax_stability(self):
        """Test softmax numerical stability with large values."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0)
        assert not np.isnan(result).any()

    def test_zscore(self):
        """Test zscore function."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # window=3, min_periods=1
        result = zscore(s, window=3)
        
        # Result should be Series
        assert isinstance(result, pd.Series)
        # Check a middle value: [1, 2, 3] -> mean=2, std=1 -> (3-2)/1 = 1.0
        # Wait, pandas std is ddof=1 by default.
        # [1, 2, 3] -> mean=2, std=1.0. (3-2)/1.0 = 1.0.
        assert np.isclose(result.iloc[2], 1.0)

    def test_rolling_percentile(self):
        """Test rolling_percentile function."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        # window=5, min_periods = window // 4 = 1
        result = rolling_percentile(s, window=5)
        
        # iloc[0] should be 0.5 because len(x) < 2
        assert result.iloc[0] == 0.5
        # iloc[4] should be 1.0 because 50 is greater than [10, 20, 30, 40]
        assert result.iloc[4] == 1.0

    def test_classify_mvrv_zone(self):
        """Test classify_mvrv_zone function according to its implementation."""
        # Zones:
        # Z < -2.0 -> -2
        # -2.0 <= Z < -1.0 -> -1
        # -1.0 <= Z < 1.5 -> 0
        # 1.5 <= Z < 2.5 -> 1
        # Z >= 2.5 -> 2
        
        assert classify_mvrv_zone(np.array([-2.5]))[0] == -2
        assert classify_mvrv_zone(np.array([-1.5]))[0] == -1
        assert classify_mvrv_zone(np.array([1.0]))[0] == 0
        assert classify_mvrv_zone(np.array([2.0]))[0] == 1
        assert classify_mvrv_zone(np.array([3.0]))[0] == 2

    def test_compute_signal_confidence(self):
        """Test compute_signal_confidence logic."""
        z = np.array([-2.0, 0.0, 3.0])
        pct = np.array([0.1, 0.5, 0.9])
        grad = np.array([0.5, 0.0, -0.5])
        ma = np.array([-0.2, 0.0, 0.8])
        
        conf = compute_signal_confidence(z, pct, grad, ma)
        
        assert len(conf) == 3
        assert all(0.0 <= c <= 1.0 for c in conf)
        # First case: agreement on "buy" (Low Z, Low Pct, Pos Grad, Below MA) should have high conf
        # Last case: agreement on "sell" (High Z, High Pct, Neg Grad, Above MA) should also have high conf
        assert conf[0] > 0.5
        assert conf[2] > 0.5
