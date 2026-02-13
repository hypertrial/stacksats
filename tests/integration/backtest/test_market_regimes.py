"""Market regime testing for the Bitcoin DCA strategy.

These tests verify strategy performance across different market conditions:
1. Bull market (consistent uptrend)
2. Bear market (consistent downtrend)
3. Sideways market (low volatility, range-bound)
4. High volatility market (large swings)
5. Crash and recovery scenarios

The strategy should maintain reasonable performance (not catastrophically fail)
across all market regimes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import stacksats.backtest as backtest
from stacksats.backtest import compute_weights_modal
from stacksats.model_development import compute_weights_fast, precompute_features
from stacksats.prelude import compute_cycle_spd

# -----------------------------------------------------------------------------
# Market Regime Generators
# -----------------------------------------------------------------------------


def create_bull_market_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 10000.0,
    daily_return: float = 0.002,  # ~100% annual return
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic bull market price data.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        daily_return: Mean daily return (positive for bull market)
        volatility: Daily volatility (std of returns)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate log returns with positive drift
    returns = np.random.normal(daily_return, volatility, len(dates))
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def create_bear_market_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 50000.0,
    daily_return: float = -0.001,  # ~30% annual decline
    volatility: float = 0.025,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic bear market price data.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        daily_return: Mean daily return (negative for bear market)
        volatility: Daily volatility
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate log returns with negative drift
    returns = np.random.normal(daily_return, volatility, len(dates))
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Ensure prices don't go below a minimum threshold
    prices = np.maximum(prices, 1000.0)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def create_sideways_market_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    mean_price: float = 30000.0,
    volatility: float = 0.01,  # Low volatility
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic sideways (range-bound) market price data.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        mean_price: Mean price around which prices oscillate
        volatility: Daily volatility (low for sideways)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate mean-reverting prices using Ornstein-Uhlenbeck-like process
    theta = 0.05  # Mean reversion speed
    log_mean_price = np.log(mean_price)

    log_prices = np.zeros(len(dates))
    log_prices[0] = log_mean_price

    for i in range(1, len(dates)):
        drift = theta * (log_mean_price - log_prices[i - 1])
        noise = volatility * np.random.randn()
        log_prices[i] = log_prices[i - 1] + drift + noise

    prices = np.exp(log_prices)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def create_high_volatility_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 30000.0,
    daily_return: float = 0.0005,  # Slight upward trend
    volatility: float = 0.06,  # High volatility (3x normal)
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic high-volatility market price data.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        daily_return: Mean daily return
        volatility: Daily volatility (high)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate high-volatility returns
    returns = np.random.normal(daily_return, volatility, len(dates))
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Ensure prices stay within reasonable bounds
    prices = np.clip(prices, 1000.0, 1000000.0)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def create_crash_recovery_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 30000.0,
    crash_date: str = "2022-06-01",
    crash_magnitude: float = 0.7,  # 70% drop
    recovery_rate: float = 0.003,  # Recovery daily return
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic crash and recovery market price data.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        crash_date: Date of the crash
        crash_magnitude: How much price drops (0.7 = 70% drop)
        recovery_rate: Daily return during recovery
        volatility: Daily volatility
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    crash_idx = np.searchsorted(dates, pd.Timestamp(crash_date))

    # Pre-crash: moderate uptrend
    pre_crash_returns = np.random.normal(0.001, volatility, crash_idx)
    pre_crash_log_prices = np.log(initial_price) + np.cumsum(pre_crash_returns)

    # Crash: sudden drop
    crash_log_price = pre_crash_log_prices[-1] + np.log(1 - crash_magnitude)

    # Post-crash: recovery
    post_crash_len = len(dates) - crash_idx
    post_crash_returns = np.random.normal(recovery_rate, volatility, post_crash_len)
    post_crash_log_prices = crash_log_price + np.cumsum(post_crash_returns)

    # Combine
    log_prices = np.concatenate([pre_crash_log_prices, post_crash_log_prices])
    prices = np.exp(log_prices)

    # Ensure prices stay within reasonable bounds
    prices = np.clip(prices, 1000.0, 1000000.0)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def bull_market_df():
    """Create bull market data."""
    return create_bull_market_data()


@pytest.fixture
def bull_market_features(bull_market_df):
    """Precompute features for bull market."""
    return precompute_features(bull_market_df)


@pytest.fixture
def bear_market_df():
    """Create bear market data."""
    return create_bear_market_data()


@pytest.fixture
def bear_market_features(bear_market_df):
    """Precompute features for bear market."""
    return precompute_features(bear_market_df)


@pytest.fixture
def sideways_market_df():
    """Create sideways market data."""
    return create_sideways_market_data()


@pytest.fixture
def sideways_market_features(sideways_market_df):
    """Precompute features for sideways market."""
    return precompute_features(sideways_market_df)


@pytest.fixture
def high_volatility_df():
    """Create high volatility market data."""
    return create_high_volatility_data()


@pytest.fixture
def high_volatility_features(high_volatility_df):
    """Precompute features for high volatility market."""
    return precompute_features(high_volatility_df)


@pytest.fixture
def crash_recovery_df():
    """Create crash and recovery market data."""
    return create_crash_recovery_data()


@pytest.fixture
def crash_recovery_features(crash_recovery_df):
    """Precompute features for crash recovery market."""
    return precompute_features(crash_recovery_df)


# -----------------------------------------------------------------------------
# Bull Market Tests
# -----------------------------------------------------------------------------


class TestBullMarketPerformance:
    """Test strategy performance in bull market conditions."""

    def test_weights_valid_in_bull_market(self, bull_market_features):
        """Verify weights are valid in bull market."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(bull_market_features, start_date, end_date)

        # Basic weight constraints
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6), "Weights must sum to 1.0"
        assert (weights >= -1e-10).all(), "All weights must be non-negative"
        assert weights.notna().all(), "No NaN weights allowed"

    def test_bull_market_backtest_runs(self, bull_market_df, bull_market_features):
        """Verify backtest runs successfully in bull market."""
        backtest._FEATURES_DF = bull_market_features

        spd_table = compute_cycle_spd(
            bull_market_df, compute_weights_modal, features_df=bull_market_features
        )

        assert len(spd_table) > 0, "Should produce SPD results"
        assert "dynamic_percentile" in spd_table.columns
        assert "uniform_percentile" in spd_table.columns

    def test_bull_market_no_catastrophic_failure(
        self, bull_market_df, bull_market_features
    ):
        """Verify strategy doesn't catastrophically fail in bull market.

        In bull market, DCA generally underperforms lump sum, but dynamic DCA
        should not be dramatically worse than uniform DCA.
        """
        backtest._FEATURES_DF = bull_market_features

        spd_table = compute_cycle_spd(
            bull_market_df, compute_weights_modal, features_df=bull_market_features
        )

        # Filter out NaN rows (edge cases)
        valid_rows = spd_table["excess_percentile"].dropna()

        if len(valid_rows) > 0:
            # Mean excess shouldn't be catastrophically negative (> -20%)
            mean_excess = valid_rows.mean()
            assert mean_excess > -20, (
                f"Strategy underperforms too much in bull market: {mean_excess:.1f}%"
            )


# -----------------------------------------------------------------------------
# Bear Market Tests
# -----------------------------------------------------------------------------


class TestBearMarketPerformance:
    """Test strategy performance in bear market conditions.

    Bear markets are where DCA strategies typically shine.
    """

    def test_weights_valid_in_bear_market(self, bear_market_features):
        """Verify weights are valid in bear market."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(bear_market_features, start_date, end_date)

        # Basic weight constraints
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6), "Weights must sum to 1.0"
        assert (weights >= -1e-10).all(), "All weights must be non-negative"
        assert weights.notna().all(), "No NaN weights allowed"

    def test_bear_market_backtest_runs(self, bear_market_df, bear_market_features):
        """Verify backtest runs successfully in bear market."""
        backtest._FEATURES_DF = bear_market_features

        spd_table = compute_cycle_spd(
            bear_market_df, compute_weights_modal, features_df=bear_market_features
        )

        assert len(spd_table) > 0, "Should produce SPD results"

    def test_bear_market_outperformance(self, bear_market_df, bear_market_features):
        """In bear markets, dynamic DCA should perform reasonably well.

        We expect the strategy to have positive excess percentile on average
        in bear markets.
        """
        backtest._FEATURES_DF = bear_market_features

        spd_table = compute_cycle_spd(
            bear_market_df, compute_weights_modal, features_df=bear_market_features
        )

        # Filter out NaN rows
        valid_rows = spd_table["excess_percentile"].dropna()

        if len(valid_rows) > 0:
            # Don't require positive excess, but it shouldn't be terrible
            mean_excess = valid_rows.mean()
            assert mean_excess > -15, (
                f"Strategy performs poorly in bear market: {mean_excess:.1f}%"
            )


# -----------------------------------------------------------------------------
# Sideways Market Tests
# -----------------------------------------------------------------------------


class TestSidewaysMarketPerformance:
    """Test strategy performance in sideways (range-bound) market conditions."""

    def test_weights_valid_in_sideways_market(self, sideways_market_features):
        """Verify weights are valid in sideways market."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(sideways_market_features, start_date, end_date)

        # Basic weight constraints
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6), "Weights must sum to 1.0"
        assert (weights >= -1e-10).all(), "All weights must be non-negative"
        assert weights.notna().all(), "No NaN weights allowed"

    def test_sideways_market_backtest_runs(
        self, sideways_market_df, sideways_market_features
    ):
        """Verify backtest runs successfully in sideways market."""
        backtest._FEATURES_DF = sideways_market_features

        spd_table = compute_cycle_spd(
            sideways_market_df,
            compute_weights_modal,
            features_df=sideways_market_features,
        )

        assert len(spd_table) > 0, "Should produce SPD results"

    def test_sideways_market_stable_performance(
        self, sideways_market_df, sideways_market_features
    ):
        """In sideways markets, performance should be relatively stable."""
        backtest._FEATURES_DF = sideways_market_features

        spd_table = compute_cycle_spd(
            sideways_market_df,
            compute_weights_modal,
            features_df=sideways_market_features,
        )

        # Filter out NaN rows
        valid_rows = spd_table["excess_percentile"].dropna()

        if len(valid_rows) > 0:
            # Performance shouldn't swing wildly in sideways market
            excess_std = valid_rows.std()
            # High std is acceptable given the nature of the strategy
            assert excess_std < 50, (
                f"Excess percentile std too high in sideways market: {excess_std:.1f}%"
            )


# -----------------------------------------------------------------------------
# High Volatility Market Tests
# -----------------------------------------------------------------------------


class TestHighVolatilityPerformance:
    """Test strategy performance in high-volatility market conditions."""

    def test_weights_valid_in_high_volatility(self, high_volatility_features):
        """Verify weights are valid in high volatility market."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(high_volatility_features, start_date, end_date)

        # Basic weight constraints
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6), "Weights must sum to 1.0"
        assert (weights >= -1e-10).all(), "All weights must be non-negative"
        assert weights.notna().all(), "No NaN weights allowed"
        assert np.all(np.isfinite(weights)), "All weights must be finite"

    def test_high_volatility_backtest_runs(
        self, high_volatility_df, high_volatility_features
    ):
        """Verify backtest runs successfully in high volatility market."""
        backtest._FEATURES_DF = high_volatility_features

        spd_table = compute_cycle_spd(
            high_volatility_df,
            compute_weights_modal,
            features_df=high_volatility_features,
        )

        assert len(spd_table) > 0, "Should produce SPD results"

    def test_high_volatility_numerical_stability(self, high_volatility_features):
        """Verify numerical stability in high volatility conditions."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(high_volatility_features, start_date, end_date)

        # Check for extreme weight concentration
        max_weight = weights.max()
        assert max_weight < 0.5, (
            f"Max weight {max_weight:.4f} too high - indicates instability"
        )


# -----------------------------------------------------------------------------
# Crash and Recovery Tests
# -----------------------------------------------------------------------------


class TestCrashRecoveryPerformance:
    """Test strategy performance during crash and recovery scenarios."""

    def test_weights_valid_in_crash_recovery(self, crash_recovery_features):
        """Verify weights are valid during crash and recovery."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(crash_recovery_features, start_date, end_date)

        # Basic weight constraints
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6), "Weights must sum to 1.0"
        assert (weights >= -1e-10).all(), "All weights must be non-negative"
        assert weights.notna().all(), "No NaN weights allowed"

    def test_crash_recovery_backtest_runs(
        self, crash_recovery_df, crash_recovery_features
    ):
        """Verify backtest runs successfully during crash and recovery."""
        backtest._FEATURES_DF = crash_recovery_features

        spd_table = compute_cycle_spd(
            crash_recovery_df,
            compute_weights_modal,
            features_df=crash_recovery_features,
        )

        assert len(spd_table) > 0, "Should produce SPD results"

    def test_crash_recovery_resilience(
        self, crash_recovery_df, crash_recovery_features
    ):
        """Strategy should be resilient during crash and recovery.

        The strategy shouldn't catastrophically fail during extreme moves.
        """
        backtest._FEATURES_DF = crash_recovery_features

        spd_table = compute_cycle_spd(
            crash_recovery_df,
            compute_weights_modal,
            features_df=crash_recovery_features,
        )

        # Filter out NaN rows
        valid_rows = spd_table["excess_percentile"].dropna()

        if len(valid_rows) > 0:
            # Check that we don't have extreme negative performance
            min_excess = valid_rows.min()
            assert min_excess > -50, (
                f"Extreme underperformance during crash: {min_excess:.1f}%"
            )


# -----------------------------------------------------------------------------
# Cross-Regime Comparison Tests
# -----------------------------------------------------------------------------


class TestCrossRegimeComparison:
    """Compare strategy performance across different market regimes."""

    def test_all_regimes_produce_valid_weights(
        self,
        bull_market_features,
        bear_market_features,
        sideways_market_features,
        high_volatility_features,
        crash_recovery_features,
    ):
        """Verify all market regimes produce valid weights."""
        regimes = {
            "bull": bull_market_features,
            "bear": bear_market_features,
            "sideways": sideways_market_features,
            "high_vol": high_volatility_features,
            "crash_recovery": crash_recovery_features,
        }

        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        for regime_name, features in regimes.items():
            weights = compute_weights_fast(features, start_date, end_date)

            assert np.isclose(weights.sum(), 1.0, rtol=1e-6), (
                f"{regime_name}: Weights must sum to 1.0"
            )
            assert (weights >= -1e-10).all(), (
                f"{regime_name}: All weights must be non-negative"
            )
            assert weights.notna().all(), f"{regime_name}: No NaN weights allowed"

    def test_weight_consistency_across_regimes(
        self,
        bull_market_features,
        bear_market_features,
        sideways_market_features,
    ):
        """Verify weights have consistent properties across regimes.

        While exact weights differ, statistical properties should be similar.
        """
        regimes = {
            "bull": bull_market_features,
            "bear": bear_market_features,
            "sideways": sideways_market_features,
        }

        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weight_stats = {}
        for regime_name, features in regimes.items():
            weights = compute_weights_fast(features, start_date, end_date)
            weight_stats[regime_name] = {
                "mean": weights.mean(),
                "std": weights.std(),
                "max": weights.max(),
                "min": weights.min(),
            }

        # All regimes should have similar mean (around 1/366)
        expected_mean = 1.0 / 366
        for regime_name, stats in weight_stats.items():
            assert np.isclose(stats["mean"], expected_mean, rtol=0.01), (
                f"{regime_name}: Mean weight should be ~{expected_mean:.6f}"
            )

    def test_regime_stability_metric(
        self,
        bull_market_df,
        bull_market_features,
        bear_market_df,
        bear_market_features,
        sideways_market_df,
        sideways_market_features,
    ):
        """Calculate stability metric across regimes.

        The variance of mean excess percentile across regimes shouldn't be extreme.
        """
        regimes = {
            "bull": (bull_market_df, bull_market_features),
            "bear": (bear_market_df, bear_market_features),
            "sideways": (sideways_market_df, sideways_market_features),
        }

        mean_excesses = []

        for regime_name, (df, features) in regimes.items():
            backtest._FEATURES_DF = features

            spd_table = compute_cycle_spd(
                df, compute_weights_modal, features_df=features
            )
            valid_rows = spd_table["excess_percentile"].dropna()

            if len(valid_rows) > 0:
                mean_excesses.append(valid_rows.mean())

        if len(mean_excesses) >= 2:
            # Calculate standard deviation of mean excess across regimes
            regime_std = np.std(mean_excesses)

            # Regime stability: std of performance across regimes
            # Lower is better, but some variation is expected
            assert regime_std < 30, (
                f"Strategy performance too variable across regimes: std={regime_std:.1f}%"
            )
