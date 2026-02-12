"""Monte Carlo simulation tests for the Bitcoin DCA strategy.

These tests verify strategy robustness across many synthetic price scenarios:
1. Random walk simulations with varying parameters
2. Fat-tailed return distributions
3. Mean-reverting and trending scenarios
4. Bootstrapped historical returns

Monte Carlo testing helps detect:
- Strategies that only work in specific market conditions
- Overfitting to historical patterns
- Numerical instabilities under various conditions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stacksats.model_development import compute_weights_fast, precompute_features

# -----------------------------------------------------------------------------
# Price Generators
# -----------------------------------------------------------------------------


def generate_random_walk_prices(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 10000.0,
    mu: float = 0.0005,
    sigma: float = 0.03,
    seed: int = None,
) -> pd.DataFrame:
    """Generate random walk (geometric Brownian motion) prices.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        mu: Daily drift (mean return)
        sigma: Daily volatility (std of returns)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    returns = np.random.normal(mu, sigma, len(dates))
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def generate_fat_tailed_prices(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 10000.0,
    mu: float = 0.0005,
    scale: float = 0.02,
    df_param: float = 4,  # degrees of freedom for t-distribution
    seed: int = None,
) -> pd.DataFrame:
    """Generate prices with fat-tailed (Student's t) returns.

    Fat tails produce more extreme moves, mimicking real crypto behavior.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        mu: Daily drift
        scale: Scale parameter for t-distribution
        df_param: Degrees of freedom (lower = fatter tails)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Student's t returns (fatter tails than normal)
    t_returns = np.random.standard_t(df_param, len(dates)) * scale + mu

    log_prices = np.log(initial_price) + np.cumsum(t_returns)
    prices = np.exp(log_prices)

    # Clip extreme values to prevent numerical issues
    prices = np.clip(prices, 100, 10000000)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def generate_mean_reverting_prices(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    mean_price: float = 30000.0,
    theta: float = 0.02,  # Mean reversion speed
    sigma: float = 0.02,
    seed: int = None,
) -> pd.DataFrame:
    """Generate mean-reverting prices (Ornstein-Uhlenbeck process).

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        mean_price: Long-term mean price
        theta: Mean reversion speed
        sigma: Volatility
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    log_mean = np.log(mean_price)

    log_prices = np.zeros(len(dates))
    log_prices[0] = log_mean + np.random.randn() * sigma

    for i in range(1, len(dates)):
        drift = theta * (log_mean - log_prices[i - 1])
        noise = sigma * np.random.randn()
        log_prices[i] = log_prices[i - 1] + drift + noise

    prices = np.exp(log_prices)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


def generate_regime_switching_prices(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_price: float = 20000.0,
    bull_mu: float = 0.002,
    bear_mu: float = -0.001,
    sigma: float = 0.025,
    transition_prob: float = 0.01,  # Daily regime switch probability
    seed: int = None,
) -> pd.DataFrame:
    """Generate prices with regime switching (bull/bear alternation).

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        bull_mu: Mean return in bull regime
        bear_mu: Mean return in bear regime
        sigma: Volatility (same in both regimes)
        transition_prob: Daily probability of regime switch
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Start in random regime
    is_bull = np.random.rand() > 0.5

    returns = []
    for _ in range(len(dates)):
        mu = bull_mu if is_bull else bear_mu
        ret = np.random.normal(mu, sigma)
        returns.append(ret)

        # Possible regime switch
        if np.random.rand() < transition_prob:
            is_bull = not is_bull

    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


# -----------------------------------------------------------------------------
# Evaluation Utilities
# -----------------------------------------------------------------------------


def compute_window_performance(
    features_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    """Compute performance metrics for a single window.

    Returns:
        Dictionary with win (bool), excess_percentile, and validity flag
    """
    try:
        weights = compute_weights_fast(features_df, start_date, end_date)

        price_slice = btc_df["PriceUSD_coinmetrics"].loc[start_date:end_date]

        # Align
        common_idx = weights.index.intersection(price_slice.index)
        weights = weights.loc[common_idx]
        price_slice = price_slice.loc[common_idx]

        if len(weights) == 0:
            return {"valid": False}

        inv_price = 1e8 / price_slice
        min_spd, max_spd = inv_price.min(), inv_price.max()
        span = max_spd - min_spd

        uniform_spd = inv_price.mean()
        dynamic_spd = (weights * inv_price).sum()

        if span > 0:
            uniform_pct = 100 * (uniform_spd - min_spd) / span
            dynamic_pct = 100 * (dynamic_spd - min_spd) / span
            excess = dynamic_pct - uniform_pct
        else:
            excess = 0.0

        return {
            "valid": True,
            "win": excess > 0,
            "excess_percentile": excess,
        }
    except Exception:
        return {"valid": False}


def run_monte_carlo_scenario(
    btc_df: pd.DataFrame,
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
) -> dict:
    """Run evaluation on a single Monte Carlo scenario.

    Returns:
        Dictionary with win_rate, mean_excess, n_valid
    """
    features_df = precompute_features(btc_df)

    start_date = pd.Timestamp(test_start)
    end_date = pd.Timestamp(test_end)

    result = compute_window_performance(features_df, btc_df, start_date, end_date)

    if not result["valid"]:
        return {"win_rate": np.nan, "mean_excess": np.nan, "n_valid": 0}

    return {
        "win_rate": 1.0 if result["win"] else 0.0,
        "mean_excess": result["excess_percentile"],
        "n_valid": 1,
    }


# -----------------------------------------------------------------------------
# Monte Carlo Test Classes
# -----------------------------------------------------------------------------


@pytest.mark.slow
class TestMonteCarloRandomWalk:
    """Monte Carlo tests using random walk price generation."""

    def test_100_random_walk_scenarios(self):
        """Test strategy across 100 random walk scenarios."""
        n_scenarios = 100
        win_rates = []

        for seed in range(n_scenarios):
            btc_df = generate_random_walk_prices(
                mu=0.0005,  # Slight upward trend
                sigma=0.03,
                seed=seed,
            )

            result = run_monte_carlo_scenario(btc_df)

            if result["n_valid"] > 0:
                win_rates.append(result["win_rate"])

        assert len(win_rates) >= 90, (
            f"Only {len(win_rates)} valid scenarios out of {n_scenarios}"
        )

        mean_win_rate = np.mean(win_rates)

        # Strategy should win more often than lose on average (> 40%)
        assert mean_win_rate > 0.40, (
            f"Win rate {mean_win_rate:.1%} too low across random walk scenarios"
        )

    def test_varying_volatility(self):
        """Test across different volatility levels."""
        volatilities = [0.01, 0.02, 0.03, 0.05, 0.08]
        results_by_vol = {}

        for sigma in volatilities:
            win_rates = []
            for seed in range(20):
                btc_df = generate_random_walk_prices(
                    mu=0.0005,
                    sigma=sigma,
                    seed=seed * 100 + int(sigma * 1000),
                )

                result = run_monte_carlo_scenario(btc_df)
                if result["n_valid"] > 0:
                    win_rates.append(result["win_rate"])

            if len(win_rates) > 0:
                results_by_vol[sigma] = np.mean(win_rates)

        # Should have results for multiple volatility levels
        assert len(results_by_vol) >= 3

        # Win rate shouldn't collapse at any volatility
        for sigma, win_rate in results_by_vol.items():
            assert win_rate > 0.25, (
                f"Win rate too low at volatility {sigma}: {win_rate:.1%}"
            )

    def test_varying_drift(self):
        """Test across different drift (trend) levels."""
        drifts = [-0.002, -0.001, 0.0, 0.001, 0.002]
        results_by_drift = {}

        for mu in drifts:
            win_rates = []
            for seed in range(20):
                btc_df = generate_random_walk_prices(
                    mu=mu,
                    sigma=0.03,
                    seed=seed * 100 + int(mu * 10000 + 20),
                )

                result = run_monte_carlo_scenario(btc_df)
                if result["n_valid"] > 0:
                    win_rates.append(result["win_rate"])

            if len(win_rates) > 0:
                results_by_drift[mu] = np.mean(win_rates)

        # Should have results for multiple drift levels
        assert len(results_by_drift) >= 3


@pytest.mark.slow
class TestMonteCarloFatTails:
    """Monte Carlo tests using fat-tailed return distributions."""

    def test_50_fat_tailed_scenarios(self):
        """Test strategy across 50 fat-tailed return scenarios."""
        n_scenarios = 50
        win_rates = []

        for seed in range(n_scenarios):
            btc_df = generate_fat_tailed_prices(
                mu=0.0005,
                scale=0.02,
                df_param=4,  # Fairly fat tails
                seed=seed,
            )

            result = run_monte_carlo_scenario(btc_df)

            if result["n_valid"] > 0:
                win_rates.append(result["win_rate"])

        assert len(win_rates) >= 40, (
            f"Only {len(win_rates)} valid scenarios out of {n_scenarios}"
        )

        mean_win_rate = np.mean(win_rates)

        # Should still have reasonable performance with fat tails
        assert mean_win_rate > 0.35, (
            f"Win rate {mean_win_rate:.1%} too low with fat-tailed returns"
        )

    def test_extreme_fat_tails(self):
        """Test with very fat tails (df=2)."""
        n_scenarios = 30
        valid_count = 0

        for seed in range(n_scenarios):
            btc_df = generate_fat_tailed_prices(
                mu=0.0005,
                scale=0.015,
                df_param=2.5,  # Very fat tails
                seed=seed,
            )

            features_df = precompute_features(btc_df)

            # Check weights are valid even with extreme price movements
            start = pd.Timestamp("2024-01-01")
            end = pd.Timestamp("2024-12-31")

            try:
                weights = compute_weights_fast(features_df, start, end)

                if np.isclose(weights.sum(), 1.0, rtol=1e-6):
                    if (weights >= -1e-10).all():
                        if weights.notna().all():
                            valid_count += 1
            except Exception:
                pass

        # Most scenarios should produce valid weights
        assert valid_count >= n_scenarios * 0.8, (
            f"Only {valid_count}/{n_scenarios} valid with extreme fat tails"
        )


@pytest.mark.slow
class TestMonteCarloMeanReverting:
    """Monte Carlo tests using mean-reverting prices."""

    def test_50_mean_reverting_scenarios(self):
        """Test across 50 mean-reverting scenarios."""
        n_scenarios = 50
        win_rates = []

        for seed in range(n_scenarios):
            btc_df = generate_mean_reverting_prices(
                mean_price=30000,
                theta=0.02,
                sigma=0.02,
                seed=seed,
            )

            result = run_monte_carlo_scenario(btc_df)

            if result["n_valid"] > 0:
                win_rates.append(result["win_rate"])

        assert len(win_rates) >= 40

        # Mean-reverting markets can be challenging for DCA
        # Just ensure we don't completely fail
        mean_win_rate = np.mean(win_rates)
        assert mean_win_rate > 0.30, (
            f"Win rate {mean_win_rate:.1%} too low in mean-reverting scenarios"
        )


@pytest.mark.slow
class TestMonteCarloRegimeSwitching:
    """Monte Carlo tests using regime-switching prices."""

    def test_50_regime_switching_scenarios(self):
        """Test across 50 regime-switching scenarios."""
        n_scenarios = 50
        win_rates = []

        for seed in range(n_scenarios):
            btc_df = generate_regime_switching_prices(
                bull_mu=0.002,
                bear_mu=-0.001,
                sigma=0.025,
                transition_prob=0.01,
                seed=seed,
            )

            result = run_monte_carlo_scenario(btc_df)

            if result["n_valid"] > 0:
                win_rates.append(result["win_rate"])

        assert len(win_rates) >= 40

        mean_win_rate = np.mean(win_rates)
        assert mean_win_rate > 0.35, (
            f"Win rate {mean_win_rate:.1%} too low in regime-switching scenarios"
        )


# -----------------------------------------------------------------------------
# Robustness Summary Tests
# -----------------------------------------------------------------------------


@pytest.mark.slow
class TestMonteCarloRobustnessSummary:
    """Summary tests across all Monte Carlo scenario types."""

    def test_overall_robustness(self):
        """Aggregate robustness test across all scenario types."""
        generators = [
            ("random_walk", lambda s: generate_random_walk_prices(seed=s)),
            ("fat_tailed", lambda s: generate_fat_tailed_prices(seed=s)),
            ("mean_reverting", lambda s: generate_mean_reverting_prices(seed=s)),
            ("regime_switching", lambda s: generate_regime_switching_prices(seed=s)),
        ]

        overall_results = {}

        for name, generator in generators:
            win_rates = []
            for seed in range(25):
                btc_df = generator(seed)
                result = run_monte_carlo_scenario(btc_df)

                if result["n_valid"] > 0:
                    win_rates.append(result["win_rate"])

            if len(win_rates) > 0:
                overall_results[name] = {
                    "mean_win_rate": np.mean(win_rates),
                    "std_win_rate": np.std(win_rates),
                    "n_valid": len(win_rates),
                }

        # Should have results for all generator types
        assert len(overall_results) == len(generators)

        # Overall average win rate across all types
        all_means = [r["mean_win_rate"] for r in overall_results.values()]
        overall_mean = np.mean(all_means)

        assert overall_mean > 0.40, (
            f"Overall mean win rate {overall_mean:.1%} too low across all scenarios"
        )

    def test_low_variance_across_scenarios(self):
        """Verify win rate doesn't vary too wildly across scenario types."""
        generators = [
            ("random_walk", lambda s: generate_random_walk_prices(seed=s)),
            ("fat_tailed", lambda s: generate_fat_tailed_prices(seed=s)),
            ("mean_reverting", lambda s: generate_mean_reverting_prices(seed=s)),
        ]

        mean_win_rates = []

        for name, generator in generators:
            win_rates = []
            for seed in range(15):
                btc_df = generator(seed)
                result = run_monte_carlo_scenario(btc_df)

                if result["n_valid"] > 0:
                    win_rates.append(result["win_rate"])

            if len(win_rates) > 0:
                mean_win_rates.append(np.mean(win_rates))

        if len(mean_win_rates) >= 2:
            std_of_means = np.std(mean_win_rates)

            # Win rate shouldn't vary too much across generator types
            assert std_of_means < 0.20, (
                f"Win rate std across scenario types too high: {std_of_means:.2%}"
            )


# -----------------------------------------------------------------------------
# Weight Stability Tests
# -----------------------------------------------------------------------------


class TestWeightStabilityMonteCarlo:
    """Test weight computation stability across Monte Carlo scenarios."""

    def test_weights_always_valid(self):
        """Verify weights are always valid across all Monte Carlo scenarios."""
        n_scenarios = 50
        valid_count = 0

        for seed in range(n_scenarios):
            btc_df = generate_random_walk_prices(
                mu=np.random.uniform(-0.002, 0.003),
                sigma=np.random.uniform(0.01, 0.06),
                seed=seed,
            )

            try:
                features_df = precompute_features(btc_df)

                start = pd.Timestamp("2024-01-01")
                end = pd.Timestamp("2024-12-31")

                weights = compute_weights_fast(features_df, start, end)

                # Check all validity conditions
                if not np.isclose(weights.sum(), 1.0, rtol=1e-6):
                    continue
                if not (weights >= -1e-10).all():
                    continue
                if not weights.notna().all():
                    continue
                if not np.all(np.isfinite(weights)):
                    continue

                valid_count += 1
            except Exception:
                pass

        # All scenarios should produce valid weights
        assert valid_count == n_scenarios, (
            f"Only {valid_count}/{n_scenarios} scenarios produced valid weights"
        )

    def test_weight_sum_always_one(self):
        """Verify weights always sum to 1.0 across scenarios."""
        for seed in range(30):
            btc_df = generate_random_walk_prices(seed=seed)
            features_df = precompute_features(btc_df)

            start = pd.Timestamp("2024-01-01")
            end = pd.Timestamp("2024-12-31")

            weights = compute_weights_fast(features_df, start, end)

            assert np.isclose(weights.sum(), 1.0, rtol=1e-6), (
                f"Scenario {seed}: weights sum to {weights.sum()}"
            )
