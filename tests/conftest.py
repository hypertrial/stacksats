"""Shared fixtures and mocking utilities for the test suite."""

import json
import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# -----------------------------------------------------------------------------
# Sample Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_btc_prices():
    """Generate realistic BTC price data for testing."""
    # Create date range from 2020 to 2025
    dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")

    # Generate realistic-looking price data with trend and volatility
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.Series(prices, index=dates, name="PriceUSD")


@pytest.fixture
def sample_btc_df(sample_btc_prices):
    """Create a sample BTC DataFrame similar to CoinMetrics data."""
    df = pd.DataFrame({"PriceUSD": sample_btc_prices})
    df["PriceUSD_coinmetrics"] = df["PriceUSD"]

    # Add MVRV data (CapMVRVCur) - cycles between 0.5 and 4.0 over ~4 years
    n = len(df)
    np.random.seed(123)
    mvrv_base = 1.5 + 1.2 * np.sin(np.arange(n) * 2 * np.pi / 1461)  # 4-year cycle
    mvrv_noise = np.random.normal(0, 0.15, n)
    df["CapMVRVCur"] = np.clip(mvrv_base + mvrv_noise, 0.5, 5.0)

    df.index.name = "time"
    return df


@pytest.fixture
def sample_features_df(sample_btc_df):
    """Create precomputed features DataFrame."""
    from stacksats.model_development import precompute_features

    return precompute_features(sample_btc_df)


@pytest.fixture
def sample_spd_df():
    """Create sample SPD backtest results DataFrame."""
    windows = [
        "2020-01-01 → 2021-01-01",
        "2020-02-01 → 2021-02-01",
        "2020-03-01 → 2021-03-01",
        "2020-04-01 → 2021-04-01",
        "2020-05-01 → 2021-05-01",
    ]

    data = {
        "min_sats_per_dollar": [1000, 1100, 900, 1050, 950],
        "max_sats_per_dollar": [5000, 5500, 4500, 5200, 4800],
        "uniform_sats_per_dollar": [2500, 2800, 2300, 2600, 2400],
        "dynamic_sats_per_dollar": [2800, 3100, 2500, 2900, 2600],
        "uniform_percentile": [37.5, 38.6, 38.9, 37.3, 37.7],
        "dynamic_percentile": [45.0, 45.5, 44.4, 44.6, 42.9],
        "excess_percentile": [7.5, 6.9, 5.5, 7.3, 5.2],
    }

    return pd.DataFrame(data, index=windows)


# -----------------------------------------------------------------------------
# Mock CoinMetrics CSV Data
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_coinmetrics_csv():
    """Generate sample CoinMetrics CSV data."""
    dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"time": dates.strftime("%Y-%m-%d"), "PriceUSD": prices})

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


# -----------------------------------------------------------------------------
# Network Mocking Fixtures
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Database Mocking Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_db_connection(mocker):
    """Mock psycopg2 database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Setup cursor context manager
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    # Default cursor behavior
    mock_cursor.fetchone.return_value = (0,)  # Empty table by default
    mock_cursor.fetchall.return_value = []
    mock_cursor.rowcount = 0

    mocker.patch("stacksats.export_weights.psycopg2.connect", return_value=mock_conn)
    mocker.patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}
    )

    return mock_conn, mock_cursor


# -----------------------------------------------------------------------------
# File System Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def snapshot_dir():
    """Return the path to the snapshots directory."""
    return Path(__file__).parent / "snapshots"


# -----------------------------------------------------------------------------
# Snapshot/Golden Test Utilities
# -----------------------------------------------------------------------------


class SnapshotManager:
    """Utility for managing snapshot/golden tests."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir

    def load(self, filename: str):
        """Load a snapshot file."""
        filepath = self.snapshot_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert to appropriate pandas type
        if "index" in data and "values" in data:
            return pd.Series(data["values"], index=pd.to_datetime(data["index"]))
        return data

    def save(self, data, filename: str):
        """Save data to a snapshot file."""
        filepath = self.snapshot_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.Series):
            save_data = {
                "index": [str(i) for i in data.index],
                "values": data.values.tolist(),
            }
        elif isinstance(data, pd.DataFrame):
            save_data = data.to_dict(orient="list")
        else:
            save_data = data

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)


@pytest.fixture
def snapshot(snapshot_dir):
    """Provide a snapshot manager for golden tests."""
    return SnapshotManager(snapshot_dir)


# -----------------------------------------------------------------------------
# Helper Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_metrics():
    """Generate sample metrics dictionary for testing."""
    return {
        "score": 65.5,
        "win_rate": 72.0,
        "exp_decay_percentile": 59.0,
        "uniform_exp_decay_percentile": 47.5,
        "exp_decay_multiple_vs_uniform": 1.2421,
        "mean_excess": 5.2,
        "median_excess": 4.8,
        "relative_improvement_pct_mean": 12.5,
        "relative_improvement_pct_median": 11.2,
        "mean_ratio": 1.12,
        "median_ratio": 1.10,
        "total_windows": 100,
        "wins": 72,
        "losses": 28,
    }


# -----------------------------------------------------------------------------
# Simulation Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simulation_price_generator():
    """Generate synthetic BTC prices using geometric Brownian motion.

    Returns a function that generates prices from start_date through end_date.
    """

    def _generate_prices(
        start_date,
        end_date,
        initial_price=100000.0,
        drift=0.0001,
        volatility=0.03,
        seed=42,
    ):
        """Generate synthetic BTC prices.

        Args:
            start_date: Start date (pd.Timestamp or string)
            end_date: End date (pd.Timestamp or string)
            initial_price: Starting price in USD
            drift: Daily drift (mean return per day)
            volatility: Daily volatility (std dev of returns)
            seed: Random seed for reproducibility

        Returns:
            pd.Series with prices indexed by date
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        np.random.seed(seed)

        # Generate log returns using geometric Brownian motion
        dt = 1.0  # 1 day
        returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), len(dates))

        # Convert to prices
        log_prices = np.log(initial_price) + np.cumsum(returns)
        prices = np.exp(log_prices)

        # Ensure prices stay within reasonable bounds
        prices = np.clip(prices, 1000.0, 1000000.0)

        return pd.Series(prices, index=dates, name="PriceUSD")

    return _generate_prices


@pytest.fixture
def simulation_btc_df(simulation_price_generator):
    """Create a BTC DataFrame with synthetic prices extending into the future.

    Generates prices from 2020-01-01 through 2028-12-31 for simulation testing.
    """
    start_date = "2020-01-01"
    end_date = "2028-12-31"
    prices = simulation_price_generator(start_date, end_date, initial_price=50000.0)

    df = pd.DataFrame({"PriceUSD": prices})
    df["PriceUSD_coinmetrics"] = df["PriceUSD"]
    df.index.name = "time"
    return df


@pytest.fixture
def simulation_mock_db(mocker):
    """Enhanced mock database that tracks inserted/updated rows for simulation tests.

    Returns a tuple of (mock_conn, mock_cursor, db_state_dict) where db_state_dict
    tracks the actual data that would be in the database.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Setup cursor context manager
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    # Track database state
    db_state = {
        "rows": {},  # Key: (day_index, start_date, end_date, date), Value: row dict
        "count": 0,
        "inserted_count": 0,
        "updated_count": 0,
    }

    # Default cursor behavior
    def _fetchone_count():
        return (db_state["count"],)

    mock_cursor.fetchone.side_effect = _fetchone_count
    mock_cursor.fetchall.return_value = []
    mock_cursor.rowcount = 0

    # Track rowcount changes
    def _track_rowcount():
        return db_state["count"]

    mocker.patch("stacksats.export_weights.psycopg2.connect", return_value=mock_conn)
    mocker.patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}
    )

    return mock_conn, mock_cursor, db_state


# -----------------------------------------------------------------------------
# Forward-Looking Bias Test Fixtures
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Validation Test Fixtures (from test_validate_neondb.py)
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_btc_df_validate():
    """Create sample BTC price data for validation testing."""
    from tests.test_helpers import PRICE_COL, SAMPLE_END, SAMPLE_START

    dates = pd.date_range(SAMPLE_START, SAMPLE_END, freq="D")
    np.random.seed(42)

    # Simulate realistic BTC prices with trend and volatility
    base_price = 50000
    trend = np.linspace(0, 50000, len(dates))
    noise = np.random.randn(len(dates)) * 2000
    prices = np.maximum(base_price + trend + noise, 10000)

    return pd.DataFrame({PRICE_COL: prices}, index=dates)


@pytest.fixture
def sample_features_df_validate(sample_btc_df_validate):
    """Precompute features for validation sample data."""
    from stacksats.model_development import precompute_features

    return precompute_features(sample_btc_df_validate)


@pytest.fixture
def sample_weights_df(sample_features_df_validate, sample_btc_df_validate):
    """Generate sample weights for testing with multiple date ranges."""
    from stacksats.export_weights import process_start_date_batch
    from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
    from tests.test_helpers import PRICE_COL

    start_date = pd.Timestamp("2025-01-01")
    end_date = start_date + pd.Timedelta(days=ALLOCATION_SPAN_DAYS - 1)
    end_dates = [end_date]
    current_date = end_date

    return process_start_date_batch(
        start_date,
        end_dates,
        sample_features_df_validate,
        sample_btc_df_validate,
        current_date,
        PRICE_COL,
    )


# Note: sample_features_df and sample_btc_df are already defined earlier in conftest.py
# Validation tests should use sample_features_df_validate and sample_btc_df_validate
# to avoid conflicts with other test fixtures


# -----------------------------------------------------------------------------
# Edge Case Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def all_wins_spd_df():
    """SPD DataFrame where dynamic always beats uniform (100% win rate)."""
    windows = [
        "2020-01-01 → 2021-01-01",
        "2020-02-01 → 2021-02-01",
        "2020-03-01 → 2021-03-01",
    ]
    data = {
        "min_sats_per_dollar": [1000, 1100, 900],
        "max_sats_per_dollar": [5000, 5500, 4500],
        "uniform_sats_per_dollar": [2500, 2800, 2300],
        "dynamic_sats_per_dollar": [3500, 3800, 3300],
        "uniform_percentile": [37.5, 38.6, 38.9],
        "dynamic_percentile": [62.5, 61.4, 66.7],
        "excess_percentile": [25.0, 22.8, 27.8],
    }
    return pd.DataFrame(data, index=windows)


@pytest.fixture
def all_losses_spd_df():
    """SPD DataFrame where dynamic always loses to uniform (0% win rate)."""
    windows = [
        "2020-01-01 → 2021-01-01",
        "2020-02-01 → 2021-02-01",
        "2020-03-01 → 2021-03-01",
    ]
    data = {
        "min_sats_per_dollar": [1000, 1100, 900],
        "max_sats_per_dollar": [5000, 5500, 4500],
        "uniform_sats_per_dollar": [2500, 2800, 2300],
        "dynamic_sats_per_dollar": [2000, 2200, 1800],
        "uniform_percentile": [37.5, 38.6, 38.9],
        "dynamic_percentile": [25.0, 25.0, 25.0],
        "excess_percentile": [-12.5, -13.6, -13.9],
    }
    return pd.DataFrame(data, index=windows)


@pytest.fixture
def single_window_spd_df():
    """SPD DataFrame with only one window for edge case testing."""
    windows = ["2020-01-01 → 2021-01-01"]
    data = {
        "min_sats_per_dollar": [1000],
        "max_sats_per_dollar": [5000],
        "uniform_sats_per_dollar": [2500],
        "dynamic_sats_per_dollar": [2800],
        "uniform_percentile": [37.5],
        "dynamic_percentile": [45.0],
        "excess_percentile": [7.5],
    }
    return pd.DataFrame(data, index=windows)
