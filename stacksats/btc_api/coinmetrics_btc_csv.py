"""CoinMetrics BTC CSV data fetcher.

This module provides functionality to download and process the CoinMetrics
Bitcoin CSV data dump from their public GitHub repository.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# CoinMetrics public CSV repository URL
COINMETRICS_BTC_CSV_URL = (
    "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv"
)


def normalize_coinmetrics_btc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CoinMetrics rows to daily, unique, sorted time index."""
    if "time" not in df.columns:
        raise ValueError("CSV missing required 'time' column")

    normalized = df.copy()
    normalized["time"] = pd.to_datetime(normalized["time"])
    normalized.set_index("time", inplace=True)
    normalized.index = normalized.index.normalize().tz_localize(None)
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized


def parse_coinmetrics_btc_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """Parse CoinMetrics CSV bytes and normalize index semantics."""
    try:
        raw_df = pd.read_csv(BytesIO(csv_bytes))
    except Exception as e:
        logging.error(f"Failed to parse CSV data: {e}")
        raise ValueError(f"Invalid CSV data: {e}") from e
    return normalize_coinmetrics_btc_dataframe(raw_df)


def fetch_coinmetrics_btc_csv(
    url: Optional[str] = None, save_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """Fetch CoinMetrics BTC CSV data dump.

    Downloads the complete Bitcoin historical data CSV from CoinMetrics'
    public GitHub repository and optionally saves it to disk.

    Args:
        url: Optional custom URL for the CSV. Defaults to CoinMetrics public repo.
        save_path: Optional path to save the CSV file. If None, file is not saved.

    Returns:
        DataFrame containing CoinMetrics BTC data with 'time' column as index.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the CSV data is invalid or missing required columns.
    """
    csv_url = url or COINMETRICS_BTC_CSV_URL

    logging.info(f"Fetching CoinMetrics BTC CSV from {csv_url}...")

    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch CoinMetrics CSV: {e}")
        raise

    df = parse_coinmetrics_btc_csv_bytes(response.content)

    # Validate required columns
    if "PriceUSD" not in df.columns:
        raise ValueError("CSV missing required 'PriceUSD' column")

    # Save to file if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        logging.info(f"Saved CoinMetrics CSV to {save_path}")

    logging.info(
        f"Loaded CoinMetrics data: {len(df)} rows, "
        f"{df.index.min().date()} to {df.index.max().date()}"
    )

    return df


if __name__ == "__main__":
    # Example usage: fetch and display basic info
    df = fetch_coinmetrics_btc_csv()
    print("\nCoinMetrics BTC Data Summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Columns: {', '.join(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
