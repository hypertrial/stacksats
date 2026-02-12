"""Robust BTC price fetching with retry logic and multiple API sources.

This module provides a reliable way to fetch current BTC-USD prices with:
- Retry logic with exponential backoff
- Multiple API sources as fallbacks
- Price validation and sanity checks
"""

import logging
from typing import Optional
from io import BytesIO
from functools import lru_cache

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# CoinMetrics CSV URL
COINMETRICS_BTC_CSV_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

# Price validation constants
MIN_BTC_PRICE = 1000.0  # Minimum reasonable BTC price (USD)
MAX_BTC_PRICE = 1000000.0  # Maximum reasonable BTC price (USD)
MAX_PRICE_CHANGE_PCT = 20.0  # Maximum reasonable day-over-day change (%)


def fetch_price_coingecko() -> float:
    """Fetch BTC price from CoinGecko API.

    Returns:
        float: BTC price in USD

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the response format is invalid
    """
    response = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": "bitcoin", "vs_currencies": "usd"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if "bitcoin" not in data or "usd" not in data["bitcoin"]:
        raise ValueError(f"Invalid CoinGecko response format: {data}")

    price = float(data["bitcoin"]["usd"])
    logger.debug(f"CoinGecko returned price: ${price:,.2f}")
    return price


def fetch_price_coinbase() -> float:
    """Fetch BTC price from Coinbase API.

    Returns:
        float: BTC price in USD

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the response format is invalid
    """
    response = requests.get(
        "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if "data" not in data or "amount" not in data["data"]:
        raise ValueError(f"Invalid Coinbase response format: {data}")

    price = float(data["data"]["amount"])
    logger.debug(f"Coinbase returned price: ${price:,.2f}")
    return price


def fetch_price_bitstamp() -> float:
    """Fetch BTC price from Bitstamp API.

    Returns:
        float: BTC price in USD

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the response format is invalid
    """
    response = requests.get(
        "https://www.bitstamp.net/api/v2/ticker/btcusd/",
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if "last" not in data:
        raise ValueError(f"Invalid Bitstamp response format: {data}")

    price = float(data["last"])
    logger.debug(f"Bitstamp returned price: ${price:,.2f}")
    return price


def fetch_price_kraken() -> float:
    """Fetch BTC price from Kraken API.

    Returns:
        float: BTC price in USD

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the response format is invalid
    """
    response = requests.get(
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise ValueError(f"Kraken API error: {data['error']}")

    # Kraken uses XXBTZUSD as the pair name
    result = data.get("result", {})
    pair_data = result.get("XXBTZUSD") or result.get("XBTUSD")

    if not pair_data or "c" not in pair_data:
        raise ValueError(f"Invalid Kraken response format: {data}")

    # "c" is the last trade closed array [price, lot volume]
    price = float(pair_data["c"][0])
    logger.debug(f"Kraken returned price: ${price:,.2f}")
    return price


def fetch_price_binance() -> float:
    """Fetch BTC price from Binance API.

    Returns:
        float: BTC price in USD

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the response format is invalid
    """
    response = requests.get(
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if "price" not in data:
        raise ValueError(f"Invalid Binance response format: {data}")

    price = float(data["price"])
    logger.debug(f"Binance returned price: ${price:,.2f}")
    return price


@lru_cache(maxsize=1)
def _load_coinmetrics_data() -> pd.DataFrame:
    """Load CoinMetrics BTC CSV data with caching.
    
    Returns:
        DataFrame with 'time' as index and 'PriceUSD' column
    """
    logger.debug("Loading CoinMetrics BTC data from GitHub...")
    response = requests.get(COINMETRICS_BTC_CSV_URL, timeout=30)
    response.raise_for_status()
    
    df = pd.read_csv(BytesIO(response.content))
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.index = df.index.normalize().tz_localize(None)
    
    # Remove duplicates and sort
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    
    if "PriceUSD" not in df.columns:
        raise ValueError("PriceUSD column not found in CoinMetrics data")
    
    logger.debug(f"Loaded CoinMetrics data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def fetch_historical_price_coinmetrics(date: "pd.Timestamp") -> float:
    """Fetch historical BTC price from CoinMetrics CSV data for a specific date.
    
    Args:
        date: pandas Timestamp for the date to fetch
        
    Returns:
        float: BTC price in USD
        
    Raises:
        ValueError: If date not found in CoinMetrics data or price is missing
    """
    df = _load_coinmetrics_data()
    date_normalized = date.normalize()
    
    if date_normalized not in df.index:
        raise ValueError(f"Date {date.date()} not found in CoinMetrics data")
    
    price = df.loc[date_normalized, "PriceUSD"]
    
    if pd.isna(price):
        raise ValueError(f"PriceUSD is missing for {date.date()} in CoinMetrics data")
    
    price = float(price)
    logger.debug(f"CoinMetrics historical ({date.date()}) returned price: ${price:,.2f}")
    return price


def fetch_historical_price_coingecko(date: "pd.Timestamp") -> float:
    """Fetch historical BTC price from CoinGecko API for a specific date.

    Args:
        date: pandas Timestamp for the date to fetch

    Returns:
        float: BTC price in USD
    """
    date_str = date.strftime("%d-%m-%Y")
    response = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/history",
        params={"date": date_str, "localization": "false"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if "market_data" not in data or "current_price" not in data["market_data"]:
        raise ValueError(f"Invalid CoinGecko historical response for {date}: {data}")

    price = float(data["market_data"]["current_price"]["usd"])
    logger.debug(f"CoinGecko historical ({date_str}) returned price: ${price:,.2f}")
    return price


def fetch_historical_price_binance(date: "pd.Timestamp") -> float:
    """Fetch historical BTC price from Binance API for a specific date.

    Uses the 1d candlestick data.

    Args:
        date: pandas Timestamp for the date to fetch

    Returns:
        float: BTC price in USD
    """
    # Start of the date in milliseconds
    start_time = int(date.timestamp() * 1000)
    # 1 day later
    end_time = start_time + (24 * 60 * 60 * 1000)

    response = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if not data or not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"No Binance historical data for {date}")

    # Index 4 is the close price
    price = float(data[0][4])
    logger.debug(f"Binance historical ({date.date()}) returned price: ${price:,.2f}")
    return price


def fetch_btc_price_historical(
    date: "pd.Timestamp",
    previous_price: Optional[float] = None,
) -> Optional[float]:
    """Fetch historical BTC price with retry logic and fallback sources.
    
    Uses CoinMetrics CSV data as the primary source, falling back to API sources
    if CoinMetrics data is unavailable for the requested date.

    Args:
        date: pandas Timestamp for the date to fetch
        previous_price: Optional previous price for validation

    Returns:
        float: BTC price in USD, or None if all sources fail
    """
    sources = [
        (lambda: fetch_historical_price_coinmetrics(date), "CoinMetrics Historical"),
        (lambda: fetch_historical_price_coingecko(date), "CoinGecko Historical"),
        (lambda: fetch_historical_price_binance(date), "Binance Historical"),
    ]

    for fetch_func, source_name in sources:
        try:
            # We use a longer wait for historical as rate limits are tighter
            price = fetch_func()
            if validate_price(price, previous_price):
                logger.debug(
                    f"Successfully fetched historical BTC price for {date.date()} from {source_name}: ${price:,.2f}"
                )
                return price
            else:
                logger.warning(
                    f"{source_name} returned invalid historical price for {date.date()}: ${price:,.2f}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to fetch historical BTC price for {date.date()} from {source_name}: {e}"
            )

    return None


def validate_price(price: float, previous_price: Optional[float] = None) -> bool:
    """Validate that a BTC price is reasonable.

    Args:
        price: The price to validate
        previous_price: Optional previous price for sanity check

    Returns:
        bool: True if price is valid, False otherwise
    """
    # Basic range check
    if not isinstance(price, (int, float)) or price <= 0:
        logger.warning(f"Invalid price type or non-positive: {price}")
        return False

    if price < MIN_BTC_PRICE or price > MAX_BTC_PRICE:
        logger.warning(
            f"Price {price:,.2f} outside reasonable range [{MIN_BTC_PRICE}, {MAX_BTC_PRICE}]"
        )
        return False

    # Sanity check against previous price if provided
    if previous_price is not None and previous_price > 0:
        change_pct = abs((price - previous_price) / previous_price) * 100
        if change_pct > MAX_PRICE_CHANGE_PCT:
            logger.warning(
                f"Price change {change_pct:.2f}% exceeds threshold {MAX_PRICE_CHANGE_PCT}% "
                f"(previous: ${previous_price:,.2f}, current: ${price:,.2f})"
            )
            # Don't reject, just warn - could be legitimate volatility

    return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _fetch_with_retry(fetch_func, source_name: str) -> float:
    """Fetch price from a single source with retry logic.

    Args:
        fetch_func: Function to call for fetching price
        source_name: Name of the source for logging

    Returns:
        float: BTC price in USD

    Raises:
        requests.RequestException: If all retries fail
    """
    try:
        price = fetch_func()
        logger.debug(f"Successfully fetched from {source_name}: ${price:,.2f}")
        return price
    except requests.RequestException as e:
        logger.debug(f"Request failed for {source_name}: {e}")
        raise
    except (ValueError, KeyError) as e:
        logger.warning(f"Invalid response format from {source_name}: {e}")
        raise ValueError(f"{source_name} returned invalid data: {e}") from e


def fetch_btc_price_robust(
    previous_price: Optional[float] = None,
    sources: Optional[list] = None,
) -> Optional[float]:
    """Fetch BTC price with retry logic and multiple fallback sources.

    This function tries multiple API sources in order, with retry logic for each.
    If all sources fail, returns None.

    Args:
        previous_price: Optional previous price for validation/sanity checks
        sources: Optional list of (fetch_func, source_name) tuples.
                 Defaults to [CoinGecko, Coinbase, Bitstamp]

    Returns:
        float: Current BTC price in USD, or None if all sources fail
    """
    if sources is None:
        sources = [
            (fetch_price_coingecko, "CoinGecko"),
            (fetch_price_coinbase, "Coinbase"),
            (fetch_price_bitstamp, "Bitstamp"),
            (fetch_price_kraken, "Kraken"),
            (fetch_price_binance, "Binance"),
        ]

    errors = []

    for fetch_func, source_name in sources:
        try:
            price = _fetch_with_retry(fetch_func, source_name)

            # Validate the price
            if validate_price(price, previous_price):
                logger.info(
                    f"Successfully fetched BTC price from {source_name}: ${price:,.2f}"
                )
                return price
            else:
                error_msg = f"{source_name} returned invalid price: ${price:,.2f}"
                logger.warning(error_msg)
                errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"{source_name} request failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
        except (ValueError, KeyError) as e:
            error_msg = f"{source_name} invalid response: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"{source_name} unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

    # All sources failed
    logger.error(f"All BTC price sources failed. Errors: {'; '.join(errors)}")
    return None


if __name__ == "__main__":
    # Configure logging for direct execution
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Checking Bitcoin price sources...")
    print("-" * 40)

    test_sources = [
        (fetch_price_coingecko, "CoinGecko"),
        (fetch_price_coinbase, "Coinbase"),
        (fetch_price_bitstamp, "Bitstamp"),
        (fetch_price_kraken, "Kraken"),
        (fetch_price_binance, "Binance"),
    ]

    success_count = 0
    for fetch_func, name in test_sources:
        try:
            price = fetch_func()
            print(f"✓ {name:<12}: ${price:,.2f}")
            success_count += 1
        except Exception as e:
            print(f"✗ {name:<12}: Error - {e}")

    print("-" * 40)
    robust_price = fetch_btc_price_robust()
    if robust_price:
        print(f"FINAL RESULT  : ${robust_price:,.2f} ({success_count}/{len(test_sources)} sources up)")
    else:
        print("FINAL RESULT  : FAILED (All sources down)")
