from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "script_name,url,payload,expected_output",
    [
        (
            "bitstamp_btc_usd.py",
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            {"last": "90000.0"},
            "90000.0",
        ),
        (
            "coinbase_btc_usd.py",
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            {"data": {"amount": "91000.0"}},
            "91000.0",
        ),
        (
            "coingecko_btc_usd.py",
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            {"bitcoin": {"usd": 92000.0}},
            92000.0,
        ),
        (
            "kraken_xbt_usd.py",
            "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
            {"result": {"XXBTZUSD": {"c": ["93000.0", "0.1"]}}},
            "93000.0",
        ),
    ],
)
def test_btc_api_script_smoke_execution(
    script_name: str,
    url: str,
    payload: dict,
    expected_output,
) -> None:
    script_path = REPO_ROOT / "stacksats" / "btc_api" / script_name

    response = MagicMock()
    response.json.return_value = payload

    with patch("requests.get", return_value=response) as mock_get, patch(
        "builtins.print"
    ) as mock_print:
        runpy.run_path(str(script_path), run_name="__main__")

    mock_get.assert_called_once_with(url, verify=False)
    mock_print.assert_called_once_with(expected_output)
