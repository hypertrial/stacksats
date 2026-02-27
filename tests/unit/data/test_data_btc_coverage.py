from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from stacksats.data_btc import BTCDataProvider, DataLoadError, _is_cache_usable


def _csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


class _Resp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_is_cache_usable_handles_internal_exception(monkeypatch) -> None:
    monkeypatch.setattr("stacksats.data_btc.pd.read_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _is_cache_usable(b"x", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")) is False


def test_load_validates_end_date_and_order() -> None:
    provider = BTCDataProvider(cache_dir=None, clock=lambda: pd.Timestamp("2024-01-10"))
    with pytest.raises(ValueError, match="Invalid end_date"):
        provider.load(backtest_start="2024-01-01", end_date="NaT")
    with pytest.raises(ValueError, match="end_date must be on or after backtest_start"):
        provider.load(backtest_start="2024-01-10", end_date="2024-01-01")


def test_download_failure_messages_include_cache_context(tmp_path: Path, monkeypatch) -> None:
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_text("bad", encoding="utf-8")
    monkeypatch.setattr(
        "stacksats.data_btc.requests.get",
        lambda *a, **k: (_ for _ in ()).throw(requests.RequestException("offline")),
    )
    provider = BTCDataProvider(cache_dir=str(tmp_path), clock=lambda: pd.Timestamp("2024-01-10"))
    with pytest.raises(DataLoadError, match="local cache file exists"):
        provider.load(backtest_start="2024-01-01")

    monkeypatch.setattr(
        "stacksats.data_btc.requests.get",
        lambda *a, **k: (_ for _ in ()).throw(requests.RequestException("offline")),
    )
    provider_no_cache = BTCDataProvider(cache_dir=str(tmp_path / "no_cache"), clock=lambda: pd.Timestamp("2024-01-10"))
    with pytest.raises(DataLoadError, match="No local cache file was available"):
        provider_no_cache.load(backtest_start="2024-01-01")


def test_cache_read_and_write_oserror_paths_are_nonfatal(tmp_path: Path, monkeypatch) -> None:
    now = pd.Timestamp("2024-01-10")
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_bytes(_csv_bytes([{"time": "2024-01-10", "PriceUSD": 50000.0}]))
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(OSError("read blocked")),
    )
    monkeypatch.setattr(
        "stacksats.data_btc.requests.get",
        lambda *a, **k: _Resp(_csv_bytes([{"time": "2024-01-10", "PriceUSD": 50000.0}])),
    )
    provider = BTCDataProvider(cache_dir=str(tmp_path), clock=lambda: now)
    df = provider.load(backtest_start="2024-01-10")
    assert float(df.iloc[0]["PriceUSD_coinmetrics"]) == 50000.0

    monkeypatch.setattr(
        Path,
        "write_bytes",
        lambda self, b: (_ for _ in ()).throw(OSError("write blocked")),
    )
    df2 = provider.load(backtest_start="2024-01-10")
    assert float(df2.iloc[0]["PriceUSD_coinmetrics"]) == 50000.0


def test_past_only_error_paths_without_usable_cache(tmp_path: Path) -> None:
    provider = BTCDataProvider(cache_dir=str(tmp_path), clock=lambda: pd.Timestamp("2024-01-10"))
    with pytest.raises(DataLoadError, match="no local CoinMetrics cache file"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")

    bad_cache = tmp_path / "coinmetrics_btc.csv"
    bad_cache.write_text("time,PriceUSD\n2024-01-01,\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="does not cover the requested window"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")


def test_past_only_with_cache_disabled_raises() -> None:
    provider = BTCDataProvider(cache_dir=None, clock=lambda: pd.Timestamp("2024-01-10"))
    with pytest.raises(DataLoadError, match="cache disabled"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")


def test_parse_failures_and_dataframe_validation_failures(monkeypatch) -> None:
    now = pd.Timestamp("2024-01-10")
    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    monkeypatch.setattr(
        "stacksats.data_btc.requests.get",
        lambda *a, **k: _Resp(b"raw"),
    )
    monkeypatch.setattr(
        "stacksats.data_btc.parse_coinmetrics_btc_csv_bytes",
        lambda *_: (_ for _ in ()).throw(ValueError("bad parse")),
    )
    with pytest.raises(DataLoadError, match="could not be parsed"):
        provider.load(backtest_start="2024-01-01")

    monkeypatch.setattr(
        "stacksats.data_btc.parse_coinmetrics_btc_csv_bytes",
        lambda *_: pd.DataFrame(index=pd.DatetimeIndex([]), columns=["PriceUSD"]),
    )
    with pytest.raises(DataLoadError, match="contained no rows"):
        provider.load(backtest_start="2024-01-01")

    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    monkeypatch.setattr(
        "stacksats.data_btc.parse_coinmetrics_btc_csv_bytes",
        lambda *_: pd.DataFrame({"PriceUSD": [None, None]}, index=idx),
    )
    with pytest.raises(DataLoadError, match="no valid PriceUSD"):
        provider.load(backtest_start="2024-01-01")

    monkeypatch.setattr(
        "stacksats.data_btc.parse_coinmetrics_btc_csv_bytes",
        lambda *_: pd.DataFrame({"PriceUSD": [100.0]}, index=pd.DatetimeIndex(["2023-01-01"])),
    )
    with pytest.raises(DataLoadError, match="does not cover the requested start date"):
        provider.load(backtest_start="2024-01-01")


def test_missing_price_and_missing_date_paths(monkeypatch) -> None:
    provider = BTCDataProvider(cache_dir=None, clock=lambda: pd.Timestamp("2024-01-03"))
    monkeypatch.setattr("stacksats.data_btc.requests.get", lambda *a, **k: _Resp(b"ok"))

    idx_missing_price = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    monkeypatch.setattr(
        "stacksats.data_btc.parse_coinmetrics_btc_csv_bytes",
        lambda *_: pd.DataFrame({"PriceUSD": [100.0, None, 120.0]}, index=idx_missing_price),
    )
    with pytest.raises(DataLoadError, match="missing PriceUSD values"):
        provider.load(backtest_start="2024-01-01")

    idx_missing_date = pd.DatetimeIndex(["2024-01-01", "2024-01-03"])
    monkeypatch.setattr(
        "stacksats.data_btc.parse_coinmetrics_btc_csv_bytes",
        lambda *_: pd.DataFrame({"PriceUSD": [100.0, 120.0]}, index=idx_missing_date),
    )
    with pytest.raises(DataLoadError, match="missing dates"):
        provider.load(backtest_start="2024-01-01")


def test_invalid_end_date_parse_error_raises_value_error() -> None:
    provider = BTCDataProvider(cache_dir=None, clock=lambda: pd.Timestamp("2024-01-10"))
    with pytest.raises(ValueError, match="Invalid end_date value"):
        provider.load(backtest_start="2024-01-01", end_date="not-a-date")


def test_cache_missing_path_write_oserror_is_nonfatal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "stacksats.data_btc.requests.get",
        lambda *a, **k: _Resp(_csv_bytes([{"time": "2024-01-10", "PriceUSD": 50000.0}])),
    )
    monkeypatch.setattr(
        Path,
        "write_bytes",
        lambda self, b: (_ for _ in ()).throw(OSError("write blocked")),
    )
    provider = BTCDataProvider(cache_dir=str(tmp_path), clock=lambda: pd.Timestamp("2024-01-10"))
    df = provider.load(backtest_start="2024-01-10")
    assert float(df.iloc[0]["PriceUSD_coinmetrics"]) == 50000.0


def test_window_empty_branch_raises_data_load_error(monkeypatch) -> None:
    provider = BTCDataProvider(cache_dir=None, clock=lambda: pd.Timestamp("2024-01-03"))
    monkeypatch.setattr("stacksats.data_btc.requests.get", lambda *a, **k: _Resp(b"ok"))
    descending = pd.DataFrame(
        {"PriceUSD": [30000.0, 20000.0, 10000.0]},
        index=pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-01"]),
    )
    monkeypatch.setattr("stacksats.data_btc.parse_coinmetrics_btc_csv_bytes", lambda *_: descending)
    with pytest.raises(DataLoadError, match="No CoinMetrics rows available"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-03")
