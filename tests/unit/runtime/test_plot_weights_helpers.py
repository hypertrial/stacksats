from __future__ import annotations

import builtins
import importlib
import runpy
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest
from matplotlib.axes import Axes

from stacksats.plot_weights import (
    fetch_weights_for_date_range,
    get_oldest_date_range,
    get_date_range_options,
    get_db_connection,
    plot_dca_weights,
    validate_date_range,
)


def test_get_db_connection_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL environment variable is not set"):
        get_db_connection()


def test_get_db_connection_uses_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_mock = MagicMock(return_value=object())
    monkeypatch.setenv("DATABASE_URL", "postgres://example")
    monkeypatch.setattr("stacksats.plot_weights.psycopg2.connect", connect_mock)

    conn = get_db_connection()

    assert conn is connect_mock.return_value
    connect_mock.assert_called_once_with("postgres://example")


def test_get_date_range_options_returns_typed_dataframe() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("2024-01-01", "2024-12-31", 366),
        ("2024-02-01", "2025-01-31", 366),
    ]

    df = get_date_range_options(conn)

    assert list(df.columns) == ["start_date", "end_date", "count"]
    assert pd.api.types.is_datetime64_any_dtype(df["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["end_date"])
    assert df.iloc[0]["count"] == 366


def test_get_date_range_options_raises_when_no_rows() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []

    with pytest.raises(ValueError, match="No data found"):
        get_date_range_options(conn)


def test_get_oldest_date_range_returns_first_option() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("2024-01-01", "2024-12-31", 366),
        ("2024-02-01", "2025-01-31", 366),
    ]

    start_date, end_date = get_oldest_date_range(conn)

    assert start_date == "2024-01-01"
    assert end_date == "2024-12-31"


def test_validate_date_range_returns_true_when_present() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (1,)

    assert validate_date_range(conn, "2024-01-01", "2024-12-31") is True


def test_validate_date_range_returns_false_when_absent() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (0,)

    assert validate_date_range(conn, "2024-01-01", "2024-12-31") is False


def test_fetch_weights_for_date_range_returns_dataframe() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("2024-01-01", 0.5, 50000.0, 1),
        ("2024-01-02", 0.5, None, 2),
    ]

    df = fetch_weights_for_date_range(conn, "2024-01-01", "2024-12-31")

    assert list(df.columns) == ["DCA_date", "weight", "btc_usd", "id"]
    assert pd.api.types.is_datetime64_any_dtype(df["DCA_date"])
    assert len(df) == 2


def test_fetch_weights_for_date_range_raises_when_no_rows() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []

    with pytest.raises(ValueError, match="No data found for date range"):
        fetch_weights_for_date_range(conn, "2024-01-01", "2024-12-31")


def test_plot_dca_weights_draws_boundary_for_mixed_past_and_future(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    boundary_calls: list[pd.Timestamp] = []
    original_axvline = Axes.axvline

    def _spy_axvline(self, *args, **kwargs):
        x = kwargs.get("x", args[0] if args else None)
        boundary_calls.append(pd.Timestamp(x))
        return original_axvline(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", _spy_axvline)
    monkeypatch.setattr("stacksats.plot_weights.plt.savefig", lambda *_args, **_kwargs: None)

    df = pd.DataFrame(
        {
            "DCA_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "weight": [0.2, 0.3, 0.25, 0.25],
            "btc_usd": [50000.0, 51000.0, None, None],
            "id": [1, 2, 3, 4],
        }
    )

    plot_dca_weights(
        df,
        start_date="2024-01-01",
        end_date="2024-01-04",
        output_path=str(tmp_path / "weights.svg"),
    )

    assert len(boundary_calls) == 1
    assert boundary_calls[0] == pd.Timestamp("2024-01-02")


def test_plot_weights_import_falls_back_when_dotenv_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stacksats.plot_weights as plot_weights_module

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dotenv":
            raise ImportError("dotenv unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    reloaded = importlib.reload(plot_weights_module)

    assert reloaded is plot_weights_module


def test_plot_weights_module_dunder_main_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("2024-01-01", "2024-12-31", 366)]

    monkeypatch.setenv("DATABASE_URL", "postgres://example")
    monkeypatch.setattr("psycopg2.connect", lambda _url: conn)
    monkeypatch.setattr(sys, "argv", ["plot_weights.py", "--list"])

    runpy.run_module("stacksats.plot_weights", run_name="__main__")

    assert conn.close.called
