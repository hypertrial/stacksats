from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import polars as pl
import pytest

import stacksats._optional as optional
import stacksats.btc_price_fetcher as btc_price_fetcher
import stacksats.plot_mvrv as plot_mvrv
import stacksats.plot_weights as plot_weights


def test_plot_mvrv_requires_viz_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        plot_mvrv,
        "_VIZ_IMPORT_ERROR",
        optional.missing_dependency_error(
            dependency="matplotlib/seaborn",
            extra="viz",
            feature="plotting commands",
        ),
    )
    with pytest.raises(ImportError, match=r"stacksats\[viz\]"):
        plot_mvrv.plot_mvrv_metrics(pl.DataFrame({"date": [], "mvrv": []}))


def test_plot_weights_requires_viz_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        plot_weights,
        "_VIZ_IMPORT_ERROR",
        optional.missing_dependency_error(
            dependency="matplotlib/seaborn",
            extra="viz",
            feature="plotting commands",
        ),
    )
    with pytest.raises(ImportError, match=r"stacksats\[viz\]"):
        plot_weights.plot_dca_weights(
            pl.DataFrame({"date": [], "weight": []}),
            "2024-01-01",
            "2024-12-31",
        )


def test_btc_price_fetcher_requires_network_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_optional = optional.import_optional

    def _fake_import_optional(module_name: str, *, extra: str, feature: str):
        if module_name == "requests":
            raise optional.missing_dependency_error(
                dependency="requests",
                extra="network",
                feature="BTC price fetching helpers",
            )
        return original_import_optional(module_name, extra=extra, feature=feature)

    monkeypatch.setattr(optional, "import_optional", _fake_import_optional)
    with pytest.raises(ImportError, match=r"stacksats\[network\]"):
        importlib.reload(btc_price_fetcher)

    monkeypatch.undo()
    importlib.reload(btc_price_fetcher)


def test_plot_mvrv_help_does_not_require_viz_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plot_mvrv,
        "_VIZ_IMPORT_ERROR",
        optional.missing_dependency_error(
            dependency="matplotlib/seaborn",
            extra="viz",
            feature="plotting commands",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["stacksats-plot-mvrv", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        plot_mvrv.main()

    assert excinfo.value.code == 0


def test_plot_weights_list_does_not_require_viz_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_conn = MagicMock()
    monkeypatch.setattr(
        plot_weights,
        "_VIZ_IMPORT_ERROR",
        optional.missing_dependency_error(
            dependency="matplotlib/seaborn",
            extra="viz",
            feature="plotting commands",
        ),
    )
    monkeypatch.setattr("stacksats.plot_weights.get_db_connection", lambda: mock_conn)
    monkeypatch.setattr(
        "stacksats.plot_weights.get_date_range_options",
        lambda _conn: pl.DataFrame(
            {
                "start_date": [pl.datetime(2024, 1, 1, 0, 0, 0, time_unit="us")],
                "end_date": [pl.datetime(2024, 12, 31, 0, 0, 0, time_unit="us")],
                "count": [366],
            }
        ),
    )
    monkeypatch.setattr(sys, "argv", ["stacksats-plot-weights", "--list"])

    plot_weights.main()

    assert mock_conn.close.called
