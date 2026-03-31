from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import polars as pl
import pytest

import stacksats._optional as optional
import stacksats.data.btc_price_fetcher as btc_price_fetcher
import stacksats.viz.plot_mvrv as plot_mvrv
import stacksats.viz.plot_weights as plot_weights
from stacksats.service import create_agent_service_app
from stacksats.strategy_types import AgentServiceConfig


def test_import_optional_raises_helpful_error_for_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_module_name: str):
        exc = ModuleNotFoundError("No module named 'requests'")
        exc.name = "requests"
        raise exc

    monkeypatch.setattr(optional.importlib, "import_module", _missing)

    with pytest.raises(ImportError, match=r"stacksats\[network\]"):
        optional.import_optional(
            "requests",
            extra="network",
            feature="BTC price fetching helpers",
        )


def test_import_optional_reraises_nested_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _nested_missing(_module_name: str):
        exc = ModuleNotFoundError("No module named 'charset_normalizer'")
        exc.name = "charset_normalizer"
        raise exc

    monkeypatch.setattr(optional.importlib, "import_module", _nested_missing)

    with pytest.raises(ModuleNotFoundError, match="charset_normalizer"):
        optional.import_optional(
            "requests",
            extra="network",
            feature="BTC price fetching helpers",
        )


def test_missing_dependency_error_builds_consistent_message() -> None:
    error = optional.missing_dependency_error(
        dependency="requests",
        extra="network",
        feature="BTC price fetching helpers",
    )

    assert isinstance(error, ImportError)
    assert "requests" in str(error)
    assert 'stacksats[network]' in str(error)
    assert "BTC price fetching helpers" in str(error)


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


def test_agent_service_requires_service_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi" or name.startswith("fastapi."):
            exc = ModuleNotFoundError("No module named 'fastapi'")
            exc.name = "fastapi"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("stacksats.service.app", None)

    with pytest.raises(ImportError, match=r"stacksats\[service\]"):
        create_agent_service_app(AgentServiceConfig())


def test_agent_service_reraises_nested_fastapi_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi" or name.startswith("fastapi."):
            exc = ModuleNotFoundError("No module named 'starlette'")
            exc.name = "starlette"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("stacksats.service.app", None)

    with pytest.raises(ModuleNotFoundError, match="starlette"):
        create_agent_service_app(AgentServiceConfig())


def test_agent_service_requires_pydantic_from_service_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydantic" or name.startswith("pydantic."):
            exc = ModuleNotFoundError("No module named 'pydantic'")
            exc.name = "pydantic"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("stacksats.service.app", None)

    with pytest.raises(ImportError, match=r"stacksats\[service\]"):
        create_agent_service_app(AgentServiceConfig())


def test_agent_service_reraises_nested_pydantic_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydantic" or name.startswith("pydantic."):
            exc = ModuleNotFoundError("No module named 'pydantic_core'")
            exc.name = "pydantic_core"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("stacksats.service.app", None)

    with pytest.raises(ModuleNotFoundError, match="pydantic_core"):
        create_agent_service_app(AgentServiceConfig())


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
    monkeypatch.setattr("stacksats.viz.plot_weights.get_db_connection", lambda: mock_conn)
    monkeypatch.setattr(
        "stacksats.viz.plot_weights.get_date_range_options",
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


def test_plot_mvrv_import_fallback_creates_placeholder_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"matplotlib.dates", "matplotlib.pyplot", "seaborn"}:
            raise ImportError(f"{name} unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    reloaded = importlib.reload(plot_mvrv)

    assert reloaded is plot_mvrv
    assert isinstance(plot_mvrv.mdates, ModuleType)
    assert isinstance(plot_mvrv.plt, ModuleType)
    assert isinstance(plot_mvrv.sns, ModuleType)
    assert plot_mvrv.plt.rcParams == {}
    with pytest.raises(ImportError, match=r"stacksats\[viz\]"):
        plot_mvrv._ensure_viz_available()

    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(plot_mvrv)


def test_plot_weights_import_fallback_creates_placeholder_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"matplotlib.dates", "matplotlib.pyplot", "seaborn"}:
            raise ImportError(f"{name} unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    reloaded = importlib.reload(plot_weights)

    assert reloaded is plot_weights
    assert isinstance(plot_weights.mdates, ModuleType)
    assert isinstance(plot_weights.plt, ModuleType)
    assert isinstance(plot_weights.sns, ModuleType)
    assert plot_weights.plt.rcParams == {}
    with pytest.raises(ImportError, match=r"stacksats\[viz\]"):
        plot_weights._ensure_viz_available()

    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(plot_weights)
