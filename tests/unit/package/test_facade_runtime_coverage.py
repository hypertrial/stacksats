from __future__ import annotations

import builtins
import importlib
import sys

import pytest

import stacksats.data as data_pkg
import stacksats.export_weights as export_weights_pkg
import stacksats.export_weights.public as export_weights_public
import stacksats.runner.provenance as runner_provenance
import stacksats.service as service_entrypoints
from stacksats import BacktestConfig
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import AgentServiceConfig


def test_data_package_lazy_exports_cover_getattr_dir_and_error() -> None:
    module = importlib.reload(data_pkg)
    for name in module.__all__:
        module.__dict__.pop(name, None)

    assert "data_setup" in dir(module)

    loaded = module.data_setup
    assert loaded is importlib.import_module("stacksats.data.data_setup")

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(module, "missing_helper")


def test_export_weights_public_facade_reexports_package_api() -> None:
    module = importlib.reload(export_weights_public)

    assert module.get_db_connection is export_weights_pkg.get_db_connection
    assert module.process_start_date_batch is export_weights_pkg.process_start_date_batch


def test_runner_provenance_config_hash_matches_runner_helper() -> None:
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        strategy_label="coverage-provenance",
    )

    assert runner_provenance.config_hash(config) == StrategyRunner._config_hash(config)


def test_service_entrypoint_wraps_missing_pydantic_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "app"
            and level == 1
            and globals
            and globals.get("__package__") == "stacksats.service"
        ):
            exc = ModuleNotFoundError("No module named 'pydantic'")
            exc.name = "pydantic"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("stacksats.service.app", None)
    module = importlib.reload(service_entrypoints)

    with pytest.raises(ImportError, match=r"stacksats\[service\]"):
        module.create_agent_service_app(AgentServiceConfig())
