from __future__ import annotations

import importlib

import pytest


def test_legacy_csv_loader_module_is_removed() -> None:
    module_name = "stacksats.btc_api." + "coin" + "metrics_btc_csv"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
