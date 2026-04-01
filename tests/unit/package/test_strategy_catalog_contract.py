from __future__ import annotations

from types import SimpleNamespace

import pytest

from importlib import import_module

from stacksats.loader import load_strategy
from stacksats.strategies import catalog
from stacksats.strategies.catalog import StrategyCatalogEntry, list_strategies
from stacksats.strategy_types import BacktestConfig, BaseStrategy, ValidationConfig


def test_catalog_entries_match_runtime_strategy_identity() -> None:
    for entry in list_strategies(public_only=False):
        strategy = load_strategy(entry.strategy_id)
        strategy_from_spec = load_strategy(entry.strategy_spec)
        strategy_cls = type(strategy)
        module = import_module(entry.module_path)

        assert strategy.metadata().strategy_id == entry.strategy_id
        assert strategy_cls.__name__ == entry.class_name
        assert strategy_cls.__module__ == entry.module_path
        assert getattr(module, entry.class_name) is strategy_cls
        assert type(strategy_from_spec) is strategy_cls
        assert strategy.default_validation_config().start_date == entry.default_validation_config.get(
            "start_date"
        )
        assert strategy.default_validation_config().end_date == entry.default_validation_config.get(
            "end_date"
        )
        assert (
            strategy.default_validation_config().min_win_rate
            == entry.default_validation_config["min_win_rate"]
        )
        assert (
            strategy.default_validation_config().strict
            == entry.default_validation_config["strict"]
        )
        assert strategy.default_backtest_config().start_date == entry.default_backtest_config.get(
            "start_date"
        )
        assert strategy.default_backtest_config().end_date == entry.default_backtest_config.get(
            "end_date"
        )
        assert strategy.default_backtest_config().strategy_label == entry.strategy_id
        assert entry.owner
        assert entry.promotion_stage in {"research", "candidate", "promoted"}


def test_catalog_tier_filter_uses_metadata_not_module_path(monkeypatch) -> None:
    patched_entry = StrategyCatalogEntry(
        strategy_id="shadow-stable",
        strategy_spec="stacksats.strategies.experimental.shadow:ShadowStableStrategy",
        class_name="ShadowStableStrategy",
        module_path="stacksats.strategies.experimental.shadow",
        tier="stable",
        public_export=True,
        audit_enabled=False,
        family="shadow",
        description="Metadata-only tier contract check.",
        docs_slug="shadow-stable",
        tags=("test",),
        owner="StackSats Maintainers",
        benchmark_strategy_ids=("uniform",),
        promotion_stage="promoted",
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={"start_date": "2018-01-01", "end_date": "2025-12-31"},
    )
    monkeypatch.setattr(catalog, "_CATALOG", (patched_entry,))
    monkeypatch.setattr(catalog, "_CATALOG_BY_ID", {patched_entry.strategy_id: patched_entry})

    assert catalog.list_strategies(tier="stable", public_only=False) == (patched_entry,)
    assert catalog.list_strategies(tier="experimental", public_only=False) == ()
    assert catalog.list_strategies(tier="stable") == (patched_entry,)


def test_custom_strategies_keep_generic_runtime_defaults() -> None:
    class CustomStrategy(BaseStrategy):
        strategy_id = "custom-local"

    strategy = CustomStrategy()

    assert strategy.default_validation_config() == ValidationConfig()
    assert strategy.default_backtest_config() == BacktestConfig(strategy_label="custom-local")


def test_catalog_benchmark_ids_reference_known_entries() -> None:
    all_ids = {entry.strategy_id for entry in list_strategies(public_only=False)}
    for entry in list_strategies(public_only=False):
        for benchmark_id in entry.benchmark_strategy_ids:
            assert benchmark_id in all_ids


def test_load_catalog_strategy_class_rejects_non_strategy_exports(monkeypatch) -> None:
    monkeypatch.setattr(
        catalog,
        "get_strategy_catalog_entry",
        lambda strategy_id: StrategyCatalogEntry(
            strategy_id=strategy_id,
            strategy_spec="fake.module:NotAStrategy",
            class_name="NotAStrategy",
            module_path="fake.module",
            tier="experimental",
            public_export=False,
            audit_enabled=False,
            family="test",
            description="Invalid strategy export for contract coverage.",
            docs_slug="fake",
            tags=("test",),
            owner="StackSats Maintainers",
            benchmark_strategy_ids=(),
            promotion_stage="research",
            default_validation_config={"min_win_rate": 50.0, "strict": True},
            default_backtest_config={"start_date": "2018-01-01", "end_date": "2025-12-31"},
        ),
    )
    monkeypatch.setattr(
        catalog.importlib,
        "import_module",
        lambda _module_path: SimpleNamespace(NotAStrategy=object()),
    )

    with pytest.raises(TypeError, match="does not resolve to a BaseStrategy subclass"):
        catalog.load_catalog_strategy_class("broken")
