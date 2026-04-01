from __future__ import annotations

from importlib import import_module

from stacksats.loader import load_strategy
from stacksats.strategies import catalog
from stacksats.strategies.catalog import StrategyCatalogEntry, list_strategies


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
        default_validation_config={"min_win_rate": 50.0, "strict": True},
        default_backtest_config={"start_date": "2018-01-01", "end_date": "2025-12-31"},
    )
    monkeypatch.setattr(catalog, "_CATALOG", (patched_entry,))
    monkeypatch.setattr(catalog, "_CATALOG_BY_ID", {patched_entry.strategy_id: patched_entry})

    assert catalog.list_strategies(tier="stable", public_only=False) == (patched_entry,)
    assert catalog.list_strategies(tier="experimental", public_only=False) == ()
    assert catalog.list_strategies(tier="stable") == (patched_entry,)
