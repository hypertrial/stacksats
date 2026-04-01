#!/usr/bin/env python3
"""Scaffold a new built-in strategy module, test, and catalog stub."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _snake_name(value: str) -> str:
    return value.replace("-", "_")


def module_path_for(*, root: Path, tier: str, family: str, strategy_id: str) -> Path:
    return root / "stacksats" / "strategies" / tier / family / f"{_snake_name(strategy_id)}.py"


def test_path_for(*, root: Path, strategy_id: str) -> Path:
    return root / "tests" / "unit" / "strategies" / f"test_{_snake_name(strategy_id)}_strategy.py"


def catalog_entry_stub(
    *,
    tier: str,
    family: str,
    strategy_id: str,
    class_name: str,
    intent: str,
) -> str:
    module_path = f"stacksats.strategies.{tier}.{family}.{_snake_name(strategy_id)}"
    return f"""    StrategyCatalogEntry(
        strategy_id="{strategy_id}",
        strategy_spec="{module_path}:{class_name}",
        class_name="{class_name}",
        module_path="{module_path}",
        tier="{tier}",
        public_export={str(tier == "stable")},
        audit_enabled=True,
        family="{family}",
        description="TODO: describe the strategy.",
        docs_slug="{strategy_id}",
        tags=("{tier}", "{family}"),
        default_validation_config={{"min_win_rate": 50.0, "strict": True}},
        default_backtest_config={{
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
        }},
    ),  # TODO: implement {intent}-mode strategy details
"""


def module_source(*, class_name: str, strategy_id: str, intent: str) -> str:
    if intent == "propose":
        hook = """
    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def propose_weight(self, state: DayState) -> float:
        return float(state.uniform_weight)
"""
        imports = "from ....strategy_types import BaseStrategy, DayState\n"
    else:
        hook = """
    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del ctx, signals
        return pl.DataFrame({
            "date": features_df["date"],
            "value": pl.Series([0.0] * features_df.height, dtype=pl.Float64),
        })
"""
        imports = (
            "import polars as pl\n\n"
            "from ....strategy_types import BaseStrategy, StrategyContext\n"
        )
    return (
        '"""Scaffolded strategy."""\n\n'
        "from __future__ import annotations\n\n"
        f"{imports}\n\n"
        f"class {class_name}(BaseStrategy):\n"
        '    """TODO: describe the strategy."""\n\n'
        f'    strategy_id = "{strategy_id}"\n'
        '    version = "0.1.0"\n'
        '    description = "TODO: describe the strategy."\n'
        f"{hook}\n"
        f'\n__all__ = ["{class_name}"]\n'
    )


def test_source(*, module_import: str, class_name: str, strategy_id: str) -> str:
    return f"""from __future__ import annotations

from {module_import} import {class_name}


def test_{_snake_name(strategy_id)}_strategy_metadata() -> None:
    strategy = {class_name}()
    assert strategy.strategy_id == "{strategy_id}"
"""


def insert_catalog_stub(catalog_path: Path, stub: str, *, tier: str) -> None:
    text = catalog_path.read_text(encoding="utf-8")
    marker = '        tier="experimental",'
    if tier == "stable" and marker in text:
        index = text.index(marker)
        entry_start = text.rfind("    StrategyCatalogEntry(", 0, index)
        updated = text[:entry_start] + stub + text[entry_start:]
    else:
        updated = text.replace("\n)\n\n_CATALOG_BY_ID", f"\n{stub})\n\n_CATALOG_BY_ID", 1)
    catalog_path.write_text(updated, encoding="utf-8")


def scaffold_strategy(
    *,
    root: Path,
    tier: str,
    family: str,
    strategy_id: str,
    class_name: str,
    intent: str,
) -> tuple[Path, Path]:
    module_path = module_path_for(root=root, tier=tier, family=family, strategy_id=strategy_id)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        module_source(class_name=class_name, strategy_id=strategy_id, intent=intent),
        encoding="utf-8",
    )

    test_path = test_path_for(root=root, strategy_id=strategy_id)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        test_source(
            module_import=(
                f"stacksats.strategies.{tier}.{family}.{_snake_name(strategy_id)}"
            ),
            class_name=class_name,
            strategy_id=strategy_id,
        ),
        encoding="utf-8",
    )

    insert_catalog_stub(
        root / "stacksats" / "strategies" / "catalog.py",
        catalog_entry_stub(
            tier=tier,
            family=family,
            strategy_id=strategy_id,
            class_name=class_name,
            intent=intent,
        ),
        tier=tier,
    )
    return module_path, test_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a built-in strategy.")
    parser.add_argument("--tier", choices=("stable", "experimental"), required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--intent", choices=("propose", "profile"), required=True)
    args = parser.parse_args()

    module_path, test_path = scaffold_strategy(
        root=ROOT,
        tier=args.tier,
        family=args.family,
        strategy_id=args.strategy_id,
        class_name=args.class_name,
        intent=args.intent,
    )
    print(f"Created {module_path.relative_to(ROOT)}")
    print(f"Created {test_path.relative_to(ROOT)}")
    print("Updated stacksats/strategies/catalog.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
