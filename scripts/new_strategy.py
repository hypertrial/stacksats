#!/usr/bin/env python3
"""Scaffold a new built-in strategy module, test, model card, and catalog stub."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
_VALID_TIERS = frozenset({"stable", "experimental"})
_STRATEGY_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_CLASS_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _snake_name(value: str) -> str:
    return value.replace("-", "_")


def _validate_inputs(*, tier: str, family: str, strategy_id: str, class_name: str) -> None:
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Unsupported tier '{tier}'. Expected one of: {', '.join(sorted(_VALID_TIERS))}."
        )
    if not family or not _FAMILY_RE.fullmatch(family):
        raise ValueError(
            "family must be a non-empty snake_case identifier using only lowercase "
            "letters, digits, and underscores."
        )
    if not _STRATEGY_ID_RE.fullmatch(strategy_id):
        raise ValueError(
            "strategy_id must match ^[a-z][a-z0-9-]*$ and use lowercase kebab-case."
        )
    if not _CLASS_NAME_RE.fullmatch(class_name):
        raise ValueError(
            "class_name must be a valid PascalCase Python class name."
        )


def module_path_for(*, root: Path, tier: str, family: str, strategy_id: str) -> Path:
    return root / "stacksats" / "strategies" / tier / family / f"{_snake_name(strategy_id)}.py"


def test_path_for(*, root: Path, strategy_id: str) -> Path:
    return root / "tests" / "unit" / "strategies" / f"test_{_snake_name(strategy_id)}_strategy.py"


def model_card_path_for(*, root: Path, strategy_id: str) -> Path:
    return root / "docs" / "reference" / "models" / f"{strategy_id}.md"


def _ensure_package_chain(path: Path, *, stop_at: Path) -> None:
    current = path
    while True:
        current.mkdir(parents=True, exist_ok=True)
        init_path = current / "__init__.py"
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")
        if current == stop_at:
            return
        if stop_at not in current.parents:
            raise ValueError(f"{stop_at} is not an ancestor of {path}")
        current = current.parent


def catalog_entry_stub(
    *,
    tier: str,
    family: str,
    strategy_id: str,
    class_name: str,
    intent: str,
) -> str:
    module_path = f"stacksats.strategies.{tier}.{family}.{_snake_name(strategy_id)}"
    promotion_stage = "promoted" if tier == "stable" else "research"
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
        owner="TODO: set owner",
        benchmark_strategy_ids=("uniform",),
        promotion_stage="{promotion_stage}",
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
    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def propose_weight(self, state: DayState) -> float:
        # Replace this placeholder with a causal daily proposal rule.
        return float(state.uniform_weight)
"""
        imports = (
            "import polars as pl\n\n"
            "from ....strategy_types import BaseStrategy, DayState, StrategyContext\n"
        )
    else:
        hook = """
    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
    ) -> dict[str, pl.Series]:
        del ctx
        return {
            "placeholder": pl.Series(
                "placeholder",
                [0.0] * features_df.height,
                dtype=pl.Float64,
            ),
        }

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx
        # Replace this placeholder with a causal preference or absolute profile.
        return TargetProfile(
            values=pl.DataFrame({
                "date": features_df["date"],
                "value": signals["placeholder"],
            }),
            mode="preference",
        )
"""
        imports = (
            "import polars as pl\n\n"
            "from ....strategy_types import BaseStrategy, StrategyContext, TargetProfile\n"
        )
    placeholder_comment = (
        "    # Replace placeholder hooks and metadata before promoting this strategy.\n"
    )
    return (
        '"""Scaffolded strategy."""\n\n'
        "from __future__ import annotations\n\n"
        f"{imports}\n\n"
        f"class {class_name}(BaseStrategy):\n"
        '    """TODO: describe the strategy."""\n\n'
        f"{placeholder_comment}"
        f'    strategy_id = "{strategy_id}"\n'
        '    version = "0.1.0"\n'
        '    description = "TODO: describe the strategy."\n'
        f"{hook}\n"
        f'\n__all__ = ["{class_name}"]\n'
    )


def test_source(*, module_import: str, class_name: str, strategy_id: str, intent: str) -> str:
    return f"""from __future__ import annotations

from stacksats.loader import load_strategy
from {module_import} import {class_name}


def test_{_snake_name(strategy_id)}_strategy_import_smoke() -> None:
    assert {class_name}.__name__ == "{class_name}"


def test_{_snake_name(strategy_id)}_strategy_metadata_smoke() -> None:
    strategy = {class_name}()
    assert strategy.strategy_id == "{strategy_id}"
    assert strategy.version


def test_{_snake_name(strategy_id)}_strategy_intent_smoke() -> None:
    strategy = {class_name}()
    assert strategy.intent_mode() == "{intent}"


def test_{_snake_name(strategy_id)}_strategy_loader_smoke() -> None:
    strategy = load_strategy("{strategy_id}")
    assert isinstance(strategy, {class_name})
"""


def model_card_source(
    *,
    tier: str,
    family: str,
    strategy_id: str,
    class_name: str,
    intent: str,
) -> str:
    return f"""---
title: {class_name}
description: Model card for the built-in `{strategy_id}` strategy.
---

# {class_name}

## Summary

- `strategy_id`: `{strategy_id}`
- intent mode: `{intent}`
- support tier: `{tier}`
- promotion stage: `{"promoted" if tier == "stable" else "research"}`
- owner: `TODO`

## Why this model exists

TODO: explain the hypothesis or operational reason for keeping this model in the library.

## Feature dependencies

- family: `{family}`
- required feature sets: `core_model_features_v1`
- required transformed columns: `TODO`

## Benchmarks

- compare against: `uniform`

## Expected comparison behavior

TODO: describe what a healthy compare-against-baselines run should look like.

## Known failure modes and caveats

TODO: describe where this model is brittle, not yet promoted, or sensitive to data coverage.
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
) -> tuple[Path, Path, Path]:
    _validate_inputs(
        tier=tier,
        family=family,
        strategy_id=strategy_id,
        class_name=class_name,
    )
    module_path = module_path_for(root=root, tier=tier, family=family, strategy_id=strategy_id)
    test_path = test_path_for(root=root, strategy_id=strategy_id)
    model_card_path = model_card_path_for(root=root, strategy_id=strategy_id)
    if module_path.exists():
        raise FileExistsError(f"Strategy module already exists: {module_path}")
    if test_path.exists():
        raise FileExistsError(f"Strategy test already exists: {test_path}")
    if model_card_path.exists():
        raise FileExistsError(f"Strategy model card already exists: {model_card_path}")

    strategy_root = root / "stacksats"
    family_package = module_path.parent
    _ensure_package_chain(family_package, stop_at=strategy_root)
    module_path.write_text(
        module_source(class_name=class_name, strategy_id=strategy_id, intent=intent),
        encoding="utf-8",
    )

    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_package = root / "tests"
    _ensure_package_chain(test_path.parent, stop_at=test_package)
    test_path.write_text(
        test_source(
            module_import=(
                f"stacksats.strategies.{tier}.{family}.{_snake_name(strategy_id)}"
            ),
            class_name=class_name,
            strategy_id=strategy_id,
            intent=intent,
        ),
        encoding="utf-8",
    )

    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(
        model_card_source(
            tier=tier,
            family=family,
            strategy_id=strategy_id,
            class_name=class_name,
            intent=intent,
        ),
        encoding="utf-8",
    )

    catalog_path = root / "stacksats" / "strategies" / "catalog.py"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing strategy catalog: {catalog_path}")

    insert_catalog_stub(
        catalog_path,
        catalog_entry_stub(
            tier=tier,
            family=family,
            strategy_id=strategy_id,
            class_name=class_name,
            intent=intent,
        ),
        tier=tier,
    )
    return module_path, test_path, model_card_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a built-in strategy.")
    parser.add_argument("--tier", choices=("stable", "experimental"), required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--intent", choices=("propose", "profile"), required=True)
    args = parser.parse_args()

    module_path, test_path, model_card_path = scaffold_strategy(
        root=ROOT,
        tier=args.tier,
        family=args.family,
        strategy_id=args.strategy_id,
        class_name=args.class_name,
        intent=args.intent,
    )
    print(f"Created {module_path.relative_to(ROOT)}")
    print(f"Created {test_path.relative_to(ROOT)}")
    print(f"Created {model_card_path.relative_to(ROOT)}")
    print("Updated stacksats/strategies/catalog.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
