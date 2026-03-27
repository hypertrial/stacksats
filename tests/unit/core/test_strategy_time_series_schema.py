from __future__ import annotations

import pytest

from stacksats.strategy_time_series_schema import (
    ColumnSpec,
    merge_schema_specs,
    schema_specs,
    validate_schema_specs,
)


def _extra(name: str) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        dtype="float64",
        required=False,
        description=f"{name} column",
        source="strategy",
    )


def test_merge_schema_specs_rejects_core_name_collisions() -> None:
    with pytest.raises(ValueError, match="collide with core WeightTimeSeries schema"):
        merge_schema_specs(schema_specs(), (_extra("weight"),))


def test_merge_schema_specs_accepts_noncore_extra_columns() -> None:
    merged = merge_schema_specs(schema_specs(), (_extra("signal_strength"),))
    assert any(spec.name == "signal_strength" for spec in merged)


def test_validate_schema_specs_allows_core_names_when_collision_checks_disabled() -> None:
    specs = validate_schema_specs((_extra("weight"),), forbid_core_name_collisions=False)
    assert specs[0].name == "weight"
