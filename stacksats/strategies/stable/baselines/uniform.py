"""Uniform baseline strategy."""

from __future__ import annotations

from ....strategy_types import BaseStrategy, DayState


class UniformStrategy(BaseStrategy):
    """Baseline strategy that allocates equally across the full date window."""

    strategy_id = "uniform"
    version = "1.0.0"
    description = "Uniform baseline strategy."

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def propose_weight(self, state: DayState) -> float:
        # Framework enforces clipping, remaining budget, and lock semantics.
        return float(state.uniform_weight)


__all__ = ["UniformStrategy"]
