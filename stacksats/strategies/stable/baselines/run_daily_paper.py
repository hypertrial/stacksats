"""Canonical daily paper strategy."""

from __future__ import annotations

from ....strategy_types import BaseStrategy, DayState, ValidationConfig


class RunDailyPaperStrategy(BaseStrategy):
    """Canonical daily decision example for agent-native docs and smoke tests."""

    strategy_id = "run-daily-paper"
    version = "1.0.0"
    description = (
        "Canonical agent-facing daily decision example; not a benchmark strategy."
    )

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def propose_weight(self, state: DayState) -> float:
        return float(state.uniform_weight)

    def default_run_daily_validation_config(self) -> ValidationConfig:
        return ValidationConfig(min_win_rate=0.0, strict=False)


__all__ = ["RunDailyPaperStrategy"]
