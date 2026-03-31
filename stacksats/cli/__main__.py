"""CLI package entrypoint for `python -m stacksats.cli`."""

from ..loader import load_strategy as current_load_strategy
from ..runner import StrategyRunner as current_strategy_runner
from . import main
import stacksats.cli as cli_module

# Rebind package-level call sites to the current loader/runner objects so
# `python -m stacksats.cli` honors monkeypatches on `stacksats.loader` and
# `stacksats.runner` without maintaining a second copy of the CLI module.
cli_module.load_strategy = current_load_strategy
cli_module.StrategyRunner = current_strategy_runner


if __name__ == "__main__":
    raise SystemExit(main())
