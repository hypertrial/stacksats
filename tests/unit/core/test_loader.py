from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from stacksats.loader import load_strategy
from stacksats.strategy_types import BaseStrategy


def _write_module(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_load_strategy_from_file_path(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "good_strategy.py",
        """
from stacksats.strategy_types import BaseStrategy

class GoodStrategy(BaseStrategy):
    strategy_id = "good"

    def propose_weight(self, state):
        return state.uniform_weight
""",
    )

    strategy = load_strategy(f"{strategy_path}:GoodStrategy")
    assert isinstance(strategy, BaseStrategy)
    assert strategy.strategy_id == "good"


def test_load_strategy_from_module_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_module(
        tmp_path / "module_strategy.py",
        """
from stacksats.strategy_types import BaseStrategy

class ModuleStrategy(BaseStrategy):
    strategy_id = "module"

    def propose_weight(self, state):
        return state.uniform_weight
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    strategy = load_strategy("module_strategy:ModuleStrategy")
    assert isinstance(strategy, BaseStrategy)
    assert strategy.strategy_id == "module"


def test_load_strategy_invalid_spec_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid strategy spec"):
        load_strategy("invalid-spec")


def test_load_strategy_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Strategy file not found"):
        load_strategy("/tmp/does-not-exist.py:Missing")


def test_load_strategy_missing_class_raises_attribute_error(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "missing_class.py",
        """
from stacksats.strategy_types import BaseStrategy

class SomeOtherStrategy(BaseStrategy):
    strategy_id = "other"

    def propose_weight(self, state):
        return state.uniform_weight
""",
    )

    with pytest.raises(AttributeError, match="Class 'MissingClass' not found"):
        load_strategy(f"{strategy_path}:MissingClass")


def test_load_strategy_rejects_non_base_strategy(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "non_base.py",
        """
class NonBaseStrategy:
    pass
""",
    )

    with pytest.raises(TypeError, match="must subclass"):
        load_strategy(f"{strategy_path}:NonBaseStrategy")


def test_load_strategy_rejects_compute_weights_override(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "illegal_compute.py",
        """
import pandas as pd
from stacksats.strategy_types import BaseStrategy

class IllegalStrategy(BaseStrategy):
    strategy_id = "illegal"

    def propose_weight(self, state):
        return state.uniform_weight

    def compute_weights(self, ctx):
        return pd.Series(dtype=float)
""",
    )

    with pytest.raises(TypeError, match="Custom compute_weights overrides"):
        load_strategy(f"{strategy_path}:IllegalStrategy")


def test_load_strategy_requires_one_intent_hook(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "no_hooks.py",
        """
from stacksats.strategy_types import BaseStrategy

class NoHooksStrategy(BaseStrategy):
    strategy_id = "no-hooks"
""",
    )

    with pytest.raises(TypeError, match="must implement propose_weight"):
        load_strategy(f"{strategy_path}:NoHooksStrategy")


def test_load_strategy_rejects_non_callable_propose_weight(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "bad_propose.py",
        """
from stacksats.strategy_types import BaseStrategy

class BadProposeStrategy(BaseStrategy):
    strategy_id = "bad-propose"
    propose_weight = 123
""",
    )

    with pytest.raises(TypeError, match="must define callable propose_weight"):
        load_strategy(f"{strategy_path}:BadProposeStrategy")


def test_load_strategy_rejects_bad_propose_weight_signature(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "bad_propose_signature.py",
        """
from stacksats.strategy_types import BaseStrategy

class BadProposeSignatureStrategy(BaseStrategy):
    strategy_id = "bad-signature"

    def propose_weight(self, state, extra):
        return state.uniform_weight
""",
    )

    with pytest.raises(TypeError, match="propose_weight must use signature"):
        load_strategy(f"{strategy_path}:BadProposeSignatureStrategy")


def test_load_strategy_rejects_non_callable_build_target_profile(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "bad_profile_callable.py",
        """
from stacksats.strategy_types import BaseStrategy

class BadProfileCallableStrategy(BaseStrategy):
    strategy_id = "bad-profile-callable"
    build_target_profile = 456
""",
    )

    with pytest.raises(TypeError, match="must define callable build_target_profile"):
        load_strategy(f"{strategy_path}:BadProfileCallableStrategy")


def test_load_strategy_rejects_bad_build_target_profile_signature(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "bad_profile_signature.py",
        """
import pandas as pd
from stacksats.strategy_types import BaseStrategy

class BadProfileSignatureStrategy(BaseStrategy):
    strategy_id = "bad-profile-signature"

    def build_target_profile(self, ctx, features_df):
        return pd.Series(dtype=float)
""",
    )

    with pytest.raises(TypeError, match="build_target_profile must use signature"):
        load_strategy(f"{strategy_path}:BadProfileSignatureStrategy")


def test_load_strategy_requires_non_empty_strategy_id(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "missing_strategy_id.py",
        """
from stacksats.strategy_types import BaseStrategy

class MissingStrategyId(BaseStrategy):
    strategy_id = ""

    def propose_weight(self, state):
        return state.uniform_weight
""",
    )

    with pytest.raises(ValueError, match="must define non-empty strategy_id"):
        load_strategy(f"{strategy_path}:MissingStrategyId")


def test_load_strategy_merges_inline_and_file_config(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "configurable.py",
        """
from stacksats.strategy_types import BaseStrategy

class ConfigurableStrategy(BaseStrategy):
    strategy_id = "configurable"

    def __init__(self, alpha=1, beta=2):
        self.alpha = alpha
        self.beta = beta

    def propose_weight(self, state):
        return state.uniform_weight
""",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"beta": 99}), encoding="utf-8")

    strategy = load_strategy(
        f"{strategy_path}:ConfigurableStrategy",
        config={"alpha": 10, "beta": 5},
        config_path=str(config_path),
    )

    assert strategy.alpha == 10
    assert strategy.beta == 99


def test_load_strategy_cleans_sys_modules_on_import_failure(tmp_path: Path) -> None:
    strategy_path = _write_module(
        tmp_path / "broken_import.py",
        """
raise RuntimeError("boom during import")
""",
    )
    module_hash = hashlib.sha1(str(strategy_path.resolve()).encode("utf-8")).hexdigest()[:10]
    module_name = f"stacksats_user_strategy_{module_hash}"

    with pytest.raises(RuntimeError, match="boom during import"):
        load_strategy(f"{strategy_path}:AnyClass")

    assert module_name not in sys.modules
