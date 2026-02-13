"""BDD tests for forward-looking bias prevention.

This module wires the forward_looking.feature to step definitions.
"""

from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.weight_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("forward_looking.feature")
