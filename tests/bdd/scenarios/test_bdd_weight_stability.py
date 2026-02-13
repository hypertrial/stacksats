"""BDD tests for weight stability over time.

This module wires the weight_stability.feature to step definitions.
Tests verify that past weights remain stable when new data arrives.
"""

from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.weight_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("weight_stability.feature")
