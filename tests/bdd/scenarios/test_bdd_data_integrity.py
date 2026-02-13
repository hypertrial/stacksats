"""BDD tests for data integrity.

This module wires the data_integrity.feature to step definitions.
"""

from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.validation_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("data_integrity.feature")
