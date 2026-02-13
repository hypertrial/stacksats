"""BDD tests for database operations.

This module wires the database_operations.feature to step definitions.
"""

from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.database_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("database_operations.feature")
