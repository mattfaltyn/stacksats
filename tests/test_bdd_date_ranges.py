"""BDD tests for date range generation.

This module wires the date_ranges.feature to step definitions.
"""

from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.step_defs.common_steps import *  # noqa: F401, F403
from tests.step_defs.database_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("date_ranges.feature")
