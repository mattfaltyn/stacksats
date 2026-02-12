"""BDD tests for model development.

This module wires the model_development.feature to step definitions.
"""

from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.step_defs.common_steps import *  # noqa: F401, F403
from tests.step_defs.weight_steps import *  # noqa: F401, F403

# Load all scenarios from the feature file
scenarios("model_development.feature")
