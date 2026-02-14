"""BDD scenarios that rely on common/validation step definitions."""

from pytest_bdd import scenarios

from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.validation_steps import *  # noqa: F401, F403

scenarios("data_integrity.feature")
scenarios("weight_constraints.feature")
