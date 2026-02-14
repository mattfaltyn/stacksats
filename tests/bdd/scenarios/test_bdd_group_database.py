"""BDD scenarios that rely on common/database step definitions."""

from pytest_bdd import scenarios

from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.database_steps import *  # noqa: F401, F403

scenarios("consistency.feature")
scenarios("database_operations.feature")
scenarios("date_ranges.feature")
scenarios("edge_cases.feature")
scenarios("export_weights.feature")
