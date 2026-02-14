"""BDD scenarios that rely on common/weight step definitions."""

from pytest_bdd import scenarios

from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.weight_steps import *  # noqa: F401, F403

scenarios("forward_looking.feature")
scenarios("golden_snapshots.feature")
scenarios("model_development.feature")
scenarios("weight_computation.feature")
scenarios("weight_stability.feature")
