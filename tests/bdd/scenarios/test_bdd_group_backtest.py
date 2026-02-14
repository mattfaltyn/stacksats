"""BDD scenarios that rely on backtest/common step definitions."""

from pytest_bdd import scenarios

from tests.bdd.step_defs.backtest_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403

scenarios("backtest.feature")
