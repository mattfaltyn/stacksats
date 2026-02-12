"""Compatibility alias for `stacksats.backtest`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`backtest` top-level module is deprecated. Use `stacksats.backtest` or run "
    "`stacksats-backtest` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.backtest")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
