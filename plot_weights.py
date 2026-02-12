"""Compatibility alias for `stacksats.plot_weights`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`plot_weights` top-level module is deprecated. Use `stacksats.plot_weights` "
    "or run `stacksats-plot-weights`.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.plot_weights")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
