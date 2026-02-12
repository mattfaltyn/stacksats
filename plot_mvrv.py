"""Compatibility alias for `stacksats.plot_mvrv`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`plot_mvrv` top-level module is deprecated. Use `stacksats.plot_mvrv` or "
    "run `stacksats-plot-mvrv`.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.plot_mvrv")
if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
