"""Compatibility alias for `stacksats.export_weights`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`export_weights` top-level module is deprecated. Use "
    "`stacksats.export_weights` or run `stacksats-export`.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.export_weights")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
