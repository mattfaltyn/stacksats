"""Compatibility alias for `stacksats.modal_app`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`modal_app` top-level module is deprecated. Use `stacksats.modal_app` or "
    "run Modal commands against `stacksats/modal_app.py`.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.modal_app")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
