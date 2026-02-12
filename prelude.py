"""Compatibility alias for `stacksats.prelude`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`prelude` top-level module is deprecated. Use `stacksats.prelude` for imports.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.prelude")
sys.modules[__name__] = _module
