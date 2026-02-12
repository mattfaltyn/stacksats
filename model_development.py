"""Compatibility alias for `stacksats.model_development`."""

from importlib import import_module
import sys
import warnings

warnings.warn(
    "`model_development` top-level module is deprecated. Use "
    "`stacksats.model_development` for imports.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.model_development")
sys.modules[__name__] = _module
