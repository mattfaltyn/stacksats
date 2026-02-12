"""Compatibility alias for `stacksats.btc_price_fetcher`."""

from importlib import import_module
import runpy
import sys
import warnings

warnings.warn(
    "`btc_price_fetcher` top-level module is deprecated. Use "
    "`stacksats.btc_price_fetcher` for imports.",
    DeprecationWarning,
    stacklevel=2,
)

_module = import_module("stacksats.btc_price_fetcher")
if __name__ == "__main__":
    runpy.run_module("stacksats.btc_price_fetcher", run_name="__main__")
else:
    sys.modules[__name__] = _module
