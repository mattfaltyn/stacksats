"""StackSats package public API."""

from .api import BacktestResult, ValidationResult, run_backtest, validate_strategy
from .loader import load_strategy
from .model_development import precompute_features
from .prelude import load_data
from .strategies.base import CallableWindowStrategy, WindowStrategy
from .strategies.mvrv import MVRVStrategy

__all__ = [
    "BacktestResult",
    "CallableWindowStrategy",
    "MVRVStrategy",
    "ValidationResult",
    "WindowStrategy",
    "load_strategy",
    "load_data",
    "precompute_features",
    "run_backtest",
    "validate_strategy",
]
