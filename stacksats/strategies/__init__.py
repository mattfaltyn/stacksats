"""Strategy interfaces and built-ins."""

from .base import CallableWindowStrategy, WindowStrategy
from .examples import MomentumStrategy, SimpleZScoreStrategy, UniformStrategy
from .mvrv import MVRVStrategy

__all__ = [
    "CallableWindowStrategy",
    "MomentumStrategy",
    "MVRVStrategy",
    "SimpleZScoreStrategy",
    "UniformStrategy",
    "WindowStrategy",
]
