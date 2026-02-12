"""Utilities for loading user-defined strategy classes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path

from .strategies.base import WindowStrategy


def _load_module(module_or_path: str):
    """Load module from dotted path or `.py` file path."""
    if module_or_path.endswith(".py"):
        file_path = Path(module_or_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")

        module_hash = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:10]
        module_name = f"stacksats_user_strategy_{module_hash}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(module_or_path)


def load_strategy(spec: str) -> WindowStrategy:
    """Load and instantiate a strategy from `module_or_path:ClassName`.

    Examples:
    - `my_strategy.py:MyStrategy`
    - `my_package.strategies:MyStrategy`
    """
    module_or_path, sep, class_name = spec.rpartition(":")
    if not sep or not module_or_path or not class_name:
        raise ValueError(
            "Invalid strategy spec. Use format 'module_or_path:ClassName'."
        )

    module = _load_module(module_or_path)
    try:
        strategy_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Class '{class_name}' not found in '{module_or_path}'."
        ) from exc

    strategy = strategy_cls()
    compute_weights = getattr(strategy, "compute_weights", None)
    if not callable(compute_weights):
        raise TypeError(
            f"Strategy '{class_name}' must define a callable compute_weights method."
        )
    return strategy
