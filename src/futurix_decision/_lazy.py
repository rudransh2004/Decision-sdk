"""Lazy package exports, so the core imports without torch."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Any

from futurix_decision.errors import MissingDependencyError

# Top-level modules that only the `torch` extra provides.
_TORCH_MODULES = frozenset({"torch", "transformers"})


def lazy_exports(package: str, exports: Mapping[str, str]) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` that imports ``exports[name]`` on first access."""

    def __getattr__(name: str) -> Any:
        module_name = exports.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if (error.name or "").partition(".")[0] not in _TORCH_MODULES:
                raise
            raise MissingDependencyError(
                f"{package}.{name} needs {error.name}: install futurix-decision[torch]"
            ) from error
        value = getattr(module, name)
        setattr(sys.modules[package], name, value)
        return value

    return __getattr__
