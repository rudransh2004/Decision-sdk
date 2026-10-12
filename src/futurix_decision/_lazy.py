"""Lazy package exports and optional-dependency checks, so the core imports without torch."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from futurix_decision.errors import MissingDependencyError

# Top-level modules that only the `torch` extra provides.
_TORCH_MODULES = frozenset({"torch", "transformers"})


@contextmanager
def torch_required(feature: str) -> Iterator[None]:
    """Turn a missing torch or transformers import into ``MissingDependencyError``.

    Only the absence of the top-level package counts: a missing submodule such
    as ``torch._C`` means a broken install, so that error propagates unchanged.
    """
    try:
        yield
    except ModuleNotFoundError as error:
        if error.name not in _TORCH_MODULES:
            raise
        raise MissingDependencyError(
            f"{feature} needs {error.name}: install futurix-decision[torch]"
        ) from error


def require_transformers(extra: str, minimum: tuple[int, int]) -> None:
    """Raise ``MissingDependencyError`` unless Transformers is at least ``minimum``."""
    try:
        release = version("transformers").split(".")
        installed = (int(release[0]), int(release[1]))
    except (PackageNotFoundError, ValueError, IndexError):
        installed = (0, 0)
    if installed < minimum:
        raise MissingDependencyError(
            f"{extra} needs Transformers {minimum[0]}.{minimum[1]}+; "
            f"install futurix-decision[{extra}] in a compatible environment"
        )


def lazy_exports(
    package: str, exports: Mapping[str, str]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return a module's ``__getattr__`` and ``__dir__`` for lazily imported names.

    ``__getattr__`` imports ``exports[name]`` on first access; ``__dir__``
    lists the exports alongside the names the module already holds.
    """

    def __getattr__(name: str) -> Any:
        module_name = exports.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        with torch_required(f"{package}.{name}"):
            module = importlib.import_module(module_name)
        value = getattr(module, name)
        setattr(sys.modules[package], name, value)
        return value

    def __dir__() -> list[str]:
        return sorted({*vars(sys.modules[package]), *exports})

    return __getattr__, __dir__
