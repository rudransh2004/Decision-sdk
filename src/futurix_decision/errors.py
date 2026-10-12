"""Exception hierarchy for the Futurix Decision SDK."""

from __future__ import annotations


class FuturixError(Exception):
    """Base class for errors raised by the SDK."""


class MissingDependencyError(FuturixError, ImportError):
    """An optional dependency needed for the requested feature is not installed."""
