"""Exception hierarchy for the Futurix Decision SDK."""

from __future__ import annotations


class FuturixError(Exception):
    """Base class for errors raised by the SDK."""


class ConfigError(FuturixError):
    """Bad arguments or configuration."""


class CheckpointError(FuturixError):
    """A checkpoint has missing files, belongs to another family, or has bad metadata."""


class MissingDependencyError(FuturixError, ImportError):
    """An optional dependency needed for the requested feature is not installed."""


class BackendError(FuturixError):
    """A transport or server failure.

    ``status_code`` is the HTTP status when the server answered, else ``None``.
    ``body`` is the response text when there was one, else ``None``.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class InvalidOutputError(BackendError):
    """A decision runtime returned probabilities or answers that fail validation."""
