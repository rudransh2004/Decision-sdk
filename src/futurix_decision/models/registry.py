"""Register additional inference architectures without changing serving code."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from futurix_decision.errors import ConfigError

_backends: dict[str, Callable[..., Any]] = {}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    if not name or name in _backends or not callable(factory):
        raise ConfigError("backend name must be new and factory callable")
    _backends[name] = factory


def create_backend(name: str, checkpoint: str, **kwargs: Any) -> Any:
    if name == "transformers":
        from futurix_decision.models.decision_model import DecisionModel

        return DecisionModel.from_pretrained(checkpoint, **kwargs)
    if name == "vllm":
        from futurix_decision.vllm.engine import VLLMEngine

        return VLLMEngine(checkpoint, **kwargs)
    if name not in _backends:
        raise ConfigError(f"unknown decision backend: {name}")
    return _backends[name](checkpoint, **kwargs)
