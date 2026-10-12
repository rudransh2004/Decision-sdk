"""Candidate-aware readouts for decoder hidden states.

Custom heads take a final-state vector, one vector per supplied option, and
an availability mask. They return one unnormalized logit per option; the SDK
handles masking, softmax, calibration, and the decision training losses.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from futurix_decision.models.letter_code import LetterCodeHead

HeadFactory = Callable[..., nn.Module]
_HEADS: dict[str, HeadFactory] = {}
CANDIDATE_MARKER = "<|decision_candidate|>"
HEAD_WEIGHTS = "decision_head.safetensors"


def register_decision_head(name: str, factory: HeadFactory) -> None:
    """Register a trusted factory for safe checkpoint reconstruction.

    A factory receives ``hidden_size=...`` and the JSON-serializable
    ``head_config`` values. Unknown names are never dynamically imported
    from checkpoint metadata.
    """
    if not name or not name.isidentifier() or not callable(factory):
        raise ValueError("head name must be an identifier and factory must be callable")
    if name in _HEADS:
        raise ValueError(f"decision head {name!r} is already registered")
    _HEADS[name] = factory


def build_decision_head(name: str, hidden_size: int, config: dict) -> nn.Module:
    if name not in _HEADS:
        raise ValueError(
            f"unknown decision head {name!r}; register_decision_head before loading this checkpoint"
        )
    head = _HEADS[name](hidden_size=hidden_size, **config)
    if not isinstance(head, nn.Module):
        raise TypeError("decision head factory must return a torch.nn.Module")
    return head


class PointerDecisionHead(nn.Module):
    """Trainable candidate readout with an optional state-dependent abstain logit."""

    def __init__(
        self, hidden_size: int, width: int | None = None, abstain: bool = False
    ) -> None:
        super().__init__()
        width = hidden_size if width is None else width
        if hidden_size < 1 or width < 1:
            raise ValueError("hidden_size and width must be positive")
        self.query = nn.Linear(hidden_size, width, bias=False)
        self.option = nn.Linear(hidden_size, width, bias=False)
        self.null = nn.Linear(hidden_size, 1) if abstain else None
        self.scale = math.sqrt(width)

    def forward(
        self, state: torch.Tensor, options: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2 or options.ndim != 3 or available.shape != options.shape[:2]:
            raise ValueError("expected state [B,H], options [B,N,H], available [B,N]")
        # A bf16 backbone can be paired with a fp32 head; autocast is optional.
        dtype = self.query.weight.dtype
        q = self.query(state.to(dtype))
        k = self.option(options.to(dtype))
        candidates = (q.unsqueeze(1) * k).sum(dim=-1) / self.scale
        return (
            torch.cat((candidates, self.null(state.to(dtype))), dim=-1)
            if self.null is not None else candidates
        )


register_decision_head("pointer", PointerDecisionHead)
register_decision_head("letter_code", LetterCodeHead)
