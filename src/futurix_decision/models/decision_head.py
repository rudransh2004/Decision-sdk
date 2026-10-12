"""Slot assignment and masking for vocabulary-backed decision tokens."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def decision_token(slot: int) -> str:
    if slot < 0:
        raise ValueError("slot must be non-negative")
    return f"<|decision_{slot:03d}|>"


def assign_slots(options: Sequence[str], *, total_slots: int = 256, seed: int = 0) -> list[int]:
    """Map ordered candidates onto distinct, randomly selected decision slots."""
    if not 2 <= len(options) <= total_slots:
        raise ValueError("number of options must be between 2 and total_slots")
    return random.Random(seed).sample(range(total_slots), len(options))


def mask_unavailable_slots(
    logits: torch.Tensor,
    assigned_slots: Sequence[Sequence[int]],
    *,
    null_slot: int | None = None,
) -> torch.Tensor:
    """Keep assigned candidate slots and, when enabled, the reserved null slot."""
    if logits.ndim != 2 or logits.shape[0] != len(assigned_slots):
        raise ValueError("logits must have shape [batch, slots]")
    if null_slot is not None and not 0 <= null_slot < logits.shape[1]:
        raise ValueError("null slot index is out of range")
    available = torch.zeros_like(logits, dtype=torch.bool)
    for row, slots in enumerate(assigned_slots):
        if len(slots) < 2 or len(set(slots)) != len(slots):
            raise ValueError("each row must have at least two distinct slots")
        if any(not 0 <= slot < logits.shape[1] for slot in slots):
            raise ValueError("slot index is out of range")
        if null_slot is not None and null_slot in slots:
            raise ValueError("the reserved null slot cannot be assigned to a candidate")
        available[row, list(slots)] = True
        if null_slot is not None:
            available[row, null_slot] = True
    return logits.masked_fill(~available, float("-inf"))
