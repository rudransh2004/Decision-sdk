"""Recover all assigned decision-token scores from vLLM token logprobs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def probabilities_from_logprobs(
    token_logprobs: Mapping[int, Any], assigned_ids: Sequence[int], *, temperature: float = 1.0
) -> list[float]:
    """Condition on the assigned slots, then apply held-out temperature."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    missing = set(assigned_ids) - set(token_logprobs)
    if missing:
        raise ValueError(f"vLLM omitted decision token IDs: {sorted(missing)}")
    scores = [
        float(getattr(token_logprobs[token_id], "logprob", token_logprobs[token_id]))
        for token_id in assigned_ids
    ]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("vLLM returned non-finite decision logprobs")
    return torch.softmax(torch.tensor(scores, dtype=torch.float32) / temperature, dim=0).tolist()
