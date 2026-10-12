"""Token-budget batching shared by ``evaluate`` and ``DecisionModel.batch_decide``. No torch."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from futurix_decision.errors import ConfigError

T = TypeVar("T")


def token_budget_batches(
    rows: Iterable[T],
    length: Callable[[T], int],
    *,
    max_batch_tokens: int,
    batch_size: int | None = None,
) -> Iterator[list[T]]:
    """Group ``rows`` in order so each padded batch holds at most ``max_batch_tokens``.

    A padded batch costs its longest row times its row count, as ``length``
    measures them. A row longer than the budget forms a batch of its own.
    ``batch_size`` also caps the rows per batch.
    """
    if max_batch_tokens < 1 or (batch_size is not None and batch_size < 1):
        raise ConfigError("batch_size and max_batch_tokens must be positive")
    pending: list[T] = []
    longest = 0
    for row in rows:
        size = length(row)
        if pending and (
            (batch_size is not None and len(pending) >= batch_size)
            or max(longest, size) * (len(pending) + 1) > max_batch_tokens
        ):
            yield pending
            pending, longest = [], 0
        pending.append(row)
        longest = max(longest, size)
    if pending:
        yield pending
