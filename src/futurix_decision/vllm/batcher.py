"""Micro-batching for the decision server: concurrent requests share one ``batch_decide``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import ConfigError, InvalidOutputError
from futurix_decision.inference.schemas import DecisionResult, check_confidence_threshold
from futurix_decision.models.protocol import DecisionBackend

T = TypeVar("T")


@dataclass(frozen=True)
class _Pending:
    example: DecisionInput
    confidence_threshold: float | None
    future: asyncio.Future[DecisionResult]

    def resolve(self, result: DecisionResult) -> None:
        if self.future.done():  # The client went away.
            return
        if self.confidence_threshold is not None:
            result = DecisionResult.from_probabilities(
                self.example,
                [result.probabilities[option] for option in self.example.options],
                latency_ms=result.latency_ms,
                confidence_threshold=self.confidence_threshold,
            )
        self.future.set_result(result)

    def fail(self, error: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(error)


class MicroBatcher:
    """Collect concurrent decisions into one ``backend.batch_decide`` call.

    The first waiting row opens a batch; it closes after ``window_ms`` or at
    ``max_batch`` rows, whichever comes first. Rows that arrive while a batch
    runs form the next one. One worker thread makes every backend call, so
    the backend never runs two calls at once and the event loop stays free.

    The batch runs without a confidence threshold; each row's own threshold
    is applied to its result. If the batch call raises, every row is retried
    alone, so an error reaches only the requests whose rows cause it.
    """

    def __init__(
        self, backend: DecisionBackend, *, window_ms: float = 5.0, max_batch: int = 32
    ) -> None:
        if not window_ms >= 0 or isinstance(max_batch, bool) or max_batch < 1:
            raise ConfigError("window_ms must be non-negative and max_batch positive")
        self.backend = backend
        self.window = window_ms / 1000
        self.max_batch = max_batch
        self.batches = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="futurix-decision")
        self._queue: asyncio.Queue[_Pending] | None = None
        self._worker: asyncio.Task[None] | None = None

    async def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        check_confidence_threshold(confidence_threshold)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DecisionResult] = loop.create_future()
        self._queue_for(loop).put_nowait(_Pending(example, confidence_threshold, future))
        return await future

    async def call(self, function: Callable[..., T], *args: Any) -> T:
        """Run ``function(*args)`` on the backend's worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(function, *args))

    def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._executor.shutdown(wait=False)

    def _queue_for(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue[_Pending]:
        # Start the worker on first use, in the loop that serves requests.
        if self._queue is None or self._worker is None or self._worker.get_loop() is not loop:
            self._queue = asyncio.Queue()
            self._worker = loop.create_task(self._work(self._queue))
        return self._queue

    async def _work(self, queue: asyncio.Queue[_Pending]) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await queue.get()]
            deadline = loop.time() + self.window
            while len(batch) < self.max_batch:
                if not queue.empty():
                    batch.append(queue.get_nowait())
                    continue
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                # asyncio.wait leaves an unfinished get cancellable without losing a row.
                getter = asyncio.ensure_future(queue.get())
                await asyncio.wait({getter}, timeout=remaining)
                if not getter.done():
                    getter.cancel()
                    break
                batch.append(getter.result())
            await self._run(batch)

    async def _run(self, batch: Sequence[_Pending]) -> None:
        rows = [item for item in batch if not item.future.done()]
        if not rows:
            return
        try:
            results = await self.call(self._batch_decide, [item.example for item in rows])
        except Exception as error:
            if len(rows) == 1:
                rows[0].fail(error)
                return
            for item in rows:
                await self._run([item])
            return
        for item, result in zip(rows, results, strict=True):
            try:
                item.resolve(result)
            except KeyError as error:
                item.fail(InvalidOutputError(f"backend result is missing option {error}"))

    def _batch_decide(self, examples: list[DecisionInput]) -> list[DecisionResult]:
        self.batches += 1
        results = self.backend.batch_decide(examples)
        if len(results) != len(examples):
            raise InvalidOutputError(
                f"backend returned {len(results)} results for {len(examples)} rows"
            )
        return results
