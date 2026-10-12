"""Micro-batching for the decision server: concurrent requests share one ``batch_decide``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import BackendError, ConfigError, InvalidOutputError
from futurix_decision.inference.schemas import DecisionResult, check_confidence_threshold
from futurix_decision.models.protocol import DecisionBackend

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _shutting_down() -> BackendError:
    return BackendError("the decision server is shutting down")


@dataclass(frozen=True)
class _Pending:
    example: DecisionInput
    confidence_threshold: float | None
    future: asyncio.Future[DecisionResult]

    def resolve(self, result: object) -> None:
        if self.future.done():  # The client went away.
            return
        if not isinstance(result, DecisionResult):
            raise InvalidOutputError(f"backend returned {type(result).__name__}, not a result")
        missing = [option for option in self.example.options if option not in result.probabilities]
        if missing:
            raise InvalidOutputError(f"backend result is missing options {missing}")
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
    is applied to its result. If the batch raises ``ConfigError`` or
    ``InvalidOutputError``, which one row can cause, every row is retried
    alone so the error reaches only the requests whose rows cause it. Any
    other error, such as a ``BackendError`` from an unreachable server, fails
    the whole batch at once. ``batches`` counts batches, not those retries.

    ``start`` runs in the serving event loop (the app's lifespan calls it, and
    the first request does if no lifespan ran). ``close`` fails every request
    still waiting; a later ``start`` serves again.
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
        self._executor: ThreadPoolExecutor | None = None
        self._queue: asyncio.Queue[_Pending] | None = None
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the backend thread and the batching task in the running loop.

        Does nothing if they are running; restarts a worker that stopped.
        """
        loop = asyncio.get_running_loop()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="futurix-decision"
            )
        if self._queue is None or self._worker is None or self._worker.get_loop() is not loop:
            self._queue = asyncio.Queue()
            self._worker = None
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._work(self._queue))

    async def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        check_confidence_threshold(confidence_threshold)
        self.start()
        assert self._queue is not None
        future: asyncio.Future[DecisionResult] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Pending(example, confidence_threshold, future))
        return await future

    async def call(self, function: Callable[..., T], *args: Any) -> T:
        """Run ``function(*args)`` on the backend's worker thread."""
        self.start()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(function, *args))

    def close(self) -> None:
        """Stop the worker and fail every request still waiting for a result."""
        if self._worker is not None:
            self._worker.cancel()
        if self._queue is not None:
            while not self._queue.empty():
                self._queue.get_nowait().fail(_shutting_down())
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        self._executor = self._queue = self._worker = None

    async def _work(self, queue: asyncio.Queue[_Pending]) -> None:
        batch: list[_Pending] = []
        try:
            while True:
                batch = []
                await self._collect(queue, batch)
                try:
                    await self._run(batch)
                except Exception as error:
                    logger.exception("decision batch failed")
                    for item in batch:
                        item.fail(error)
        except asyncio.CancelledError:
            for item in batch:
                item.fail(_shutting_down())
            raise

    async def _collect(self, queue: asyncio.Queue[_Pending], batch: list[_Pending]) -> None:
        loop = asyncio.get_running_loop()
        batch.append(await queue.get())
        deadline = loop.time() + self.window
        while len(batch) < self.max_batch:
            if not queue.empty():
                batch.append(queue.get_nowait())
                continue
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            # asyncio.wait leaves an unfinished get cancellable without losing a row.
            getter = asyncio.ensure_future(queue.get())
            await asyncio.wait({getter}, timeout=remaining)
            if not getter.done():
                getter.cancel()
                return
            batch.append(getter.result())

    async def _run(self, batch: Sequence[_Pending], *, retry: bool = False) -> None:
        rows = [item for item in batch if not item.future.done()]
        if not rows:
            return
        if not retry:
            self.batches += 1
        try:
            results = await self.call(self._batch_decide, [item.example for item in rows])
        except (ConfigError, InvalidOutputError) as error:
            if len(rows) == 1:
                rows[0].fail(error)
                return
            for item in rows:
                await self._run([item], retry=True)
            return
        except Exception as error:
            for item in rows:
                item.fail(error)
            return
        for item, result in zip(rows, results, strict=True):
            try:
                item.resolve(result)
            except InvalidOutputError as error:
                item.fail(error)
            except Exception as error:
                item.fail(InvalidOutputError(f"backend returned a malformed result: {error}"))

    def _batch_decide(self, examples: list[DecisionInput]) -> list[DecisionResult]:
        results = self.backend.batch_decide(examples)
        if len(results) != len(examples):
            raise InvalidOutputError(
                f"backend returned {len(results)} results for {len(examples)} rows"
            )
        return results
