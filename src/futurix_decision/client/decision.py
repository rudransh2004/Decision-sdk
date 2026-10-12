"""Clients for the ``futurix-decision serve`` HTTP API."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import httpx
from pydantic import ValidationError

from futurix_decision.client._transport import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    build_async_client,
    build_client,
    check_settings,
    send,
    send_async,
)
from futurix_decision.datasets.schema import DecisionInput, DecisionKind, State
from futurix_decision.errors import ConfigError, InvalidOutputError
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
    choice_input,
    noul_input,
    score_input,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_KIND_PATHS = {
    "choice": "/v1/decision/choice",
    "noul": "/v1/decision/noul",
    "score": "/v1/decision/score",
}
# DecisionInput fields that the server's DecisionRequest has no place for.
_UNSENDABLE = (
    "candidate_ids",
    "candidate_labels",
    "candidate_descriptions",
    "native_question",
    "images",
)


def _body(example: DecisionInput, confidence_threshold: float | None) -> dict[str, Any]:
    """Translate a DecisionInput into the server's DecisionRequest JSON."""
    check_confidence_threshold(confidence_threshold)
    unsendable = [name for name in _UNSENDABLE if getattr(example, name) is not None]
    if example.preserve_order:
        unsendable.append("preserve_order")
    if unsendable:
        raise ConfigError(f"the decision server does not accept {', '.join(unsendable)}")
    body: dict[str, Any] = {
        "kind": example.kind,
        "state": example.state,
        "question": example.question,
    }
    if example.kind == "choice":
        body["choices"] = example.options
    elif example.kind == "score":
        # DecisionInput guarantees consecutive integer levels for score.
        body["min_score"] = int(example.options[0])
        body["max_score"] = int(example.options[-1])
    if confidence_threshold is not None:
        body["confidence_threshold"] = confidence_threshold
    return body


def _result(response: httpx.Response, kind: DecisionKind) -> DecisionResult:
    try:
        result = DecisionResult.model_validate_json(response.content)
    except ValidationError as error:
        raise InvalidOutputError(
            f"server returned an invalid decision: {error}",
            status_code=response.status_code,
            body=response.text,
        ) from error
    if result.kind != kind:
        raise InvalidOutputError(
            f"server answered a {result.kind} decision to a {kind} request",
            status_code=response.status_code,
            body=response.text,
        )
    return result


def _health(response: httpx.Response) -> dict[str, str]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise InvalidOutputError(
            "server returned an invalid health response",
            status_code=response.status_code,
            body=response.text,
        )
    return payload


def _check_injected(base_url: str | None, api_key: str | None, transport: object | None) -> None:
    if base_url is not None or api_key is not None or transport is not None:
        raise ConfigError(
            "configure base_url, api_key, and transport on the http_client you pass in"
        )


class DecisionClient:
    """Synchronous client for a ``futurix-decision serve`` server.

    ``base_url`` defaults to the server's default address. ``timeout`` (seconds)
    and ``max_retries`` apply to every request. Connect errors and HTTP 429,
    502, 503, and 504 are retried with exponential backoff and jitter, honoring
    ``Retry-After``. Other failures, and the last failed retry, raise
    ``BackendError`` carrying ``status_code`` and ``body``. A 502 that the
    server marks as invalid backend output raises ``InvalidOutputError`` at
    once, without retrying.

    Pass ``transport`` to route the client's own connection, or ``http_client``
    to use a configured ``httpx.Client`` (its base URL, headers, and timeout
    apply). The client closes only an ``httpx.Client`` it created.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        check_settings(timeout, max_retries)
        self.max_retries = max_retries
        self._owns_client = http_client is None
        if http_client is None:
            http_client = build_client(
                base_url or DEFAULT_BASE_URL, api_key=api_key, timeout=timeout, transport=transport
            )
        else:
            _check_injected(base_url, api_key, transport)
        self._http = http_client

    def _post(self, path: str, body: dict[str, Any]) -> DecisionResult:
        response = send(self._http, "POST", path, json=body, max_retries=self.max_retries)
        return _result(response, body["kind"])

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        """Decide one example and return its result."""
        return self._post("/v1/decision", _body(example, confidence_threshold))

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]:
        """Decide each example in order.

        The server has no batch endpoint, so this sends one request per example,
        sequentially. Use ``AsyncDecisionClient.batch_decide`` for concurrency.
        """
        bodies = [_body(example, confidence_threshold) for example in examples]
        return [self._post("/v1/decision", body) for body in bodies]

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Pick one of ``choices`` for ``state``."""
        body = _body(choice_input(state, choices, question), confidence_threshold)
        return self._post(_KIND_PATHS["choice"], body)

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Answer a yes/no question about ``state``; ``decision`` is a bool."""
        body = _body(noul_input(state, question), confidence_threshold)
        return self._post(_KIND_PATHS["noul"], body)

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Rate ``state`` from ``min_score`` to ``max_score``; ``decision`` is an int."""
        example = score_input(state, question, min_score, max_score)
        return self._post(_KIND_PATHS["score"], _body(example, confidence_threshold))

    def health(self) -> dict[str, str]:
        """Return the server's ``/health`` body, such as ``{"status": "ok"}``."""
        return _health(send(self._http, "GET", "/health", max_retries=self.max_retries))

    def close(self) -> None:
        """Close the connection pool if this client created it."""
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> DecisionClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncDecisionClient:
    """Asynchronous ``DecisionClient`` with the same settings and retry policy.

    Pass ``transport`` (an ``httpx.AsyncBaseTransport``) or ``http_client`` (an
    ``httpx.AsyncClient``) as with ``DecisionClient``. Use ``async with`` or
    ``await client.close()``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        check_settings(timeout, max_retries)
        self.max_retries = max_retries
        self._owns_client = http_client is None
        if http_client is None:
            http_client = build_async_client(
                base_url or DEFAULT_BASE_URL, api_key=api_key, timeout=timeout, transport=transport
            )
        else:
            _check_injected(base_url, api_key, transport)
        self._http = http_client

    async def _post(self, path: str, body: dict[str, Any]) -> DecisionResult:
        response = await send_async(
            self._http, "POST", path, json=body, max_retries=self.max_retries
        )
        return _result(response, body["kind"])

    async def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        """Decide one example and return its result."""
        return await self._post("/v1/decision", _body(example, confidence_threshold))

    async def batch_decide(
        self,
        examples: Sequence[DecisionInput],
        *,
        confidence_threshold: float | None = None,
        max_concurrency: int = 8,
    ) -> list[DecisionResult]:
        """Decide each example, returning results in input order.

        The server has no batch endpoint, so this sends one request per example,
        at most ``max_concurrency`` at a time. The first failure is raised.
        """
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ConfigError("max_concurrency must be a positive integer")
        bodies = [_body(example, confidence_threshold) for example in examples]
        semaphore = asyncio.Semaphore(max_concurrency)

        async def one(body: dict[str, Any]) -> DecisionResult:
            async with semaphore:
                return await self._post("/v1/decision", body)

        return list(await asyncio.gather(*(one(body) for body in bodies)))

    async def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Pick one of ``choices`` for ``state``."""
        body = _body(choice_input(state, choices, question), confidence_threshold)
        return await self._post(_KIND_PATHS["choice"], body)

    async def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Answer a yes/no question about ``state``; ``decision`` is a bool."""
        body = _body(noul_input(state, question), confidence_threshold)
        return await self._post(_KIND_PATHS["noul"], body)

    async def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Rate ``state`` from ``min_score`` to ``max_score``; ``decision`` is an int."""
        example = score_input(state, question, min_score, max_score)
        return await self._post(_KIND_PATHS["score"], _body(example, confidence_threshold))

    async def health(self) -> dict[str, str]:
        """Return the server's ``/health`` body, such as ``{"status": "ok"}``."""
        response = await send_async(self._http, "GET", "/health", max_retries=self.max_retries)
        return _health(response)

    async def close(self) -> None:
        """Close the connection pool if this client created it."""
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncDecisionClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
