"""HTTP transport shared by the decision clients and the native HTTP runtime.

Retries connect errors and 429, 502, 503, and 504 responses with exponential
backoff and full jitter, honoring ``Retry-After``. Every other failure, and the
last failed attempt, raises ``BackendError``. A response whose ``ERROR_HEADER``
says ``invalid_output`` raises ``InvalidOutputError`` without a retry.
"""

from __future__ import annotations

import asyncio
import email.utils
import math
import random
import time
from collections.abc import Mapping
from typing import Any

import httpx

from futurix_decision.errors import BackendError, ConfigError, InvalidOutputError
from futurix_decision.inference.schemas import ERROR_HEADER, INVALID_OUTPUT

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
RETRY_STATUSES = frozenset({429, 502, 503, 504})
# A connect failure means the server never saw the request, so retrying is safe.
_RETRY_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 8.0
_RETRY_AFTER_MAX = 60.0


def check_settings(timeout: float | None, max_retries: int) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ConfigError("timeout must be a positive number of seconds or None")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ConfigError("max_retries must be a non-negative integer")


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def build_client(
    base_url: str,
    *,
    api_key: str | None,
    timeout: float | None,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    return httpx.Client(
        base_url=base_url, timeout=timeout, headers=_headers(api_key), transport=transport
    )


def build_async_client(
    base_url: str,
    *,
    api_key: str | None,
    timeout: float | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, timeout=timeout, headers=_headers(api_key), transport=transport
    )


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = moment.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return min(max(seconds, 0.0), _RETRY_AFTER_MAX)


def _delay(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        retry_after = _retry_after(response)
        if retry_after is not None:
            return retry_after
    return random.uniform(0, min(_BACKOFF_MAX, _BACKOFF_BASE * 2**attempt))


def _invalid_output(response: httpx.Response) -> bool:
    """Whether the server says its backend returned invalid output, which a retry repeats."""
    return response.headers.get(ERROR_HEADER) == INVALID_OUTPUT


def _retryable(response: httpx.Response) -> bool:
    return response.status_code in RETRY_STATUSES and not _invalid_output(response)


def _status_error(response: httpx.Response, attempts: int) -> BackendError:
    request = response.request
    tried = f" after {attempts} attempts" if attempts > 1 else ""
    error = InvalidOutputError if _invalid_output(response) else BackendError
    return error(
        f"{request.method} {request.url} returned HTTP {response.status_code}{tried}",
        status_code=response.status_code,
        body=response.text,
    )


def _transport_error(error: httpx.TransportError, attempts: int) -> BackendError:
    tried = f" after {attempts} attempts" if attempts > 1 else ""
    try:
        target = f"{error.request.method} {error.request.url}"
    except RuntimeError:  # The error was raised before a request existed.
        target = "request"
    return BackendError(f"{target} failed{tried}: {type(error).__name__}: {error}")


def send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    json: Mapping[str, Any] | None = None,
) -> httpx.Response:
    """Send one request, retrying transient failures; return a 2xx response."""
    for attempt in range(max_retries + 1):
        last = attempt == max_retries
        try:
            response = client.request(method, url, json=json)
        except _RETRY_ERRORS as error:
            if last:
                raise _transport_error(error, attempt + 1) from error
            time.sleep(_delay(attempt, None))
            continue
        except httpx.TransportError as error:
            raise _transport_error(error, attempt + 1) from error
        if _retryable(response) and not last:
            response.close()
            time.sleep(_delay(attempt, response))
            continue
        if response.is_error:
            raise _status_error(response, attempt + 1)
        return response
    raise AssertionError("unreachable")


async def send_async(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int,
    json: Mapping[str, Any] | None = None,
) -> httpx.Response:
    """Async ``send``: same retry policy, sleeping with ``asyncio.sleep``."""
    for attempt in range(max_retries + 1):
        last = attempt == max_retries
        try:
            response = await client.request(method, url, json=json)
        except _RETRY_ERRORS as error:
            if last:
                raise _transport_error(error, attempt + 1) from error
            await asyncio.sleep(_delay(attempt, None))
            continue
        except httpx.TransportError as error:
            raise _transport_error(error, attempt + 1) from error
        if _retryable(response) and not last:
            await response.aclose()
            await asyncio.sleep(_delay(attempt, response))
            continue
        if response.is_error:
            raise _status_error(response, attempt + 1)
        return response
    raise AssertionError("unreachable")
