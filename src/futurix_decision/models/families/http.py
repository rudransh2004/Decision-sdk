"""A publisher's System One server over HTTP, as a ``PublisherRuntime``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from futurix_decision.client._transport import send
from futurix_decision.errors import InvalidOutputError
from futurix_decision.models.families.publisher import PublisherRuntime


def response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise InvalidOutputError(
            f"{response.request.url} did not return JSON",
            status_code=response.status_code,
            body=response.text,
        ) from exc


class HttpRuntime(PublisherRuntime):
    """Posts payloads to ``/v1/systemone``. Images go as the caller passed them."""

    def __init__(self, client: httpx.Client, max_retries: int) -> None:
        super().__init__(client)
        self.max_retries = max_retries

    def system_one(self, payload: Mapping[str, Any]) -> Any:
        response = send(
            self.native, "POST", "/v1/systemone", json=payload, max_retries=self.max_retries
        )
        return response_json(response)
