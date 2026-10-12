"""OpenJev: letter log-probabilities, served only by the publisher's vLLM System One helper."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter


class OpenJevAdapter(PublisherAdapter):
    """Needs no local files: the SDK only reaches OpenJev over HTTP."""

    name = "openjev"
    extra = None
