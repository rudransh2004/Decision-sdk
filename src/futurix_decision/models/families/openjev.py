"""OpenJev: letter log-probabilities, served only by the publisher's vLLM System One helper."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import httpx

from futurix_decision.client._transport import send
from futurix_decision.errors import BackendError, ConfigError, InvalidOutputError
from futurix_decision.models.families import server_calibration
from futurix_decision.models.families.http import response_json
from futurix_decision.models.families.publisher import PublisherAdapter


class OpenJevAdapter(PublisherAdapter):
    """Needs no local files: the SDK only reaches OpenJev over HTTP."""

    name = "openjev"
    extra = None
    readout = "letter-logprobs"
    vllm_server = True

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        raise ConfigError(
            "OpenJev's supported decision readout needs its vLLM System One helper; pass base_url"
        )

    def verify_server(self, client: httpx.Client, checkpoint: str, max_retries: int) -> None:
        info = response_json(send(client, "GET", "/v1/version", max_retries=max_retries))
        if not isinstance(info, Mapping):
            raise InvalidOutputError("OpenJev /v1/version did not return a JSON object")
        expected = checkpoint.rsplit("/", 1)[-1].lower()
        served = str(info.get("model_dir", "")).lower()
        calibration = server_calibration("openjev", checkpoint)
        flags = info.get("flags", {})
        if not isinstance(flags, Mapping) or (
            served != expected
            or not math.isclose(float(info.get("T", -1)), calibration.temperature, abs_tol=1e-6)
            or not math.isclose(
                float(info.get("noul_t", -1)), calibration.noul_temperature, abs_tol=1e-6
            )
            or not math.isclose(float(info.get("noul_bias", -1)), 0, abs_tol=1e-6)
            or flags.get("targeted") is not True
            or flags.get("instr_style") != "pyrepr"
        ):
            raise BackendError(
                f"OpenJev helper does not serve {checkpoint} with its published "
                "decision readout and calibration; inspect /v1/version"
            )
