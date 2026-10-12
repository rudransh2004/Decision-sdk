"""``NativeDecisionModel``: one decision API over publisher-preserving runtimes.

Never replace a released decision head with an untrained Futurix head.
Publisher packages are imported only when their checkpoint is selected; each
family's ``PublisherAdapter`` in ``models/families`` loads and runs its model.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from futurix_decision.client._transport import DEFAULT_MAX_RETRIES
from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.datasets.systemone import native_question
from futurix_decision.errors import CheckpointError, ConfigError, InvalidOutputError
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
)
from futurix_decision.models.families.http import HttpRuntime
from futurix_decision.models.families.publisher import PublisherAdapter, pinned_remote
from futurix_decision.models.protocol import DecisionShortcuts
from futurix_decision.models.registry import (
    family_for_checkpoint,
    get_family,
    infer_local_family,
    resolve_alias,
)
from futurix_decision.models.sources import checkpoint_source


@dataclass(frozen=True)
class NativeCapabilities:
    inference: bool
    training: bool
    vllm_inference: bool
    native_readout: str


def _publisher(family: str) -> PublisherAdapter:
    try:
        adapter = get_family(family)
    except ConfigError:
        adapter = None
    if not isinstance(adapter, PublisherAdapter):
        raise ConfigError(f"unknown selected native family: {family}")
    return adapter


def _source(checkpoint: str | Path) -> tuple[str, str]:
    name, local = checkpoint_source(resolve_alias(str(checkpoint)))
    if local:
        family = infer_local_family(Path(name))
        if family is None or not isinstance(get_family(family), PublisherAdapter):
            raise CheckpointError(
                f"cannot identify native decision checkpoint at {name!r}; "
                "pass family='clef', 'pplx-decider', 'kev', 'laya', etc."
            )
        return name, family
    adapter = family_for_checkpoint(name)
    if not isinstance(adapter, PublisherAdapter) or name not in adapter.checkpoints:
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    return name, adapter.name


def _answers(output: Any) -> Mapping[str, Any]:
    if not isinstance(output, Mapping) or not isinstance(output.get("answers"), Mapping):
        raise InvalidOutputError("native model did not return a System One answers mapping")
    answers: Mapping[str, Any] = output["answers"]
    return answers


def _result(
    example: DecisionInput,
    answer: Mapping[str, Any],
    latency_ms: float,
    confidence_threshold: float | None,
) -> DecisionResult:
    if answer.get("type") not in (None, example.kind):
        raise InvalidOutputError("publisher returned the wrong decision type")
    if example.kind == "noul":
        if "noul" not in answer:
            raise InvalidOutputError("publisher did not return P(true)")
        yes = float(answer["noul"])
        probabilities = {"false": 1.0 - yes, "true": yes}
    else:
        values = answer.get("probabilities")
        if not isinstance(values, Mapping):
            raise InvalidOutputError(
                "publisher must return per-option probabilities (not only a label/mean)"
            )
        keys = (
            [str(i) for i in range(len(example.options))]
            if example.kind == "score"
            else example.options
        )
        if set(values) != set(keys):
            raise InvalidOutputError(
                f"publisher returned different option IDs: {set(values) ^ set(keys)}"
            )
        probabilities = dict(
            zip(example.options, (float(values[key]) for key in keys), strict=True)
        )
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        raise InvalidOutputError("publisher returned invalid probabilities")
    total = sum(probabilities.values())
    if abs(total - 1.0) > len(probabilities) * 0.000051:
        raise InvalidOutputError("publisher probabilities do not sum to 1")
    return DecisionResult.from_probabilities(
        example,
        [probabilities[option] / total for option in example.options],
        latency_ms=latency_ms,
        confidence_threshold=confidence_threshold,
    )


class NativeDecisionModel(DecisionShortcuts):
    """One `decide`/`choice`/`noul`/`score` API over distinct publisher runtimes.

    `runtime` is public for advanced access to native hooks, heads, or processors.
    External HTTP inference requires the publisher's System One server, not an
    arbitrary generative vLLM endpoint.
    """

    def __init__(
        self,
        checkpoint: str,
        family: str,
        runtime: Any,
        backend: str,
        *,
        source_revision: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        owns_runtime: bool = False,
    ) -> None:
        adapter = _publisher(family)
        self.max_retries = max_retries
        self._owns_runtime = owns_runtime
        self.adapter = adapter
        """The family adapter that loaded this model and runs its training."""
        self.checkpoint, self.family, self.runtime, self.backend = (
            checkpoint,
            family,
            runtime,
            backend,
        )
        self._publisher = (
            HttpRuntime(runtime, max_retries)
            if backend == "http"
            else adapter.runtime_type(runtime)
        )
        self.source_revision = source_revision
        self.capabilities = NativeCapabilities(
            inference=True,
            training=adapter.trainable and backend != "http",
            vllm_inference=adapter.vllm == "server" and backend == "http",
            native_readout=adapter.readout,
        )
        # Required by the existing evaluation/journal interface.
        self.prompt_format = f"native-{family}"
        self.temperature = 1.0  # The publisher has already applied its own calibration.

    @classmethod
    def from_runtime(
        cls,
        checkpoint: str,
        family: str,
        runtime: Any,
        *,
        backend: str = "systemone",
        source_revision: str | None = None,
    ) -> NativeDecisionModel:
        """Inject an already loaded native model/server for custom loading and tests."""
        return cls(checkpoint, family, runtime, backend, source_revision=source_revision)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        timeout: float | None = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        device: str = "cpu",
        revision: str | None = None,
        trust_remote_code: bool = False,
        family: str | None = None,
        api_key: str | None = None,
        verify_server: bool = True,
        **native_kwargs: Any,
    ) -> NativeDecisionModel:
        """Load a publisher runtime locally, or connect to its System One server.

        With ``base_url``, requests use ``timeout`` (seconds) and retry connect
        errors and HTTP 429/502/503/504 up to ``max_retries`` times. Pass
        ``client`` to use your own ``httpx.Client``; ``close()`` closes only a
        client this method created.
        """
        resolved, local = checkpoint_source(checkpoint)
        if family is not None and local:
            _publisher(family)
            name = resolved
            identified = infer_local_family(Path(name))
            if identified is not None and identified != family:
                raise CheckpointError(f"{name} belongs to {identified}, not {family}")
        else:
            name, identified = _source(checkpoint)
            if family is not None and family != identified:
                raise CheckpointError(f"{name} belongs to {identified}, not {family}")
            family = identified
        return _publisher(family).open_model(
            name,
            local=local,
            check_files=True,
            base_url=base_url,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            device=device,
            revision=revision,
            trust_remote_code=trust_remote_code,
            api_key=api_key,
            verify_server=verify_server,
            **native_kwargs,
        )

    def _images(self, images: Sequence[Any]) -> list[Any]:
        if not self.adapter.images:
            raise ConfigError(f"{self.family} does not expose an image decision path here")
        return self._publisher.prepare_images(images)

    def _predict(self, example: DecisionInput) -> Mapping[str, Any]:
        limit = self.adapter.max_options
        if limit is not None and len(example.options) > limit:
            raise ConfigError(f"{self.family} native readout supports at most {limit} options")
        payload: dict[str, Any] = {
            "model": self.checkpoint,
            "state": example.state,
            "questions": {"decision": native_question(example)},
        }
        if example.images:
            payload["images"] = self._images(example.images)
        answer = _answers(self._publisher.decide(example, payload)).get("decision")
        if not isinstance(answer, Mapping):
            raise InvalidOutputError("native model did not answer the requested question")
        return answer

    def close(self) -> None:
        """Close the HTTP client that ``from_pretrained`` created; leave others open."""
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self) -> NativeDecisionModel:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        images: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Answer several native typed questions over the same state in one call.

        Keeps the publisher's answer shape, including an ordinal expected score
        and any native metadata. Images are only sent to verified media paths.
        """
        if not questions:
            raise ConfigError("system_one needs at least one named question")
        payload: dict[str, Any] = {
            "model": self.checkpoint,
            "state": state,
            "questions": dict(questions),
        }
        if images:
            payload["images"] = self._images(images)
        output = self._publisher.system_one(payload)
        if set(_answers(output)) != set(questions):
            raise InvalidOutputError("native model answered a different set of question IDs")
        return dict(output)

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        check_confidence_threshold(confidence_threshold)
        started = time.perf_counter()
        answer = self._predict(example)
        return _result(
            example, answer, (time.perf_counter() - started) * 1000, confidence_threshold
        )

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]:
        check_confidence_threshold(confidence_threshold)
        if not examples:
            return []
        batched = self._publisher.batch(examples) if self.backend == "local" else None
        if batched is None:
            return [self.decide(row, confidence_threshold=confidence_threshold) for row in examples]
        return [
            _result(row, answer, latency, confidence_threshold)
            for row, (answer, latency) in zip(examples, batched, strict=True)
        ]

    def finetune(
        self, dataset: Any, output_dir: str | Path, *, config: Any | None = None, **kwargs: Any
    ) -> NativeDecisionModel:
        """Train and reload in the publisher format; unsupported paths fail before loading data."""
        if not self.capabilities.training:
            raise NotImplementedError(
                f"{self.checkpoint} has no verified native training/export adapter; "
                "its inference-only runtime cannot safely preserve a fine-tuned decision head"
            )
        if self.adapter.remote_code:
            pinned_remote(
                self.source_revision,
                True,
                self.family,
                local=Path(self.checkpoint).is_dir(),
            )
        from futurix_decision.trainers.native import finetune_native

        return finetune_native(self, dataset, output_dir, config=config, **kwargs)
