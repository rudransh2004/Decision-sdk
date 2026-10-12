"""Publisher-preserving adapters for the selected decision checkpoints.

Never replace a released decision head with an untrained Futurix head.
Publisher packages are imported only when their checkpoint is selected; each
family's ``PublisherAdapter`` in ``models/families`` holds its runtime.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, get_args, overload

import httpx

from futurix_decision._lazy import require_transformers, torch_required
from futurix_decision.client._transport import DEFAULT_MAX_RETRIES
from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.datasets.systemone import native_question
from futurix_decision.errors import CheckpointError, ConfigError, InvalidOutputError
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
    choice_input,
    noul_input,
    score_input,
)
from futurix_decision.models.families import identify_family, reject_native_checkpoint
from futurix_decision.models.families.futurix import QWEN35, on_device
from futurix_decision.models.families.http import HttpRuntime
from futurix_decision.models.families.publisher import (
    PUBLISHER_CHECKPOINTS,
    PublisherAdapter,
    pinned_remote,
)
from futurix_decision.models.protocol import DecisionBackend
from futurix_decision.models.registry import family_for_checkpoint, get_family
from futurix_decision.models.sources import checkpoint_source, infer_local_family

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel

ALIASES = {
    "bosun-0.6b": "Hanno-Labs/bosun-v3.1-0.6b",
    "bosun-1.7b": "Hanno-Labs/bosun-v3.1-1.7b",
    "kev": "jaredpalmer/kev-9b",
    "laya": "convaiinnovations/laya",
    "laya-multilingual": "convaiinnovations/laya-multilingual",
    "gliner": "fastino/GLiNER2.5-Decide",
    "clef": "Cloudflare/clef",
    "clef-flash": "Cloudflare/clef-flash",
    "pplx-decider": "perplexity-ai/pplx-decider-v1.1-27b",
    "openjev": "openjev/openjev",
    "openjev-flash": "openjev/OpenJev-Flash-9B",
    "nox": "vllm-sr/Decision-2.0-Nox-4B",
    "lux": "vllm-sr/Decision-2.0-Lux-9B",
    "liquid-d1": "LiquidAI/d1-3B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


@dataclass(frozen=True)
class NativeCapabilities:
    inference: bool
    training: bool
    vllm_inference: bool
    native_readout: str


def _publisher(family: str) -> PublisherAdapter:
    if family not in set(PUBLISHER_CHECKPOINTS.values()):
        raise ConfigError(f"unknown selected native family: {family}")
    adapter = get_family(family)
    assert isinstance(adapter, PublisherAdapter)
    return adapter


def _source(checkpoint: str | Path) -> tuple[str, str]:
    name = str(checkpoint)
    if name in ALIASES:
        name = ALIASES[name]
    name, local = checkpoint_source(name)
    if local:
        family = infer_local_family(Path(name))
        if family not in set(PUBLISHER_CHECKPOINTS.values()):
            raise CheckpointError(
                f"cannot identify native decision checkpoint at {name!r}; "
                "pass family='clef', 'pplx-decider', 'kev', 'laya', etc."
            )
        return name, family
    if name not in PUBLISHER_CHECKPOINTS:
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    return name, PUBLISHER_CHECKPOINTS[name]


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


class NativeDecisionModel:
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
            vllm_inference=adapter.vllm_server and backend == "http",
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

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(
            choice_input(state, choices, question), confidence_threshold=confidence_threshold
        )

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(noul_input(state, question), confidence_threshold=confidence_threshold)

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(
            score_input(state, question, min_score, max_score),
            confidence_threshold=confidence_threshold,
        )

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


Strategy = Literal["pointer", "bosun", "letter_code"]


@overload
def load_model(
    checkpoint: str | Path,
    *,
    family: str | None = None,
    convert: Literal[True],
    strategy: Strategy | None = None,
    head_type: str | None = None,
    train_backbone: bool | None = None,
    revision: str | None = None,
    device: str | None = None,
    **family_kwargs: Any,
) -> DecisionModel: ...


@overload
def load_model(
    checkpoint: str | Path,
    *,
    family: str | None = None,
    convert: bool = False,
    strategy: Strategy | None = None,
    head_type: str | None = None,
    train_backbone: bool | None = None,
    revision: str | None = None,
    device: str | None = None,
    **family_kwargs: Any,
) -> DecisionBackend: ...


def load_model(
    checkpoint: str | Path,
    *,
    family: str | None = None,
    convert: bool = False,
    strategy: Strategy | None = None,
    head_type: str | None = None,
    train_backbone: bool | None = None,
    revision: str | None = None,
    device: str | None = None,
    **family_kwargs: Any,
) -> DecisionBackend:
    """Load a known Hub release on demand or an existing local decision checkpoint.

    For a local copy with no recognizable publisher metadata, specify ``family``.
    This never converts a trained native decision readout into an untrained LM.

    - ``convert=True`` turns a raw causal LM into an untrained ``DecisionModel``
      with the readout that ``strategy`` names: ``"pointer"`` (default),
      ``"bosun"`` tokens, or a fixed ``"letter_code"`` table. Only Qwen3.5 has
      a noncausal adapter.
    - ``head_type`` and ``train_backbone`` configure Futurix checkpoints and
      conversions; a saved checkpoint rejects values that differ from its own.
    - ``revision`` pins the Hub commit. ``device`` places the model, for example
      ``"cuda:0"``; publisher runtimes default to ``"cpu"``.
    - ``family_kwargs`` go to the family's loader, for example ``base_url`` and
      ``trust_remote_code`` for publisher runtimes, or ``max_sequence_length``
      and Transformers ``from_pretrained`` arguments for ``DecisionModel``.

    Raises ``MissingDependencyError`` when the checkpoint needs torch and it is
    not installed.
    """
    if strategy is not None and strategy not in get_args(Strategy):
        raise ConfigError("strategy must be 'pointer', 'bosun', or 'letter_code'")
    if strategy is not None and not convert:
        raise ConfigError("strategy is only for converting a plain causal LM")
    if device is not None and "device_map" in family_kwargs:
        raise ConfigError("pass either device or device_map, not both")
    kwargs = dict(family_kwargs)
    for key, value in (
        ("revision", revision),
        ("head_type", head_type),
        ("train_backbone", train_backbone),
    ):
        if value is not None:
            kwargs[key] = value
    with torch_required(f"loading {checkpoint}"):
        return _load_model(
            checkpoint, convert=convert, family=family, strategy=strategy, device=device, **kwargs
        )


def _load_model(
    checkpoint: str | Path,
    *,
    convert: bool,
    family: str | None,
    strategy: Strategy | None,
    device: str | None,
    **kwargs: Any,
) -> DecisionBackend:
    original, _ = checkpoint_source(checkpoint)
    name, local = checkpoint_source(ALIASES.get(original, original))
    path = Path(name)
    registered = None if local else identify_family(name)
    if local:
        identified = infer_local_family(path)
    else:
        known = family_for_checkpoint(name)
        identified = known.name if known else (registered.name if registered else None)
    if family is not None and identified is not None and family != identified:
        raise CheckpointError(f"{name} belongs to {identified}, not {family}")
    family = family or identified
    if convert:
        known_as = family or (registered.name if registered else None)
        if known_as is not None:
            raise ConfigError(
                f"{name} belongs to the known decision-model family {known_as!r} and already "
                "contains a trained decision head; convert=True would discard its native "
                "readout. Load it without convert"
            )
        return on_device(_convert(name, local=local, strategy=strategy, **kwargs), device)
    if name in QWEN35:
        raise ConfigError(
            "Qwen3.5-9B is not decision-trained; pass convert=True and train before inference"
        )
    from futurix_decision.utils.config import load_decision_config

    if family == "futurix" and not local:
        metadata = load_decision_config(name, **kwargs)
        if not metadata:
            raise CheckpointError(
                f"incomplete Futurix Hub checkpoint {name!r}; missing: decision_config.json"
            )
        if metadata.get("architecture") != "futurix-decision":
            raise CheckpointError(f"{name} does not contain a Futurix decision checkpoint")
    if family is None and not local and name.count("/") == 1:
        if load_decision_config(name, **kwargs).get("architecture") == "futurix-decision":
            family = "futurix"
    if family is None:
        if local:
            raise CheckpointError(
                f"cannot identify local decision checkpoint at {name!r}; "
                "pass the correct family=... or include the publisher's native config/head"
            )
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    adapter = get_family(family)
    if local:
        adapter.requires(path)
    elif adapter.checkpoints and name not in adapter.checkpoints:
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    return adapter.load(name, device=device, **kwargs)


def _convert(name: str, *, local: bool, strategy: Strategy | None, **kwargs: Any) -> DecisionModel:
    """Turn a plain causal LM into an untrained ``DecisionModel`` with ``strategy``'s readout."""
    from futurix_decision.models.decision_model import DecisionModel

    if name in QWEN35:
        require_transformers("qwen3.5", (5, 17))
        if strategy == "bosun":
            if kwargs.pop("head_type", "tokens") != "tokens":
                raise ConfigError("strategy='bosun' requires head_type='tokens'")
            if kwargs.get("decision_tokens", 256) != 256:
                raise ConfigError("strategy='bosun' requires exactly 256 decision tokens")
            quantization = kwargs.get("quantization_config")
            if getattr(quantization, "load_in_4bit", False) or getattr(
                quantization, "load_in_8bit", False
            ):
                raise ConfigError("Bosun token rows must be added before quantizing the base")
            return DecisionModel.from_pretrained(
                name,
                head_type="tokens",
                prompt_format="bosun-v3",
                bosun_template_fallback=True,
                train_backbone=kwargs.pop("train_backbone", True),
                **kwargs,
            )
        if strategy == "letter_code":
            if kwargs.pop("head_type", "letter_code") != "letter_code":
                raise ConfigError("strategy='letter_code' requires head_type='letter_code'")
            return DecisionModel.from_pretrained(
                name,
                head_type="letter_code",
                head_prompt_version="letter-code-v1",
                decision_tokens=kwargs.pop("decision_tokens", 255),
                train_backbone=kwargs.pop("train_backbone", True),
                attention_mode=kwargs.pop("attention_mode", "noncausal_full_attention"),
                **kwargs,
            )
        if strategy == "pointer" and kwargs.get("head_type", "pointer") != "pointer":
            raise ConfigError("strategy='pointer' requires head_type='pointer'")
        return DecisionModel.from_pretrained(
            name, head_type=kwargs.pop("head_type", "pointer"), **kwargs
        )
    if not local and name.count("/") != 1:
        raise ConfigError(
            "convert=True needs an existing local directory or a Hugging Face model ID"
        )
    from futurix_decision.utils.config import load_decision_config

    metadata = load_decision_config(name, **kwargs)
    if metadata.get("architecture") == "futurix-decision":
        raise ConfigError(
            "this is already a trained Futurix decision checkpoint; omit convert=True"
        )
    reject_native_checkpoint(name, metadata)
    if not local and kwargs.get("trust_remote_code"):
        pinned_remote(kwargs.get("revision"), True, "conversion")
    from transformers import AutoConfig

    config_kwargs = {
        key: kwargs[key]
        for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
        if key in kwargs
    }
    config = AutoConfig.from_pretrained(name, **config_kwargs)
    architectures = getattr(config, "architectures", None) or ()
    selected = strategy or "pointer"
    qwen_multimodal = (
        selected == "letter_code"
        and getattr(config, "model_type", None) == "qwen3_5"
        and getattr(config, "vision_config", None) is not None
    )
    if getattr(config, "is_encoder_decoder", False) or (
        getattr(config, "vision_config", None) and not qwen_multimodal
    ):
        raise CheckpointError(
            "conversion supports text-only causal LMs, not seq2seq or multimodal models"
        )
    if (
        architectures
        and not any(
            architecture.endswith(("ForCausalLM", "LMHeadModel")) for architecture in architectures
        )
        and not (qwen_multimodal and "Qwen3_5ForConditionalGeneration" in architectures)
    ):
        raise CheckpointError(
            f"{name} does not declare a text causal LM architecture: {architectures}"
        )
    required_head = {"bosun": "tokens", "pointer": "pointer", "letter_code": "letter_code"}[
        selected
    ]
    head_type = kwargs.pop("head_type", required_head)
    if head_type != required_head:
        raise ConfigError(f"strategy={selected!r} requires head_type={required_head!r}")
    if selected == "bosun":
        if kwargs.get("decision_tokens", 256) != 256:
            raise ConfigError("strategy='bosun' requires exactly 256 decision tokens")
        quantization = kwargs.get("quantization_config")
        if getattr(quantization, "load_in_4bit", False) or getattr(
            quantization, "load_in_8bit", False
        ):
            raise ConfigError("Bosun token rows must be added before quantizing the base")
    train_backbone = kwargs.pop("train_backbone", selected != "pointer")
    if not isinstance(train_backbone, bool):
        raise ConfigError("train_backbone must be a boolean")
    if selected == "letter_code":
        if qwen_multimodal:
            require_transformers("qwen3.5", (5, 17))
        attention_mode = kwargs.pop(
            "attention_mode",
            "noncausal_full_attention"
            if getattr(config, "model_type", None) == "qwen3_5"
            else "causal",
        )
        if attention_mode != "causal" and getattr(config, "model_type", None) != "qwen3_5":
            raise ConfigError("noncausal letter-code attention needs a Qwen3.5-family backbone")
        if kwargs.get("decision_tokens", 255) > 255:
            raise ConfigError("letter-code readout supports at most 255 answer codes")
        return DecisionModel.from_pretrained(
            name,
            head_type="letter_code",
            head_prompt_version="letter-code-v1",
            decision_tokens=kwargs.pop("decision_tokens", 255),
            attention_mode=attention_mode,
            multimodal_loader=qwen_multimodal,
            train_backbone=train_backbone,
            **kwargs,
        )
    return DecisionModel.from_pretrained(
        name,
        head_type=head_type,
        **(
            {"prompt_format": "bosun-v3", "bosun_template_fallback": True}
            if selected == "bosun"
            else {"head_prompt_version": "span-v1"}
        ),
        train_backbone=train_backbone,
        **kwargs,
    )
