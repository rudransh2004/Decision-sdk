"""Publisher-preserving adapters for the selected decision checkpoints.

Never replace a released decision head with an untrained Futurix head.
Publisher packages are imported only when their checkpoint is selected.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from futurix_decision._lazy import torch_required
from futurix_decision.client._transport import (
    DEFAULT_MAX_RETRIES,
    build_client,
    check_settings,
    send,
)
from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    InvalidOutputError,
    MissingDependencyError,
)
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
    choice_input,
    noul_input,
    score_input,
)
from futurix_decision.models.families import identify_family, reject_native_checkpoint
from futurix_decision.models.sources import (
    checkpoint_source,
    infer_local_family,
    read_checkpoint_json,
    require_checkpoint_files,
)

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
_NATIVE = {
    "jaredpalmer/kev-9b": "kev",
    "convaiinnovations/laya": "laya",
    "convaiinnovations/laya-multilingual": "laya",
    "fastino/GLiNER2.5-Decide": "gliner",
    "Cloudflare/clef": "clef",
    "Cloudflare/clef-flash": "clef",
    "perplexity-ai/pplx-decider-v1.1-27b": "pplx-decider",
    "openjev/openjev": "openjev",
    "openjev/OpenJev-Flash-9B": "openjev",
    "vllm-sr/Decision-1.0-Nox-4B": "nox",
    "vllm-sr/Decision-2.0-Nox-4B": "nox",
    "vllm-sr/Decision-1.0-Lux-9B": "lux",
    "vllm-sr/Decision-2.0-Lux-9B": "lux",
    "LiquidAI/d1-3B": "liquid-d1",
}
_TRAINABLE = {"kev", "laya", "gliner", "clef", "pplx-decider"}


@dataclass(frozen=True)
class NativeCapabilities:
    inference: bool
    training: bool
    vllm_inference: bool
    native_readout: str


def _source(checkpoint: str | Path) -> tuple[str, str]:
    name = str(checkpoint)
    if name in ALIASES:
        name = ALIASES[name]
    name, local = checkpoint_source(name)
    if local:
        family = infer_local_family(Path(name))
        if family not in set(_NATIVE.values()):
            raise CheckpointError(
                f"cannot identify native decision checkpoint at {name!r}; "
                "pass family='clef', 'pplx-decider', 'kev', 'laya', etc."
            )
        return name, family
    if name not in _NATIVE:
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    return name, _NATIVE[name]


def _pinned_remote(
    revision: str | None,
    trust_remote_code: bool,
    family: str,
    *,
    local: bool = False,
) -> None:
    if not trust_remote_code or (
        not local
        and (
            revision is None
            or len(revision) != 40
            or any(digit not in "0123456789abcdefABCDEF" for digit in revision)
        )
    ):
        raise ConfigError(
            f"{family} runs publisher code: pass trust_remote_code=True"
            + ("" if local else " and an immutable 40-character revision from the model card")
            + ". Review that code before loading."
        )


def _require_hub_files(name: str, family: str, revision: str | None, *filenames: str) -> None:
    """Check small publisher metadata/heads before a large snapshot is downloaded."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for filename in filenames:
        try:
            hf_hub_download(name, filename, revision=revision)
        except EntryNotFoundError as exc:
            raise CheckpointError(
                f"incomplete {family} Hub checkpoint {name!r}; missing: {filename}"
            ) from exc


def _requires_transformers(extra: str, minimum: tuple[int, int]) -> None:
    try:
        release = version("transformers").split(".")
        installed = (int(release[0]), int(release[1]))
    except (PackageNotFoundError, ValueError, IndexError):
        installed = (0, 0)
    if installed < minimum:
        raise MissingDependencyError(
            f"{extra} needs Transformers {minimum[0]}.{minimum[1]}+; "
            f"install futurix-decision[{extra}] in a compatible environment"
        )


def _images_for_native(images: Sequence[Any], family: str) -> list[Any]:
    if family in ("clef", "liquid-d1"):
        from transformers.image_utils import load_image

        return [load_image(image) if isinstance(image, str) else image for image in images]
    return list(images)


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise InvalidOutputError(
            f"{response.request.url} did not return JSON",
            status_code=response.status_code,
            body=response.text,
        ) from exc


def _verify_openjev(client: httpx.Client, checkpoint: str, max_retries: int) -> None:
    info = _json(send(client, "GET", "/v1/version", max_retries=max_retries))
    if not isinstance(info, Mapping):
        raise InvalidOutputError("OpenJev /v1/version did not return a JSON object")
    expected = checkpoint.rsplit("/", 1)[-1].lower()
    served = str(info.get("model_dir", "")).lower()
    temperature, noul_temperature = (1.07, 1.074766) if "Flash" in checkpoint else (0.85, 1.829074)
    flags = info.get("flags", {})
    if not isinstance(flags, Mapping) or (
        served != expected
        or not math.isclose(float(info.get("T", -1)), temperature, abs_tol=1e-6)
        or not math.isclose(float(info.get("noul_t", -1)), noul_temperature, abs_tol=1e-6)
        or not math.isclose(float(info.get("noul_bias", -1)), 0, abs_tol=1e-6)
        or flags.get("targeted") is not True
        or flags.get("instr_style") != "pyrepr"
    ):
        raise BackendError(
            f"OpenJev helper does not serve {checkpoint} with its published "
            "decision readout and calibration; inspect /v1/version"
        )


def _question(example: DecisionInput) -> dict[str, Any]:
    if example.native_question is not None:
        question = copy.deepcopy(example.native_question)
        criteria = question.get("criteria")
        if question.get("type", "choice") != example.kind:
            raise ConfigError("native question type differs from its canonical decision")
        if example.kind == "choice" and isinstance(criteria, Mapping):
            if list(criteria) != example.options:
                raise ConfigError("native choice criteria order differs from canonical options")
            return question
        if example.kind == "score" and isinstance(criteria, list):
            if len(criteria) != len(example.options):
                raise ConfigError("native score criteria count differs from canonical options")
            return question
        if example.kind == "noul" and (
            criteria is None
            or (isinstance(criteria, Mapping) and not (set(criteria) - {"false", "true"}))
        ):
            if set(example.options) != {"false", "true"}:
                raise ConfigError("native noul readouts require explicit false/true candidate IDs")
            return question
    if example.kind == "noul":
        if set(example.options) != {"false", "true"}:
            raise ConfigError("native noul readouts require explicit false/true candidate IDs")
        criteria = (
            {
                key: example.candidate_descriptions[example.options.index(key)]
                for key in ("false", "true")
            }
            if example.candidate_descriptions
            else None
        )
    elif example.kind == "score":
        if example.preserve_order or any(
            int(option) != index + int(example.options[0])
            for index, option in enumerate(example.options)
        ):
            raise ConfigError("native score readouts need consecutive, ordered integer levels")
        criteria = [
            (example.candidate_descriptions[i] or example.candidate_labels[i])
            if example.candidate_descriptions and example.candidate_labels
            else (example.candidate_descriptions[i] if example.candidate_descriptions else option)
            for i, option in enumerate(example.options)
        ]
    else:
        criteria = {}
        for i, name in enumerate(example.options):
            label = example.candidate_labels[i] if example.candidate_labels else name
            description = (
                example.candidate_descriptions[i] if example.candidate_descriptions else ""
            )
            criteria[name] = (
                f"{label}: {description}"
                if description and label != name
                else (description or (label if label != name else None))
            )
    return {
        "type": example.kind,
        "instructions": example.question,
        **({"criteria": criteria} if criteria is not None else {}),
    }


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


def _pplx_answer(example: DecisionInput, values: Sequence[float]) -> dict[str, Any]:
    keys = (
        ["false", "true"]
        if example.kind == "noul"
        else [str(i) for i in range(len(example.options))]
        if example.kind == "score"
        else example.options
    )
    if len(values) != len(keys):
        raise InvalidOutputError("PPLX returned a different number of option probabilities")
    return {
        "type": example.kind,
        "probabilities": dict(zip(keys, values, strict=True)),
        **({"noul": values[1]} if example.kind == "noul" else {}),
    }


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
        self.max_retries = max_retries
        self._owns_runtime = owns_runtime
        self.checkpoint, self.family, self.runtime, self.backend = (
            checkpoint,
            family,
            runtime,
            backend,
        )
        self.source_revision = source_revision
        self.capabilities = NativeCapabilities(
            inference=True,
            training=family in _TRAINABLE and backend != "http",
            vllm_inference=family == "openjev" and backend == "http",
            native_readout={
                "kev": "pointer",
                "laya": "encoder",
                "gliner": "schema-encoder",
                "clef": "joint-schema",
                "pplx-decider": "noncausal-readout",
                "openjev": "letter-logprobs",
                "nox": "candidate-head",
                "lux": "candidate-head",
                "liquid-d1": "verbalizer",
            }[family],
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
        if family not in set(_NATIVE.values()):
            raise ConfigError(f"unknown selected native family: {family}")
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
            if family not in set(_NATIVE.values()):
                raise ConfigError(f"unknown selected native family: {family}")
            name = resolved
            identified = infer_local_family(Path(name))
            if identified is not None and identified != family:
                raise CheckpointError(f"{name} belongs to {identified}, not {family}")
        else:
            name, identified = _source(checkpoint)
            if family is not None and family != identified:
                raise CheckpointError(f"{name} belongs to {identified}, not {family}")
            family = identified
        manifest = Path(name) / "futurix_native.json"
        if revision is None and manifest.is_file():
            revision = read_checkpoint_json(manifest).get("source_revision")
        if base_url is not None:
            if family == "gliner":
                raise ConfigError("GLiNER does not publish a System One HTTP runtime")
            if client is not None and api_key is not None:
                raise ConfigError("configure api_key headers on your supplied client")
            check_settings(timeout, max_retries)
            runtime = client or build_client(base_url, api_key=api_key, timeout=timeout)
            model = cls(
                name,
                family,
                runtime,
                "http",
                max_retries=max_retries,
                owns_runtime=client is None,
            )
            if family == "openjev" and verify_server:
                try:
                    _verify_openjev(runtime, name, max_retries)
                except BaseException:
                    model.close()
                    raise
            return model
        if client is not None:
            raise ConfigError("client needs base_url for a native System One server")
        if api_key is not None:
            raise ConfigError("api_key is only for a native System One server")
        if local and family != "openjev":
            require_checkpoint_files(Path(name), family, source=name)
        if family == "laya":
            try:
                import laya
            except ImportError as exc:
                raise MissingDependencyError(
                    "laya is required: install futurix-decision[laya]"
                ) from exc
            runtime = laya.load(name, device=device, revision=revision, **native_kwargs)
        elif family == "gliner":
            try:
                from gliner2.classification import Classifier
            except ImportError as exc:
                raise MissingDependencyError(
                    "gliner2 is required: install futurix-decision[gliner]"
                ) from exc
            runtime = Classifier.from_pretrained(
                name, device=device, **({"revision": revision} if revision else {}), **native_kwargs
            )
            # The publisher's constructor places encoded inputs on `device`
            # but leaves the extractor weights on CPU until `.to()` is called.
            runtime.to(device=device)
        elif family == "kev":
            try:
                from kev.checkpoint import Checkpoint, LoadOptions
            except ImportError as exc:
                raise MissingDependencyError(
                    "kev is required: install futurix-decision[kev-native]"
                ) from exc
            if not local:
                _require_hub_files(name, family, revision, "head.pt")
            source = f"{name}@{revision}" if revision and not Path(name).is_dir() else name
            tokenizer, model = Checkpoint(source).load(device, LoadOptions(**native_kwargs))
            runtime = (tokenizer, model)
        elif family == "openjev":
            raise ConfigError(
                "OpenJev's supported decision readout needs its vLLM System One helper; "
                "pass base_url"
            )
        elif family == "clef":
            _pinned_remote(revision, trust_remote_code, family, local=local)
            _requires_transformers("clef", (5, 10))
            from huggingface_hub import snapshot_download

            if not local:
                _require_hub_files(
                    name,
                    family,
                    revision,
                    "joint_schema_model.py",
                    "joint_head_config.json",
                    "joint_head.safetensors",
                )
            directory = Path(name) if local else Path(snapshot_download(name, revision=revision))
            require_checkpoint_files(directory, family, source=name)
            source = directory / "joint_schema_model.py"
            module_name = (
                "futurix_native_clef_"
                + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
            )
            spec = importlib.util.spec_from_file_location(module_name, source)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise
            model, processor = module.load_release_model(directory, device=device, **native_kwargs)
            runtime = (module, model, processor)
        elif family == "pplx-decider":
            _pinned_remote(revision, trust_remote_code, family, local=local)
            _requires_transformers("pplx-decider", (5, 17))
            try:
                from autojev.model import DecisionModel as PplxModel
            except ImportError as exc:
                raise MissingDependencyError(
                    "Install the publisher's pinned source/src/autojev package to load "
                    "PPLX; a Transformers causal model cannot restore its readout"
                ) from exc
            from huggingface_hub import snapshot_download

            if not local:
                _require_hub_files(
                    name, family, revision, "decision_config.json", "readout.safetensors"
                )
            directory = Path(name) if local else Path(snapshot_download(name, revision=revision))
            require_checkpoint_files(directory, family, source=name)
            runtime = PplxModel(checkpoint=directory, train=False, device=device, **native_kwargs)
        else:
            _pinned_remote(revision, trust_remote_code, family, local=local)
            _requires_transformers(
                "nox-lux" if family in ("nox", "lux") else "liquid-d1",
                (5, 17) if family in ("nox", "lux") else (5, 14),
            )
            from huggingface_hub import snapshot_download
            from transformers import AutoModel

            # Inspect the complete, immutable publisher package before remote
            # code can silently load an incomplete head/backbone as a base LM.
            directory = Path(name) if local else Path(snapshot_download(name, revision=revision))
            require_checkpoint_files(directory, family, source=name)
            runtime = AutoModel.from_pretrained(
                str(directory), trust_remote_code=True, **native_kwargs
            ).to(device)
        return cls(name, family, runtime, "local", source_revision=revision)

    def _predict(self, example: DecisionInput) -> Mapping[str, Any]:
        if self.family in ("kev", "pplx-decider") and len(example.options) > 255:
            raise ConfigError(f"{self.family} native readout supports at most 255 options")
        question = _question(example)
        payload = {
            "model": self.checkpoint,
            "state": example.state,
            "questions": {"decision": question},
        }
        if example.images:
            if self.family not in ("clef", "pplx-decider", "liquid-d1"):
                raise ConfigError(f"{self.family} does not expose an image decision path here")
            payload["images"] = (
                example.images
                if self.backend == "http"
                else _images_for_native(example.images, self.family)
            )
        if self.backend == "http":
            output = self._post_systemone(payload)
        elif self.family == "laya":
            output = self.runtime.predict(example.state, payload["questions"])
        elif self.family == "gliner":
            from gliner2.classification import ClassificationSchema

            labels = {
                option: (
                    example.candidate_descriptions[i]
                    if example.candidate_descriptions and example.candidate_descriptions[i]
                    else None
                )
                for i, option in enumerate(example.options)
            }
            schema = ClassificationSchema().single("decision", labels, instruction=example.question)
            text = (
                example.state
                if isinstance(example.state, str)
                else json.dumps(example.state, ensure_ascii=False)
            )
            result = self.runtime.classify(text, schema)
            if not result["decision"].exclusive:
                raise InvalidOutputError(
                    "GLiNER returned a multi-label rather than exclusive decision"
                )
            keys = example.options
            if example.kind == "score":
                values = dict(
                    zip(
                        (str(i) for i in range(len(keys))),
                        (result.probabilities("decision")[key] for key in keys),
                        strict=True,
                    )
                )
            else:
                values = dict(result.probabilities("decision"))
            output = {
                "answers": {
                    "decision": {
                        "type": example.kind,
                        "probabilities": values,
                        **({"noul": values["true"]} if example.kind == "noul" else {}),
                    }
                }
            }
        elif self.family == "kev":
            from kev.api import SystemOneRequest, to_answers, to_record
            from kev.model import admit

            tokenizer, model = self.runtime
            request = SystemOneRequest.model_validate(payload)
            record, metadata = to_record(request)
            encoded = admit(model, tokenizer, record)
            probabilities = [prob.tolist() for prob in model.probs(encoded)]
            output = {"answers": to_answers(probabilities, metadata)}
        elif self.family == "clef":
            module, model, processor = self.runtime
            output = module.systemone(model, processor, payload)
        elif self.family == "pplx-decider":
            values = self.runtime.predict(
                [
                    {
                        "state": example.state,
                        "question": question,
                        **({"images": payload["images"]} if example.images else {}),
                    }
                ]
            )[0]
            output = {"answers": {"decision": _pplx_answer(example, values)}}
        else:
            output = self.runtime.system_one(
                state=example.state,
                questions=payload["questions"],
                **({"images": payload["images"]} if example.images else {}),
            )
        if not isinstance(output, Mapping) or not isinstance(output.get("answers"), Mapping):
            raise InvalidOutputError("native model did not return a System One answers mapping")
        answer = output["answers"].get("decision")
        if not isinstance(answer, Mapping):
            raise InvalidOutputError("native model did not answer the requested question")
        return answer

    def _post_systemone(self, payload: Mapping[str, Any]) -> Any:
        response = send(
            self.runtime, "POST", "/v1/systemone", json=payload, max_retries=self.max_retries
        )
        return _json(response)

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
        if images and self.family not in ("clef", "pplx-decider", "liquid-d1"):
            raise ConfigError(f"{self.family} does not expose an image decision path here")
        payload = {"model": self.checkpoint, "state": state, "questions": dict(questions)}
        if images:
            payload["images"] = (
                list(images) if self.backend == "http" else _images_for_native(images, self.family)
            )
        if self.backend == "http":
            output = self._post_systemone(payload)
        elif self.family == "laya":
            output = self.runtime.predict(state, payload["questions"])
        elif self.family == "kev":
            from kev.api import SystemOneRequest, to_answers, to_record
            from kev.model import admit

            tokenizer, model = self.runtime
            record, metadata = to_record(SystemOneRequest.model_validate(payload))
            encoded = admit(model, tokenizer, record)
            output = {
                "answers": to_answers(
                    [prob.tolist() for prob in model.probs(encoded)],
                    metadata,
                )
            }
        elif self.family == "clef":
            module, model, processor = self.runtime
            output = module.systemone(model, processor, payload)
        elif self.family == "pplx-decider":
            from autojev.model import answer as native_answer

            records = [
                {
                    "state": state,
                    "question": question,
                    **({"images": payload["images"]} if images else {}),
                }
                for question in questions.values()
            ]
            probabilities = self.runtime.predict(records)
            if len(probabilities) != len(questions):
                raise InvalidOutputError("PPLX returned a different number of answers")
            output = {
                "answers": {
                    qid: native_answer(question, values)
                    for (qid, question), values in zip(
                        questions.items(), probabilities, strict=True
                    )
                }
            }
        elif self.family == "gliner":
            from gliner2.classification import ClassificationSchema

            schema = ClassificationSchema()
            for qid, question in questions.items():
                kind = question.get("type")
                criteria = question.get("criteria")
                if kind == "noul":
                    if criteria is not None and (
                        not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}
                    ):
                        raise ConfigError("noul criteria must be a false/true mapping")
                    labels = {key: (criteria or {}).get(key) for key in ("false", "true")}
                elif kind == "score" and isinstance(criteria, list):
                    labels = {str(i): description for i, description in enumerate(criteria)}
                elif kind == "choice" and isinstance(criteria, Mapping):
                    labels = dict(criteria)
                else:
                    raise ConfigError(f"{qid!r} needs native choice/noul/score criteria")
                schema.single(qid, labels, instruction=question.get("instructions") or qid)
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            result = self.runtime.classify(text, schema)
            answers = {}
            for qid, question in questions.items():
                probs = dict(result.probabilities(qid))
                if not probs or not result[qid].exclusive:
                    raise InvalidOutputError("GLiNER did not return exclusive label probabilities")
                winner = max(probs, key=probs.__getitem__)
                if question["type"] == "noul":
                    answer = {"type": "noul", "noul": probs["true"]}
                elif question["type"] == "score":
                    answer = {
                        "type": "score",
                        "score": sum(int(key) * value for key, value in probs.items()),
                        "probabilities": probs,
                    }
                else:
                    answer = {"type": "choice", "choice": winner, "probabilities": probs}
                answers[qid] = answer
            output = {"answers": answers}
        else:
            output = self.runtime.system_one(
                state=state,
                questions=payload["questions"],
                **({"images": payload["images"]} if images else {}),
            )
        if not isinstance(output, Mapping) or not isinstance(output.get("answers"), Mapping):
            raise InvalidOutputError("native model did not return a System One answers mapping")
        if set(output["answers"]) != set(questions):
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
        if self.backend == "local" and self.family == "pplx-decider":
            if any(len(row.options) > 255 for row in examples):
                raise ConfigError("PPLX native readout supports at most 255 options")
            started = time.perf_counter()
            rows = [
                {
                    "state": row.state,
                    "question": _question(row),
                    **({"images": row.images} if row.images else {}),
                }
                for row in examples
            ]
            predictions = self.runtime.predict(rows)
            if len(predictions) != len(examples):
                raise InvalidOutputError("PPLX returned a different batch size")
            latency = (time.perf_counter() - started) * 1000 / len(examples)
            return [
                _result(row, _pplx_answer(row, values), latency, confidence_threshold)
                for row, values in zip(examples, predictions, strict=True)
            ]
        if (
            self.backend == "local"
            and self.family == "laya"
            and hasattr(self.runtime, "predict_batch")
        ):
            grouped: dict[str, list[tuple[int, DecisionInput]]] = {}
            for index, row in enumerate(examples):
                if row.images:
                    raise ConfigError("Laya's encoder does not accept images")
                schema = json.dumps(_question(row), ensure_ascii=False, sort_keys=True)
                grouped.setdefault(schema, []).append((index, row))
            results: dict[int, DecisionResult] = {}
            for schema, group in grouped.items():
                started = time.perf_counter()
                predictions = self.runtime.predict_batch(
                    [row.state for _, row in group],
                    {"decision": json.loads(schema)},
                )
                if len(predictions) != len(group):
                    raise InvalidOutputError("Laya returned a different batch size")
                latency = (time.perf_counter() - started) * 1000 / len(group)
                for (index, row), prediction in zip(group, predictions, strict=True):
                    answer = prediction.get("answers", {}).get("decision")
                    if not isinstance(answer, Mapping):
                        raise InvalidOutputError("Laya did not answer the requested question")
                    results[index] = _result(row, answer, latency, confidence_threshold)
            return [results[index] for index in range(len(examples))]
        # Other native readouts can have heterogeneous schemas and runtimes.
        return [self.decide(row, confidence_threshold=confidence_threshold) for row in examples]

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
        if self.family in ("clef", "pplx-decider"):
            _pinned_remote(
                self.source_revision,
                True,
                self.family,
                local=Path(self.checkpoint).is_dir(),
            )
        from futurix_decision.trainers.native import finetune_native

        return finetune_native(self, dataset, output_dir, config=config, **kwargs)


def load_model(
    checkpoint: str | Path,
    *,
    convert: bool = False,
    family: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a known Hub release on demand or an existing local decision checkpoint.

    For a local copy with no recognizable publisher metadata, specify `family`.
    This never converts a trained native decision readout into an untrained LM.
    For raw LMs, ``convert=True`` accepts pointer, Bosun-style tokens, or a
    fixed letter-code readout. Only Qwen3.5 has a noncausal adapter.

    Raises ``MissingDependencyError`` when the checkpoint needs torch and it is
    not installed.
    """
    with torch_required(f"loading {checkpoint}"):
        return _load_model(checkpoint, convert=convert, family=family, **kwargs)


def _load_model(
    checkpoint: str | Path,
    *,
    convert: bool,
    family: str | None,
    **kwargs: Any,
) -> Any:
    strategy = kwargs.pop("strategy", None)
    if strategy == "pplx":  # Legacy call sites; never write this name to new exports.
        strategy = "letter_code"
    if strategy is not None and strategy not in ("pointer", "bosun", "letter_code"):
        raise ConfigError("strategy must be 'pointer', 'bosun', or 'letter_code'")
    if strategy is not None and not convert:
        raise ConfigError("strategy is only for converting a plain causal LM")
    if convert and kwargs.get("head_type") == "pplx":
        kwargs["head_type"] = "letter_code"
    original, _ = checkpoint_source(checkpoint)
    name, local = checkpoint_source(ALIASES.get(original, original))
    path = Path(name)
    registered = None if local else identify_family(name)
    identified = (
        infer_local_family(path)
        if local
        else (_NATIVE.get(name) or (registered.name if registered else None))
    )
    if identified == "encoder":  # Hub catalogue groups two different publisher runtimes.
        identified = _NATIVE.get(name)
    if family is not None and identified is not None and family != identified:
        raise CheckpointError(f"{name} belongs to {identified}, not {family}")
    family = family or identified
    if family == "futurix":
        if convert:
            raise ConfigError("already trained Futurix checkpoints do not need conversion")
        if not local:
            from futurix_decision.utils.config import load_decision_config

            metadata = load_decision_config(name, **kwargs)
            if not metadata:
                raise CheckpointError(
                    f"incomplete Futurix Hub checkpoint {name!r}; missing: decision_config.json"
                )
            if metadata.get("architecture") != "futurix-decision":
                raise CheckpointError(f"{name} does not contain a Futurix decision checkpoint")
        if local:
            require_checkpoint_files(path, "futurix")
            metadata = read_checkpoint_json(path / "decision_config.json")
            if metadata.get("source_checkpoint") in (
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-9B-Base",
            ):
                _requires_transformers("qwen3.5", (5, 17))
        from futurix_decision.models.decision_model import DecisionModel

        return DecisionModel.from_pretrained(name, **kwargs)
    if family == "bosun" and (
        local
        or name
        in (
            "Hanno-Labs/bosun-v3.1-0.6b",
            "Hanno-Labs/bosun-v3.1-1.7b",
        )
    ):
        if convert:
            raise ConfigError("Bosun v3.1 already contains a trained decision head")
        if local:
            require_checkpoint_files(path, "bosun")
        from futurix_decision.models.decision_model import DecisionModel

        return DecisionModel.from_bosun(name, **kwargs)
    if family in ("decider", "jpt", "eikos") and (
        local or (registered is not None and registered.supports(name, "inference"))
    ):
        if convert:
            raise ConfigError(f"{family} already contains a trained decision readout")
        if local:
            require_checkpoint_files(path, family)
        from futurix_decision.models.decision_model import DecisionModel

        return getattr(DecisionModel, f"from_{family}")(name, **kwargs)
    if name in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base"):
        if not convert:
            raise ConfigError(
                "Qwen3.5-9B is not decision-trained; pass convert=True and train before inference"
            )
        _requires_transformers("qwen3.5", (5, 17))
        from futurix_decision.models.decision_model import DecisionModel

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
    if convert:
        if family is not None or registered is not None:
            raise ConfigError(
                f"{name} is a known decision-model family; convert=True would discard "
                "its native readout. Load it through its supported importer instead"
            )
        if not local and name.count("/") != 1:
            raise ConfigError(
                "convert=True needs an existing local directory or a Hugging Face model ID"
            )
        from futurix_decision.models.decision_model import DecisionModel
        from futurix_decision.utils.config import load_decision_config

        metadata = load_decision_config(name, **kwargs)
        if metadata.get("architecture") == "futurix-decision":
            raise ConfigError(
                "this is already a trained Futurix decision checkpoint; omit convert=True"
            )
        reject_native_checkpoint(name, metadata)
        if not local and kwargs.get("trust_remote_code"):
            _pinned_remote(kwargs.get("revision"), True, "conversion")
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
                architecture.endswith(("ForCausalLM", "LMHeadModel"))
                for architecture in architectures
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
                _requires_transformers("qwen3.5", (5, 17))
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
    if family is None and not local and name.count("/") == 1:
        from futurix_decision.utils.config import load_decision_config

        if load_decision_config(name, **kwargs).get("architecture") == "futurix-decision":
            from futurix_decision.models.decision_model import DecisionModel

            return DecisionModel.from_pretrained(name, **kwargs)
    if local and family is None:
        raise CheckpointError(
            f"cannot identify local decision checkpoint at {name!r}; "
            "pass the correct family=... or include the publisher's native config/head"
        )
    return NativeDecisionModel.from_pretrained(name, family=family, **kwargs)
