"""PPLX Decider: Perplexity's noncausal readout, run through the publisher's ``autojev``."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.datasets.systemone import native_question, option_keys
from futurix_decision.errors import ConfigError, InvalidOutputError, MissingDependencyError
from futurix_decision.models.families.publisher import (
    BatchAnswers,
    PublisherAdapter,
    PublisherRuntime,
)
from futurix_decision.models.sources import CheckpointFiles


def _answer(example: DecisionInput, values: Sequence[float]) -> dict[str, Any]:
    keys = option_keys(example)
    if len(values) != len(keys):
        raise InvalidOutputError("PPLX returned a different number of option probabilities")
    return {
        "type": example.kind,
        "probabilities": dict(zip(keys, values, strict=True)),
        **({"noul": values[1]} if example.kind == "noul" else {}),
    }


class PplxRuntime(PublisherRuntime):
    """``native`` is ``autojev``'s model; ``predict`` takes one record per question."""

    def decide(self, example: DecisionInput, payload: Mapping[str, Any]) -> Any:
        record = {
            "state": example.state,
            "question": payload["questions"]["decision"],
            **({"images": payload["images"]} if "images" in payload else {}),
        }
        values = self.native.predict([record])[0]
        return {"answers": {"decision": _answer(example, values)}}

    def system_one(self, payload: Mapping[str, Any]) -> Any:
        from autojev.model import answer as native_answer

        questions: Mapping[str, Any] = payload["questions"]
        records = [
            {
                "state": payload["state"],
                "question": question,
                **({"images": payload["images"]} if "images" in payload else {}),
            }
            for question in questions.values()
        ]
        probabilities = self.native.predict(records)
        if len(probabilities) != len(questions):
            raise InvalidOutputError("PPLX returned a different number of answers")
        return {
            "answers": {
                qid: native_answer(question, values)
                for (qid, question), values in zip(questions.items(), probabilities, strict=True)
            }
        }

    def batch(self, examples: Sequence[DecisionInput]) -> BatchAnswers | None:
        if any(len(row.options) > 255 for row in examples):
            raise ConfigError("PPLX native readout supports at most 255 options")
        started = time.perf_counter()
        rows = [
            {
                "state": row.state,
                "question": native_question(row),
                **({"images": row.images} if row.images else {}),
            }
            for row in examples
        ]
        predictions = self.native.predict(rows)
        if len(predictions) != len(examples):
            raise InvalidOutputError("PPLX returned a different batch size")
        latency = (time.perf_counter() - started) * 1000 / len(examples)
        return [
            (_answer(row, values), latency)
            for row, values in zip(examples, predictions, strict=True)
        ]


class PplxAdapter(PublisherAdapter):
    name = "pplx-decider"
    extra = "pplx-decider"
    readout = "noncausal-readout"
    runtime_type = PplxRuntime
    trainable = True
    images = True
    max_options = 255
    remote_code = True
    transformers = (5, 17)

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        try:
            from autojev.model import DecisionModel as PplxModel
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the publisher's pinned source/src/autojev package to load "
                "PPLX; a Transformers causal model cannot restore its readout"
            ) from exc
        directory = self.snapshot(
            name,
            local=local,
            revision=revision,
            heads=("decision_config.json", "readout.safetensors"),
        )
        return PplxModel(checkpoint=directory, train=False, device=device, **kwargs)

    def files(self, files: CheckpointFiles) -> None:
        files.require("config.json", "decision_config.json", "readout.safetensors")
        files.tokenizer()
        files.weights()
