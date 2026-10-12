"""PPLX Decider: Perplexity's noncausal readout, run through the publisher's ``autojev``."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.datasets.schema import DecisionExample, DecisionInput
from futurix_decision.datasets.systemone import native_question, option_keys
from futurix_decision.errors import ConfigError, InvalidOutputError, MissingDependencyError
from futurix_decision.models.families.publisher import (
    BatchAnswers,
    NativeExport,
    PublisherAdapter,
    PublisherRuntime,
)
from futurix_decision.models.sources import CheckpointFiles

if TYPE_CHECKING:
    from futurix_decision.models.native import NativeDecisionModel


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

    def train(
        self,
        model: NativeDecisionModel,
        rows: Iterable[DecisionExample],
        *,
        data: Path,
        output: Path,
        config: Any | None,
        **kwargs: Any,
    ) -> NativeExport:
        """Fit the readout (and backbone, unless frozen) on soft targets; ``save`` the release."""
        import torch
        import torch.nn.functional as functional

        from futurix_decision.trainers.native import NativeSFTConfig, sft_rows

        examples = sft_rows(self.name, rows, config, kwargs)
        settings = config or NativeSFTConfig()
        native = model.runtime
        native.requires_grad_(True)
        if not settings.train_backbone:
            native.backbone.requires_grad_(False)
        native.train(True)
        optimizer = torch.optim.AdamW(
            (p for p in native.parameters() if p.requires_grad),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        losses: list[float] = []
        for _epoch in range(settings.epochs):
            for offset in range(0, len(examples), settings.batch_size):
                batch_rows = examples[offset : offset + settings.batch_size]
                if any(len(row.options) > 255 for row in batch_rows):
                    raise ConfigError("PPLX native readout supports at most 255 options")
                records = [
                    {
                        "state": row.state,
                        "question": native_question(row),
                        **({"images": row.images} if row.images else {}),
                    }
                    for row in batch_rows
                ]
                logits = native(native.prepare(records))
                if logits.shape[0] != len(batch_rows):
                    raise InvalidOutputError("PPLX returned the wrong batch size")
                terms = []
                for prediction, row in zip(logits, batch_rows, strict=True):
                    if prediction.shape[0] < len(row.options):
                        raise InvalidOutputError("PPLX returned fewer logits than decision options")
                    target = torch.tensor(
                        row.target_distribution(), device=prediction.device, dtype=torch.float32
                    )
                    terms.append(
                        -(
                            target
                            * functional.log_softmax(prediction[: len(row.options)].float(), dim=-1)
                        ).sum()
                    )
                loss = torch.stack(terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        native.eval()
        native.save(output)
        return NativeExport(len(examples), {"train_items": len(examples), "losses": losses}, output)
