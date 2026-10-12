"""Laya: ConvAI's encoder agent, run through the ``laya`` package."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.datasets.schema import DecisionExample, DecisionInput
from futurix_decision.datasets.systemone import native_question
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


class LayaRuntime(PublisherRuntime):
    def system_one(self, payload: Mapping[str, Any]) -> Any:
        return self.native.predict(payload["state"], payload["questions"])

    def batch(self, examples: Sequence[DecisionInput]) -> BatchAnswers | None:
        """One ``predict_batch`` call per distinct question schema."""
        if not hasattr(self.native, "predict_batch"):
            return None
        grouped: dict[str, list[tuple[int, DecisionInput]]] = {}
        for index, row in enumerate(examples):
            if row.images:
                raise ConfigError("Laya's encoder does not accept images")
            schema = json.dumps(native_question(row), ensure_ascii=False, sort_keys=True)
            grouped.setdefault(schema, []).append((index, row))
        answers: dict[int, tuple[Mapping[str, Any], float]] = {}
        for schema, group in grouped.items():
            started = time.perf_counter()
            predictions = self.native.predict_batch(
                [row.state for _, row in group],
                {"decision": json.loads(schema)},
            )
            if len(predictions) != len(group):
                raise InvalidOutputError("Laya returned a different batch size")
            latency = (time.perf_counter() - started) * 1000 / len(group)
            for (index, _), prediction in zip(group, predictions, strict=True):
                answer = prediction.get("answers", {}).get("decision")
                if not isinstance(answer, Mapping):
                    raise InvalidOutputError("Laya did not answer the requested question")
                answers[index] = (answer, latency)
        return [answers[index] for index in range(len(examples))]


class LayaAdapter(PublisherAdapter):
    name = "laya"
    extra = "laya"
    readout = "encoder"
    runtime_type = LayaRuntime
    trainable = True

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        try:
            import laya
        except ImportError as exc:
            raise MissingDependencyError(
                "laya is required: install futurix-decision[laya]"
            ) from exc
        return laya.load(name, device=device, revision=revision, **kwargs)

    def files(self, files: CheckpointFiles) -> None:
        files.require("rl_agent_config.json", "model.safetensors")

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
        """Run ``laya.train.finetune`` on grouped System One records."""
        from futurix_decision.trainers.native import (
            NativeSFTConfig,
            grouped_records,
            text_only,
            write_rows,
        )

        if kwargs.keys() - {"device"}:
            raise TypeError(
                f"unsupported Laya training options: {sorted(kwargs.keys() - {'device'})}"
            )
        count = write_rows(data, grouped_records(text_only(rows, "Laya")))
        try:
            from laya.train import finetune
        except ImportError as exc:
            raise MissingDependencyError("install futurix-decision[laya] to train Laya") from exc
        if isinstance(config, NativeSFTConfig):
            from laya.train import TrainConfig

            config = TrainConfig(
                epochs=config.epochs,
                micro_batch=config.batch_size,
                grad_accum=1,
                encoder_lr=config.learning_rate,
                head_lr=config.learning_rate,
                weight_decay=config.weight_decay,
                freeze_encoder=not config.train_backbone,
            )
        report = finetune(
            str(data),
            model.checkpoint,
            str(output),
            config=config,
            device=kwargs.get("device", "auto"),
        )
        return NativeExport(count, report, output)
