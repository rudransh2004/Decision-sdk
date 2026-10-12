"""GLiNER2.5-Decide: Fastino's schema encoder, run through ``gliner2``'s classifier."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.datasets.schema import DecisionExample, DecisionInput
from futurix_decision.errors import ConfigError, InvalidOutputError, MissingDependencyError
from futurix_decision.models.families.publisher import (
    NativeExport,
    PublisherAdapter,
    PublisherRuntime,
)
from futurix_decision.models.sources import CheckpointFiles

if TYPE_CHECKING:
    from futurix_decision.models.native import NativeDecisionModel


def _text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _record(row: DecisionExample, *, target_policy: str) -> dict[str, Any]:
    """One hard-label classification task, as ``gliner2``'s trainer reads it."""
    if row.images:
        raise ConfigError("GLiNER's encoder cannot train on image decisions")
    target = row.target_distribution()
    if target_policy not in ("error", "argmax"):
        raise ConfigError("target_policy must be 'error' or 'argmax'")
    if max(target) < 1 - 1e-7 and target_policy != "argmax":
        raise ConfigError(
            "GLiNER's native classifier only accepts hard labels; use "
            "target_policy='argmax' to explicitly discard soft probabilities"
        )
    winner = row.options[max(range(len(target)), key=target.__getitem__)]
    labels = (
        dict(zip(row.options, row.candidate_descriptions, strict=True))
        if row.candidate_descriptions
        else None
    )
    task: dict[str, Any] = {
        "task": row.question_id or "decision",
        "labels": row.options,
        "true_label": [winner],
        "prompt": row.question,
    }
    if labels:
        task["label_descriptions"] = {
            name: description for name, description in labels.items() if description
        }
    return {"input": _text(row.state), "output": {"classifications": [task]}}


class GlinerRuntime(PublisherRuntime):
    """Classifies with one exclusive label task per question."""

    def decide(self, example: DecisionInput, payload: Mapping[str, Any]) -> Any:
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
        result = self.native.classify(_text(example.state), schema)
        if not result["decision"].exclusive:
            raise InvalidOutputError("GLiNER returned a multi-label rather than exclusive decision")
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
        return {
            "answers": {
                "decision": {
                    "type": example.kind,
                    "probabilities": values,
                    **({"noul": values["true"]} if example.kind == "noul" else {}),
                }
            }
        }

    def system_one(self, payload: Mapping[str, Any]) -> Any:
        from gliner2.classification import ClassificationSchema

        questions: Mapping[str, Mapping[str, Any]] = payload["questions"]
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
        result = self.native.classify(_text(payload["state"]), schema)
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
        return {"answers": answers}


class GlinerAdapter(PublisherAdapter):
    name = "gliner"
    extra = "gliner"
    readout = "schema-encoder"
    runtime_type = GlinerRuntime
    trainable = True
    http_error = "GLiNER does not publish a System One HTTP runtime"

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        try:
            from gliner2.classification import Classifier
        except ImportError as exc:
            raise MissingDependencyError(
                "gliner2 is required: install futurix-decision[gliner]"
            ) from exc
        runtime = Classifier.from_pretrained(
            name, device=device, **({"revision": revision} if revision else {}), **kwargs
        )
        # The publisher's constructor places encoded inputs on `device`
        # but leaves the extractor weights on CPU until `.to()` is called.
        runtime.to(device=device)
        return runtime

    def files(self, files: CheckpointFiles) -> None:
        files.require("config.json", "encoder_config/config.json")
        files.one_of("model.safetensors", "pytorch_model.bin")

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
        """Run ``gliner2``'s ``ExtractorTrainer``; it exports a release under ``final``."""
        from futurix_decision.trainers.native import NativeSFTConfig, write_rows

        target_policy = kwargs.pop("target_policy", "error")
        if kwargs:
            raise TypeError(f"unsupported GLiNER training options: {sorted(kwargs)}")
        count = write_rows(data, (_record(row, target_policy=target_policy) for row in rows))
        try:
            from gliner2 import AutoExtractor
            from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
        except ImportError as exc:
            raise MissingDependencyError(
                "install futurix-decision[gliner-train] to train GLiNER"
            ) from exc
        if config is None:
            config = TrainingConfig(output_dir=str(output))
        elif isinstance(config, NativeSFTConfig):
            if not config.train_backbone:
                raise ConfigError("GLiNER's native trainer does not expose readout-only training")
            config = TrainingConfig(
                output_dir=str(output),
                num_epochs=config.epochs,
                batch_size=config.batch_size,
                encoder_lr=config.learning_rate,
                task_lr=config.learning_rate,
                weight_decay=config.weight_decay,
                num_workers=0,
            )
        elif Path(config.output_dir) != output:
            raise ConfigError("GLiNER config.output_dir must equal output_dir")
        extractor = AutoExtractor.from_pretrained(model.checkpoint)
        report = ExtractorTrainer(extractor, config).train(train_data=str(data))
        return NativeExport(count, report, output / "final")
