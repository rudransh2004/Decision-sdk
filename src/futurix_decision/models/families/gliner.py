"""GLiNER2.5-Decide: Fastino's schema encoder, run through ``gliner2``'s classifier."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import ConfigError, InvalidOutputError, MissingDependencyError
from futurix_decision.models.families.publisher import PublisherAdapter, PublisherRuntime
from futurix_decision.models.sources import CheckpointFiles


def _text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


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
