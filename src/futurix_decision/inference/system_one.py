"""Translate System One questions into ``DecisionInput``s and results back into answers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.errors import ConfigError
from futurix_decision.inference.schemas import DecisionResult


def system_one_inputs(
    state: State, questions: Mapping[str, Mapping[str, Any]]
) -> list[DecisionInput]:
    """One ``DecisionInput`` per named question, all over ``state``."""
    if not questions:
        raise ConfigError("system_one needs at least one question")
    examples = []
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id or not isinstance(question, Mapping):
            raise ConfigError("system_one questions must be named mappings")
        kind = question.get("type", "choice")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}
            ):
                raise ConfigError("noul criteria must use false/true keys")
            names = ["false", "true"]
            descriptions = [(criteria or {}).get(name) for name in names]
        elif kind in ("choice", "score"):
            if isinstance(criteria, Mapping):
                keys = (
                    sorted(criteria, key=lambda key: int(key))
                    if kind == "score"
                    else list(criteria)
                )
                names = [str(key) for key in keys]
                descriptions = [criteria[key] for key in keys]
            elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
                names = (
                    [str(i) for i in range(len(criteria))] if kind == "score" else list(criteria)
                )
                descriptions = list(criteria) if kind == "score" else [None] * len(names)
            else:
                raise ConfigError(f"{kind} question needs criteria options")
        else:
            raise ConfigError(f"unsupported system_one question type: {kind!r}")
        examples.append(
            DecisionInput(
                kind=kind,
                state=state,
                question=question.get("instructions") or question_id,
                options=names,
                candidate_descriptions=[
                    value
                    if isinstance(value, str)
                    else (
                        ""
                        if value is None
                        else json.dumps(value, ensure_ascii=False, sort_keys=True)
                    )
                    for value in descriptions
                ],
            )
        )
    return examples


def system_one_answers(
    questions: Mapping[str, Any],
    examples: Sequence[DecisionInput],
    results: Sequence[DecisionResult],
) -> dict[str, Any]:
    """``{"answers": {question_id: answer}}``, with an expected value for score questions."""
    answers = {}
    for question_id, example, result in zip(questions, examples, results, strict=True):
        answer: dict[str, Any] = {"type": example.kind, "probabilities": result.probabilities}
        if example.kind == "choice":
            answer["choice"] = result.decision
        elif example.kind == "noul":
            answer["noul"] = result.probabilities["true"]
        else:
            answer["score"] = sum(
                int(level) * probability for level, probability in result.probabilities.items()
            )
        answers[question_id] = answer
    return {"answers": answers}
