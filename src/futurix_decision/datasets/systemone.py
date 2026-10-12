"""Convert between System One questions and canonical decisions.

The public System One wire format groups typed questions by ID under a single
state. A dataset adds a `targets` mapping keyed by those question IDs. Each
target can be a hard option key, a boolean, or an aligned probability map/list.
``expand_systemone_row`` flattens such a record without dropping question
provenance; ``native_question`` turns one decision back into a question.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import ConfigError


def _description(value: Any) -> str:
    if value is None:
        return ""
    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def expand_systemone_row(row: Mapping[str, Any], *, index: int) -> Iterator[dict[str, Any]]:
    questions = row.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ConfigError("System One rows need a nonempty questions mapping")
    targets = row.get("targets", row.get("expected", row.get("gold")))
    if not isinstance(targets, Mapping):
        raise ConfigError("System One training rows need targets keyed by question ID")
    missing = set(questions) - set(targets)
    if missing:
        raise ConfigError(f"System One row is missing targets for questions: {sorted(missing)}")
    extra = set(targets) - set(questions)
    if extra:
        raise ConfigError(f"System One row has unknown target question IDs: {sorted(extra)}")
    group_id = str(row.get("id", f"row-{index:08d}"))
    if row.get("videos") or row.get("media"):
        raise ConfigError("video and generic media datasets need a model-specific adapter")
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ConfigError("System One question IDs must be nonempty strings")
        if not isinstance(question, Mapping):
            raise ConfigError(f"question {question_id!r} must be a mapping")
        kind = question.get("type", "choice")
        if kind not in ("choice", "noul", "score"):
            raise ConfigError(f"question {question_id!r} has unsupported type {kind!r}")
        criteria = question.get("criteria")
        names: list[str]
        descriptions: list[str]
        if kind == "noul":
            if criteria is not None and not isinstance(criteria, Mapping):
                raise ConfigError(f"noul question {question_id!r} criteria must be a mapping")
            criteria = criteria or {}
            if set(criteria) - {"true", "false"}:
                raise ConfigError(f"noul question {question_id!r} has unknown criteria")
            names = ["false", "true"]
            descriptions = [_description(criteria.get(name)) for name in names]
        elif isinstance(criteria, Mapping):
            if kind == "score":
                try:
                    keys = sorted(criteria, key=lambda key: int(key))
                except (ValueError, TypeError) as exc:
                    raise ConfigError("score criteria map keys must be integer levels") from exc
                names = [str(key) for key in keys]
            else:
                keys = list(criteria)
                names = [str(key) for key in keys]
            descriptions = [_description(criteria[key]) for key in keys]
        elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
            if kind == "score":
                names = [str(i) for i in range(len(criteria))]
                descriptions = [_description(value) for value in criteria]
            else:
                if not all(isinstance(item, str) for item in criteria):
                    raise ConfigError("choice criteria list must contain option strings")
                names = list(criteria)
                descriptions = [""] * len(names)
        else:
            raise ConfigError(f"question {question_id!r} needs criteria options")
        target = targets[question_id]
        if isinstance(target, Mapping):
            if "probabilities" in target:
                target = target["probabilities"]
            elif kind in target:
                target = target[kind]
            elif "target" in target:
                target = target["target"]
        if kind == "score" and isinstance(target, float) and not target.is_integer():
            raise ConfigError(
                "score training needs a level label or a distribution, not a mean score"
            )
        instructions = question.get("instructions") or question_id
        yield {
            "id": f"{group_id}:{question_id}",
            "group_id": group_id,
            "question_id": question_id,
            "kind": kind,
            "state": row.get("state"),
            "question": _description(instructions),
            "native_question": dict(question),
            "images": row.get("images"),
            "options": names,
            "candidate_ids": names,
            "candidate_labels": names,
            "candidate_descriptions": descriptions,
            "target": target,
            **{key: row[key] for key in ("domain", "family", "source") if key in row},
        }


def native_question(example: DecisionInput) -> dict[str, Any]:
    """The System One question a publisher runtime reads for one canonical decision.

    Keeps ``example.native_question`` when it agrees with the canonical options,
    so a row read from System One JSONL round-trips unchanged.
    """
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


def option_keys(example: DecisionInput) -> list[str]:
    """The answer keys a publisher uses for ``example``'s options, in canonical order."""
    if example.kind == "noul":
        return ["false", "true"]
    if example.kind == "score":
        return [str(i) for i in range(len(example.options))]
    return list(example.options)
