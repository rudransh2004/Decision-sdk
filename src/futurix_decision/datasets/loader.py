"""Local, in-memory, and Hugging Face dataset loading."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from futurix_decision.datasets.benchmarks import normalize_decisionbench
from futurix_decision.datasets.schema import ABSTAIN_TARGET, DecisionExample
from futurix_decision.datasets.systemone import expand_systemone_row
from futurix_decision.errors import ConfigError, MissingDependencyError

OPEN_DECISIONS_DATASET = "SargeDev/jev-distill-corpus-v3"


def normalize_decision(row: Mapping[str, Any], *, index: int = 0) -> DecisionExample:
    """Normalize familiar external column names into the canonical schema.

    Unknown fields (such as dataset provenance) are intentionally ignored.
    An unrecognized target/schema fails explicitly rather than guessing labels.
    """
    if "primitive" in row and "candidates_json" in row:
        return normalize_decisionbench(row)
    if row.get("videos") or row.get("media"):
        raise ConfigError("video and generic media datasets need a model-specific adapter")
    kind = row.get("kind", row.get("type", "choice"))
    kind = {"binary": "noul", "boolean": "noul", "ordinal": "score"}.get(kind, kind)
    options = row.get("options", row.get("choices", row.get("candidates")))
    if options is None:
        if kind == "noul":
            options = ["false", "true"]
        elif kind == "score":
            options = [str(level) for level in range(6)]
        else:
            raise ConfigError("choice rows must provide options/choices/candidates")
    options = [str(option) for option in options]
    target = row.get("target", row.get("probabilities", row.get("label", row.get("answer"))))
    if target is None:
        raise ConfigError("row must provide target/probabilities/label/answer")
    if isinstance(target, Mapping):
        target = [target.get(option, 0.0) for option in options] + (
            [target[ABSTAIN_TARGET]] if ABSTAIN_TARGET in target else []
        )
    # Two published corpus streams store soft targets to four decimal places.
    # Three values of 0.3333, for example, sum to 0.9999. Only repair drift
    # within the maximum possible rounding error for these identified sources;
    # arbitrary invalid labels must still fail validation.
    if (
        row.get("source") in ("openjev_v2", "yuri_v3")
        and isinstance(target, list)
        and len(target) == len(options)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in target
        )
    ):
        total = sum(target)
        if total > 0 and abs(total - 1) <= len(target) * 0.0000501:
            target = [value / total for value in target]
    if isinstance(target, float) and kind == "noul":
        if not 0 <= target <= 1:
            raise ConfigError("noul probability must be in [0, 1]")
        target = [1.0 - target, target]
    if isinstance(target, str) and kind == "noul" and target.lower() in ("false", "true"):
        target = target.lower()
    question = row.get("question", row.get("query"))
    state = row.get("state", row.get("context", row.get("input")))
    return DecisionExample.model_validate(
        {
            "id": str(row.get("id", f"row-{index:08d}")),
            "kind": kind,
            "state": state,
            "question": question,
            "options": options,
            "target": target,
            "candidate_ids": row.get("candidate_ids"),
            "candidate_labels": row.get("candidate_labels"),
            "candidate_descriptions": row.get("candidate_descriptions"),
            "native_question": row.get("native_question"),
            "images": row.get("images"),
            "preserve_order": row.get("preserve_order", False),
            "group_id": row.get("group_id"),
            "question_id": row.get("question_id"),
            "domain": row.get("domain"),
            "family": row.get("family"),
            "source": row.get("source"),
        }
    )


def _jsonl_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise ConfigError(f"{path}:{line_number}: expected a JSON object")
            yield row


def _rows(
    source: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
    split: str,
    streaming: bool,
    revision: str | None = None,
) -> Iterable[Mapping[str, Any]]:
    if isinstance(source, Mapping):
        return [source]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix in (".json", ".jsonl"):
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix == ".jsonl":
                return _jsonl_rows(path)
            contents = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(contents, Mapping) and "data" in contents:
                contents = contents["data"]
            if isinstance(contents, Mapping):
                return [contents]
            if not isinstance(contents, list):
                raise ConfigError("JSON dataset must be a row or an array of rows")
            return contents
        if isinstance(source, Path) or path.exists():
            raise ConfigError("local dataset files must end in .json or .jsonl")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise MissingDependencyError(
                "Hugging Face loading requires pip install futurix-decision[datasets]"
            ) from exc
        options = {"revision": revision} if revision is not None else {}
        return load_dataset(str(source), split=split, streaming=streaming, **options)
    return source


def load_decisions(
    source: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    streaming: bool = False,
    revision: str | None = None,
) -> list[DecisionExample] | Iterator[DecisionExample]:
    """Load and validate decisions. With streaming=True, return a lazy iterator."""

    def validated() -> Iterator[DecisionExample]:
        for index, row in enumerate(_rows(source, split, streaming, revision)):
            if not isinstance(row, Mapping):
                raise ConfigError(f"row {index}: expected a mapping")
            try:
                if "questions" in row:
                    for question in expand_systemone_row(row, index=index):
                        yield normalize_decision(question, index=index)
                else:
                    yield normalize_decision(row, index=index)
            except (ConfigError, ValueError, TypeError) as exc:
                raise ConfigError(f"row {index}: {exc}") from exc

    return validated() if streaming else list(validated())


def load_open_decisions(
    *, split: str = "train", streaming: bool = True
) -> list[DecisionExample] | Iterator[DecisionExample]:
    """Load the public SargeDev corpus; stream by default (its train split is large)."""
    return load_decisions(OPEN_DECISIONS_DATASET, split=split, streaming=streaming)
