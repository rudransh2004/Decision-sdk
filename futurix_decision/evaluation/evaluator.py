"""Batched, no-generation decision evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import sys
from typing import Any

import torch

from futurix_decision.datasets import DecisionExample, load_decisions, normalize_decision
from futurix_decision.datasets.systemone import expand_systemone_row
from futurix_decision.evaluation.metrics import DEFAULT_THRESHOLDS, compute_metrics
from futurix_decision.models import DecisionBatch, DecisionModel
from futurix_decision.models.bosun import format_bosun_prompt
from futurix_decision.models.tokenizer import format_decision_prompt

EvaluationSource = str | Path | Mapping[str, Any] | Iterable[DecisionExample | Mapping[str, Any]]


def iter_examples(dataset: EvaluationSource) -> Iterator[DecisionExample]:
    if isinstance(dataset, (str, Path, Mapping)):
        yield from load_decisions(dataset, streaming=True)
    else:
        for index, row in enumerate(dataset):
            if isinstance(row, DecisionExample):
                yield row
            elif "questions" in row:
                for question in expand_systemone_row(row, index=index):
                    yield normalize_decision(question, index=index)
            else:
                yield normalize_decision(row, index=index)


def example_batches(dataset: EvaluationSource, batch_size: int) -> Iterator[list[DecisionExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = iter_examples(dataset)
    while batch := list(islice(rows, batch_size)):
        yield batch


def token_budget_batches(
    model: Any, dataset: EvaluationSource, batch_size: int, max_batch_tokens: int
) -> Iterator[list[DecisionExample]]:
    """Keep input order while avoiding padding seven short rows to one long row."""
    if batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("batch_size and max_batch_tokens must be positive")
    pending: list[DecisionExample] = []
    longest = 0
    for row in iter_examples(dataset):
        if isinstance(model, DecisionModel):
            prompt, _ = model._prompt_and_slots(row, None)
        elif model.prompt_format == "bosun-v3":
            prompt, _ = format_bosun_prompt(
                row, model.tokenizer, decision_tokens=model.decision_tokens,
                seed=model.seed, prompt_schema=model.prompt_schema,
            )
        else:
            prompt, _ = format_decision_prompt(
                row, decision_tokens=model.decision_tokens, seed=model.seed,
            )
        length = len(model.tokenizer.encode(prompt, add_special_tokens=True))
        if pending and (
            len(pending) >= batch_size
            or max(longest, length) * (len(pending) + 1) > max_batch_tokens
        ):
            yield pending
            pending, longest = [], 0
        pending.append(row)
        longest = max(longest, length)
    if pending:
        yield pending


def _inference_batches(
    model: DecisionModel, examples: list[DecisionExample]
) -> Iterator[tuple[list[DecisionExample], DecisionBatch]]:
    """Keep exact rows and order when one unusually long batch exhausts VRAM."""
    try:
        yield examples, model.forward_decisions(examples)
    except torch.OutOfMemoryError:
        if len(examples) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(examples) // 2
        yield from _inference_batches(model, examples[:middle])
        yield from _inference_batches(model, examples[middle:])


def evaluate(
    model: Any,
    dataset: EvaluationSource,
    *,
    batch_size: int = 16,
    bins: int = 15,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    progress_every: int = 0,
    predictions_output: str | Path | None = None,
    resume: bool = False,
    run_id: str | None = None,
    max_batch_tokens: int | None = None,
) -> dict:
    """Evaluate stable inference slots; no token generation or per-item LM pass."""
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if resume and predictions_output is None:
        raise ValueError("resuming evaluation requires predictions_output")
    if max_batch_tokens is not None and max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be positive")
    predicted, targets = [], []
    next_progress = progress_every
    groups: dict[str, dict[str, list[int]]] = {
        "by_kind": defaultdict(list),
        "by_family": defaultdict(list),
        "by_domain": defaultdict(list),
    }

    def record(row: DecisionExample, p: torch.Tensor, y: torch.Tensor) -> None:
        index = len(predicted)
        predicted.append(p)
        targets.append(y)
        groups["by_kind"][row.kind].append(index)
        if row.family:
            groups["by_family"][row.family].append(index)
        if row.domain:
            groups["by_domain"][row.domain].append(index)

    rows = iter_examples(dataset)
    output = Path(predictions_output) if predictions_output is not None else None
    if output is not None:
        manifest = output.with_suffix(output.suffix + ".meta.json")
        identity = {
            "run_id": run_id,
            "prompt_format": model.prompt_format,
            "temperature": model.temperature,
            **(
                {"head_type": model.head_type, "head_config": model.head_config}
                if isinstance(model, DecisionModel) and model.decision_head is not None
                else {}
            ),
        }
        if resume:
            if not output.is_file() or not manifest.is_file():
                raise FileNotFoundError("cannot resume without the prediction journal and manifest")
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("evaluation identity differs from the saved journal")
            with output.open("r+b") as stream:
                while line := stream.readline():
                    position = stream.tell() - len(line)
                    try:
                        saved = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if stream.read(1):
                            raise ValueError("evaluation journal has a corrupt nonfinal row")
                        # An interrupted append can leave only the last row
                        # incomplete. Keep every preceding complete response.
                        stream.truncate(position)
                        break
                    row = next(rows, None)
                    if row is None or row.id != saved["id"] or row.options != saved["options"]:
                        raise ValueError("saved evaluation rows no longer match the dataset")
                    if not torch.allclose(
                        torch.tensor(row.target_distribution()),
                        torch.tensor(saved["target"]),
                        rtol=0,
                        atol=1e-7,
                    ):
                        raise ValueError("saved evaluation targets no longer match the dataset")
                    record(
                        row,
                        torch.tensor(saved["probabilities"], dtype=torch.float32),
                        torch.tensor(saved["target"], dtype=torch.float32),
                    )
        else:
            if output.exists() or manifest.exists():
                raise FileExistsError(f"evaluation journal already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    if isinstance(model, DecisionModel):
        model.backbone.eval()
        if model.decision_head is not None:
            model.decision_head.eval()
    journal = output.open("a", encoding="utf-8") if output is not None else nullcontext(None)
    batches = (
        token_budget_batches(model, rows, batch_size, max_batch_tokens)
        if max_batch_tokens is not None
        else example_batches(rows, batch_size)
    )
    with journal as stream, torch.inference_mode():
        for examples in batches:
            if isinstance(model, DecisionModel):
                predictions = (
                    (row, torch.softmax(batch.logits[i, slots].float() / model.temperature, dim=-1).cpu())
                    for subset, batch in _inference_batches(model, examples)
                    for i, (row, slots) in enumerate(zip(subset, batch.slots))
                )
            else:
                decisions = model.batch_decide(examples)
                if len(decisions) != len(examples):
                    raise ValueError("decision engine returned a different number of rows")
                predictions = (
                    (row, torch.tensor(
                        [result.probabilities[option] for option in row.options],
                        dtype=torch.float32,
                    ))
                    for row, result in zip(examples, decisions)
                )
            for row, p in predictions:
                y = torch.tensor(row.target_distribution(), dtype=torch.float32)
                record(row, p, y)
                if stream is not None:
                    stream.write(json.dumps({
                        "id": row.id, "options": row.options,
                        "probabilities": p.tolist(), "target": y.tolist(),
                    }, allow_nan=False) + "\n")
            if stream is not None:
                stream.flush()
            if progress_every and len(predicted) >= next_progress:
                print(f"evaluated {len(predicted)} rows", file=sys.stderr, flush=True)
                next_progress = len(predicted) + progress_every
    result = compute_metrics(predicted, targets, bins=bins, thresholds=thresholds)
    for group_name, named_indices in groups.items():
        result[group_name] = {
            name: compute_metrics(
                [predicted[i] for i in indices],
                [targets[i] for i in indices],
                bins=bins,
                thresholds=thresholds,
            )
            for name, indices in sorted(named_indices.items())
        }
    return result
