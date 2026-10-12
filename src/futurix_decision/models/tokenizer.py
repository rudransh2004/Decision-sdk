"""Shared prompt and slot formatting for HF and vLLM backends."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import ConfigError
from futurix_decision.models.decision_head import assign_slots, decision_token


def row_seed(*identity: Any) -> int:
    """A 64-bit seed that depends only on ``identity``, stable across processes and runs.

    Callers pass the model seed, the assignment seed, and the row's content, so
    a row's candidate order is reproducible without a shared RNG.
    """
    text = json.dumps(list(identity), sort_keys=True, ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def format_decision_prompt(
    example: DecisionInput,
    *,
    decision_tokens: int = 256,
    seed: int = 0,
    assignment_seed: int | None = None,
) -> tuple[str, list[int]]:
    if len(example.options) > decision_tokens:
        raise ConfigError("more options supplied than available decision tokens")
    state = json.dumps(example.state, ensure_ascii=False, sort_keys=True)
    question = json.dumps(example.question, ensure_ascii=False)
    slots = assign_slots(
        example.options,
        total_slots=decision_tokens,
        seed=row_seed(seed, assignment_seed, example.kind, state, question, example.options),
    )
    candidates = "\n".join(
        f"{decision_token(slot)}: {json.dumps(option, ensure_ascii=False)}"
        for option, slot in zip(example.options, slots, strict=True)
    )
    prompt = f"State: {state}\nQuestion: {question}\nCandidates:\n{candidates}\nDecision:"
    return prompt, slots


def _candidate_order(
    example: DecisionInput,
    max_options: int,
    seed: int,
    assignment_seed: int | None,
) -> tuple[str, str, list[str], list[int], list[int]]:
    if not 2 <= len(example.options) <= max_options:
        raise ConfigError("number of options must be between 2 and max_options")
    state = json.dumps(example.state, ensure_ascii=False, sort_keys=True)
    question = json.dumps(example.question, ensure_ascii=False)
    choices = [json.dumps(option, ensure_ascii=False) for option in example.options]
    order = list(range(len(choices)))
    random.Random(
        row_seed(seed, assignment_seed, example.kind, state, question, example.options)
    ).shuffle(order)
    positions = [0] * len(order)
    for position, original_index in enumerate(order):
        positions[original_index] = position
    return state, question, choices, order, positions


def format_tokenized_head_prompt(
    example: DecisionInput,
    tokenizer: Any,
    *,
    max_options: int = 256,
    seed: int = 0,
    assignment_seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize candidate segments and record endpoints without adding vocabulary rows.

    Tokenization is intentionally segment-wise and versioned: the same input IDs
    and candidate endpoints are reconstructed when a trained checkpoint reloads.
    """
    state, question, choices, order, slots = _candidate_order(
        example,
        max_options,
        seed,
        assignment_seed,
    )
    ids = tokenizer.encode(
        f"State: {state}\nQuestion: {question}\nCandidates:\n",
        add_special_tokens=False,
    )
    bos = tokenizer.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids.insert(0, bos)
    endpoints = [0] * len(choices)
    for index in order:
        text = choices[index]
        if example.candidate_descriptions and example.candidate_descriptions[index]:
            text += ": " + json.dumps(example.candidate_descriptions[index], ensure_ascii=False)
        segment = tokenizer.encode(text + "\n", add_special_tokens=False)
        if not segment:
            raise ConfigError("each candidate must encode to at least one token")
        ids.extend(segment)
        endpoints[index] = len(ids) - 1
    suffix = tokenizer.encode("Decision:", add_special_tokens=False)
    if not suffix:
        raise ConfigError("the decision query must encode to at least one token")
    ids.extend(suffix)
    return ids, slots, endpoints
