"""Bosun v3 stable-slot prompt contract (Apache-2.0; see upstream NOTICE).

This is deliberately separate from the SDK's generic, randomly sampled
decision slots. Existing Bosun weights must see the prompt they were trained
on; otherwise even a successful checkpoint load gives meaningless scores.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import ConfigError
from futurix_decision.models.decision_head import decision_token

BOSUN_PROMPT_SCHEMA = "bosun-decision-prompt-v3-stable-slots"
BOSUN_SYSTEM_PROMPT = (
    "Choose exactly one supplied decision token. Return only that token. Do not explain the answer."
)
BOSUN_ABSTAIN_SYSTEM_PROMPT = (
    "Choose exactly one candidate decision token, or <|decision_255|> if none applies. "
    "Return only that token. Do not explain the answer."
)
# For a plain causal LM with no chat template, keep the same Bosun-style
# system/user/assistant and JSON structure. These role strings are ordinary
# text, not a claim of parity with the publisher's Qwen chat tokenization.
BOSUN_CONVERSION_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "[{{ message['role'] }}]\n{{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}[assistant]\n{% endif %}"
)


def format_bosun_prompt(
    example: DecisionInput,
    tokenizer: Any,
    *,
    decision_tokens: int = 256,
    seed: int = 0,
    assignment_seed: int | None = None,
    prompt_schema: str = BOSUN_PROMPT_SCHEMA,
    learn_abstain: bool = False,
) -> tuple[str, list[int]]:
    """Render Bosun's ChatML/JSON request and return slots in candidate order."""
    count = len(example.options)
    if not 2 <= count <= min(255, decision_tokens - 1):
        raise ConfigError("Bosun reserves slot 255 for null and supports 2 to 255 choices")
    row_id = str(getattr(example, "id", "0"))
    permutation = list(range(count))
    order_seed = seed if assignment_seed is None else assignment_seed
    digest = hashlib.sha256(f"{order_seed}:{row_id}:candidate-order".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(permutation)

    slots = [0] * count
    criteria: list[dict[str, str | int]] = []
    for slot, candidate_index in enumerate(permutation):
        slots[candidate_index] = slot
        criteria.append(
            {
                "t": decision_token(slot),
                "o": candidate_index,
                "n": (
                    example.candidate_labels[candidate_index]
                    if example.candidate_labels is not None
                    else example.options[candidate_index]
                ),
                "d": (
                    example.candidate_descriptions[candidate_index]
                    if example.candidate_descriptions is not None
                    else ""
                ),
            }
        )

    description_fields: list[str] | None = None
    parsed_descriptions: list[dict[str, Any]] = []
    for criterion in criteria:
        try:
            parsed = json.loads(str(criterion["d"]))
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
            description_fields = None
            break
        keys = sorted(parsed)
        if description_fields is not None and keys != description_fields:
            description_fields = None
            break
        description_fields = keys
        parsed_descriptions.append(parsed)
    if description_fields is not None and len(parsed_descriptions) == count:
        fields = ["t", "o", "n", *description_fields]
        rows = [
            [
                criterion["t"],
                criterion["o"],
                criterion["n"],
                *(parsed[key] for key in description_fields),
            ]
            for criterion, parsed in zip(criteria, parsed_descriptions, strict=True)
        ]
    else:
        fields = ["t", "o", "n", "d"]
        rows = [[criterion[key] for key in fields] for criterion in criteria]

    content = json.dumps(
        {
            "schema": prompt_schema,
            "state": example.state,
            "question": {
                "instructions": example.question,
                "type": example.kind,
                "criteria_fields": fields,
                "criteria": rows,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": BOSUN_ABSTAIN_SYSTEM_PROMPT if learn_abstain else BOSUN_SYSTEM_PROMPT,
            },
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return str(prompt), slots
