"""Shared base for released checkpoints that answer through existing option-label tokens.

Decider, JPT, and Eikos keep their released LM weights. Each prints the options
under labels such as ``A``, ``B``, and ``C``, then reads the decision from the
next-token logits of those labels. No decision tokens or heads are added.
A subclass supplies only its prompt (``_rows``), its label token IDs, and the
order in which it presents noul options.
"""

from __future__ import annotations

import random
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.decision_model import DecisionModel
from futurix_decision.models.readouts import (
    DecisionBatch,
    ReadoutStrategy,
    set_padding,
    with_position_ids,
)
from futurix_decision.models.sources import tokenizer_kwargs
from futurix_decision.models.tokenizer import row_seed

LabelTable = tuple[list[str], list[int]]
_LABEL_TABLES: weakref.WeakKeyDictionary[Any, dict[str, LabelTable]] = weakref.WeakKeyDictionary()
_M = TypeVar("_M", bound="LabelReadoutModel")


@dataclass(frozen=True)
class LabelRow:
    """One tokenized prompt and the label token IDs to read, in presented order."""

    input_ids: list[int]
    label_ids: list[int]


def cached_label_table(
    tokenizer: Any, family: str, width: int, build: Callable[[Any, int], LabelTable]
) -> LabelTable:
    """``build(tokenizer, width)``, cached per tokenizer and family; a wider table serves narrower
    requests."""
    tables = _LABEL_TABLES.setdefault(tokenizer, {})
    cached = tables.get(family)
    if cached is not None and len(cached[1]) >= width:
        return cached[0][:width], cached[1][:width]
    tables[family] = build(tokenizer, width)
    return tables[family]


def presented_order(example: DecisionInput, noul_order: tuple[str, str], family: str) -> list[int]:
    """Original option indices in the order the prompt lists them, before any permutation."""
    if example.kind != "noul":
        return list(range(len(example.options)))
    if set(example.options) != {"false", "true"}:
        raise ConfigError(f"{family} noul requires literal true/false options")
    return [example.options.index(option) for option in noul_order]


def noul_description(example: DecisionInput, index: int, default: str | None) -> str | None:
    """A noul option's description, else its label, else ``default``."""
    return (example.candidate_descriptions[index] if example.candidate_descriptions else None) or (
        example.candidate_labels[index] if example.candidate_labels else default
    )


def score_description(example: DecisionInput, index: int) -> str:
    """A score level's description, else its label, else the level itself."""
    if example.candidate_descriptions and example.candidate_descriptions[index]:
        return example.candidate_descriptions[index]
    return example.candidate_labels[index] if example.candidate_labels else example.options[index]


def choice_description(example: DecisionInput, index: int) -> str | None:
    """A choice's label and description joined by an em dash, or ``None`` if it has neither."""
    parts = []
    if example.candidate_labels:
        parts.append(example.candidate_labels[index])
    if example.candidate_descriptions and example.candidate_descriptions[index]:
        parts.append(example.candidate_descriptions[index])
    return " — ".join(parts) if parts else None


class LabelReadoutModel(DecisionModel):
    """A ``DecisionModel`` that scores a release's own option-label logits.

    Subclasses set the class attributes and implement ``_label_token_ids`` and
    ``_rows``. ``_presented_logits`` defaults to the single row's label logits.
    """

    release_format: ClassVar[str]
    """The ``prompt_format`` value saved with exports; no other format is accepted."""

    family: ClassVar[str]
    """The release name used in error messages."""

    noul_order: ClassVar[tuple[str, str]]
    """The order the prompt lists noul options in, matching the release's training prompt."""

    default_width: ClassVar[int | None] = 255
    """Option labels when the caller sets no ``decision_tokens``. ``None`` means the
    release metadata sets the width."""

    native_options: ClassVar[tuple[str, ...]] = ()
    """Extra ``from_native`` keywords a training config may set."""

    seeded_kinds: ClassVar[frozenset[str]] = frozenset({"choice", "noul", "score"})
    """Decision kinds whose options ``assignment_seed`` permutes during training."""

    @classmethod
    def _defaults(cls) -> dict[str, Any]:
        defaults: dict[str, Any] = {"prompt_format": cls.release_format}
        if cls.default_width is not None:
            defaults["decision_tokens"] = cls.default_width
        return defaults

    @classmethod
    def _reads(cls, prompt_format: str) -> bool:
        return prompt_format == cls.release_format

    def _make_readout(self, decision_head: torch.nn.Module | None) -> ReadoutStrategy:
        if decision_head is not None:
            raise ConfigError(f"{self.family} requires its released prompt and letter readout")
        return LabelReadout(self)

    @classmethod
    def from_native(cls, checkpoint: str | Path, **kwargs: Any) -> LabelReadoutModel:
        """Load the publisher's release, checking that it uses this readout contract."""
        raise NotImplementedError

    def _label_token_ids(self) -> list[int]:
        """Token IDs of the first ``decision_tokens`` option labels."""
        raise NotImplementedError

    def _rows(self, example: DecisionInput, order: list[int]) -> list[LabelRow]:
        """Tokenize ``example`` with its options listed in ``order``."""
        raise NotImplementedError

    def _presented_logits(
        self, example: DecisionInput, selected: list[torch.Tensor]
    ) -> torch.Tensor:
        """One logit per option, in presented order, from each row's label logits."""
        return selected[0]

    @classmethod
    def _load_release(
        cls: type[_M],
        checkpoint: str | Path,
        hf_kwargs: Mapping[str, Any],
        *,
        backbone_class: Any = AutoModelForCausalLM,
        **model_kwargs: Any,
    ) -> _M:
        """Load the release's tokenizer and backbone and record where they came from."""
        source = str(checkpoint)
        revision = hf_kwargs.get("revision")
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs(hf_kwargs))
        backbone = backbone_class.from_pretrained(source, **hf_kwargs)
        return cls(
            backbone,
            tokenizer,
            base_checkpoint=source,
            base_model_revision=revision,
            source_checkpoint=source,
            source_revision=revision,
            **model_kwargs,
        )

    def _option_order(self, example: DecisionInput, assignment_seed: int | None) -> list[int]:
        order = presented_order(example, self.noul_order, self.family)
        if assignment_seed is not None and example.kind in self.seeded_kinds:
            random.Random(self._row_seed(example, assignment_seed)).shuffle(order)
        return order

    def _row_seed(self, example: DecisionInput, assignment_seed: int) -> int:
        return row_seed(
            self.config.seed,
            assignment_seed,
            example.kind,
            example.state,
            example.question,
            example.options,
        )


class LabelReadout(ReadoutStrategy):
    """Score a release's existing option-label rows in the next-token logits.

    The prompt, label IDs, option order, and per-row combination come from the
    ``LabelReadoutModel`` subclass. Batches are left-padded.
    """

    model: LabelReadoutModel

    def __init__(self, model: LabelReadoutModel) -> None:
        super().__init__(model)
        set_padding(model.tokenizer, "left", f"{model.family} tokenizer")
        ids = model._label_token_ids()
        embedding = model.backbone.get_input_embeddings()
        if max(ids) >= embedding.num_embeddings:
            raise CheckpointError(
                f"{model.family} tokenizer has label tokens outside the backbone vocabulary"
            )
        self.token_ids = ids

    def prompt_tokens(self, example: DecisionInput) -> int:
        order = self.model._option_order(example, None)
        return max(len(row.input_ids) for row in self.model._rows(example, order))

    def _next_token_logits(self, rows: Sequence[list[int]]) -> torch.Tensor:
        """Left-pad ``rows`` into one batch and return each row's next-token logits."""
        longest = max(len(tokens) for tokens in rows)
        limit = self.model.config.max_sequence_length
        if longest > limit:
            raise ConfigError(
                f"prompt exceeds max_sequence_length={limit}; "
                "increase the limit instead of silently truncating candidates"
            )
        backbone = self.model.backbone
        device = next(backbone.parameters()).device
        ids = torch.full(
            (len(rows), longest), self.model.tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        mask = torch.zeros_like(ids)
        for index, tokens in enumerate(rows):
            ids[index, -len(tokens) :] = torch.tensor(tokens, device=device)
            mask[index, -len(tokens) :] = 1
        inputs = with_position_ids({"input_ids": ids, "attention_mask": mask})
        output = backbone(**inputs, logits_to_keep=1)
        return output.logits[:, -1, :]

    def forward(
        self, examples: Sequence[DecisionInput], assignment_seed: int | None
    ) -> DecisionBatch:
        model = self.model
        orders = [model._option_order(example, assignment_seed) for example in examples]
        groups = [
            model._rows(example, order) for example, order in zip(examples, orders, strict=True)
        ]
        logits = self._next_token_logits([row.input_ids for group in groups for row in group])
        results = logits.new_full(
            (len(examples), max(len(example.options) for example in examples)), float("-inf")
        )
        cursor = 0
        for index, (example, order, group) in enumerate(zip(examples, orders, groups, strict=True)):
            selected = [
                logits[cursor + offset]
                .index_select(0, torch.tensor(row.label_ids, device=logits.device))
                .float()
                for offset, row in enumerate(group)
            ]
            cursor += len(group)
            results[index, order] = model._presented_logits(example, selected)
        return DecisionBatch(results, [list(range(len(example.options))) for example in examples])
