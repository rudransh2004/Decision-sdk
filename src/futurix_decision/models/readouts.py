"""How a ``DecisionModel`` reads per-option logits from one backbone forward pass.

``DecisionModel`` builds one strategy from its config: ``TokenReadout`` for
Futurix and Bosun decision tokens, ``PointerReadout`` for candidate-aware heads
over candidate span endpoints, and ``LetterCodeReadout`` for a fixed table of
answer codes. ``LabelReadoutModel`` supplies its own for released letter heads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import CheckpointError, ConfigError, MissingDependencyError
from futurix_decision.models.bosun import BOSUN_CONVERSION_CHAT_TEMPLATE, format_bosun_prompt
from futurix_decision.models.custom_head import HEAD_WEIGHTS, build_decision_head
from futurix_decision.models.decision_head import decision_token, mask_unavailable_slots
from futurix_decision.models.letter_code import (
    LetterCodeHead,
    answer_codes,
    enable_qwen_noncausal_attention,
    format_letter_code_prompt,
)
from futurix_decision.models.tokenizer import (
    format_decision_prompt,
    format_tokenized_head_prompt,
)

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel

Encoded = dict[str, torch.Tensor]
_Head = TypeVar("_Head", bound=torch.nn.Module)


@dataclass
class DecisionBatch:
    """Differentiable masked logits and candidate-to-output assignments."""

    logits: torch.Tensor  # [batch, output_width]; unavailable positions are -inf
    slots: list[list[int]]  # slots[row][original_candidate_index]
    null_slot: int | None = None

    def assigned_logits(self, row: int) -> torch.Tensor:
        """Candidate-order logits, with the learned abstain logit last if enabled."""
        indices = self.slots[row] + ([self.null_slot] if self.null_slot is not None else [])
        return self.logits[row, indices]


def set_padding(tokenizer: Any, side: str, owner: str = "tokenizer") -> None:
    """Pad batches on ``side``, falling back to EOS when the tokenizer has no pad token."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise CheckpointError(f"{owner} needs a pad or EOS token for batch inference")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = side


def with_position_ids(encoded: Mapping[str, torch.Tensor]) -> Encoded:
    """``encoded`` plus position IDs that count from each row's first unpadded token.

    Backbones number positions from the start of the padded row unless told
    otherwise, so a left-padded row would shift by however many pad tokens
    its longest batch neighbour adds. On absolute-position models that
    changes the row's logits, making a decision depend on its batch.
    """
    positions = (encoded["attention_mask"].long().cumsum(dim=-1) - 1).clamp(min=0)
    return {**encoded, "position_ids": positions}


def add_tokens(model: DecisionModel, tokens: list[str], *, what: str) -> list[int]:
    """Add ``tokens`` as special tokens and return their IDs.

    Rows the vocabulary lacked get fresh, seeded embeddings; rows a checkpoint
    already trained are left alone.
    """
    tokenizer, backbone = model.tokenizer, model.backbone
    vocabulary = tokenizer.get_vocab()
    added = {token for token in tokens if token not in vocabulary}
    existing = getattr(tokenizer, "additional_special_tokens", None)
    if existing is None:  # Transformers 5 uses extra_special_tokens.
        existing = getattr(tokenizer, "extra_special_tokens", [])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(dict.fromkeys(existing + tokens))}
    )
    if backbone.get_input_embeddings().num_embeddings != len(tokenizer):
        if getattr(backbone, "is_loaded_in_4bit", False):
            raise ConfigError(
                f"cannot resize a 4-bit checkpoint; add the {what} rows and "
                "save the unquantized base before loading it with BitsAndBytesConfig"
            )
        # Independently initialize new rows; mean/covariance resizing can
        # produce near-identical logits with small vocabularies.
        backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    if any(
        tokenizer.encode(token, add_special_tokens=False) != [id_]
        for token, id_ in zip(tokens, ids, strict=True)
    ):
        raise ConfigError(f"each {what} must encode to exactly one token ID")
    new_ids = [id_ for token, id_ in zip(tokens, ids, strict=True) if token in added]
    if new_ids:
        # Some LMs (including Qwen3) ship more embedding rows than the tokenizer
        # has tokens. New tokens consume those padded rows, which resizing does
        # not initialize, so seed the input and output rows explicitly.
        input_weight = backbone.get_input_embeddings().weight
        output_embedding = backbone.get_output_embeddings()
        output_weight = output_embedding.weight if output_embedding is not None else None
        std = float(getattr(backbone.config, "initializer_range", 0.02))
        generator = torch.Generator(device="cpu").manual_seed(model.config.seed)
        with torch.no_grad():
            for weight in (
                [input_weight]
                if output_weight is None or output_weight is input_weight
                else [input_weight, output_weight]
            ):
                fresh = torch.randn(
                    (len(new_ids), weight.shape[1]), generator=generator, dtype=torch.float32
                ).mul_(std)
                weight.index_copy_(
                    0,
                    torch.tensor(new_ids, dtype=torch.long, device=weight.device),
                    fresh.to(device=weight.device, dtype=weight.dtype),
                )
    return ids


def input_device(backbone: Any) -> torch.device:
    """Where a (possibly device-dispatched) backbone takes its input IDs."""
    device = backbone.get_input_embeddings().weight.device
    if device.type != "meta":
        return device
    for target in getattr(backbone, "hf_device_map", {}).values():
        if isinstance(target, int):
            return torch.device("cuda", target)
        if isinstance(target, str) and target.startswith("cuda"):
            return torch.device(target)
    return torch.device("cpu")


def reject_images(examples: Sequence[DecisionInput]) -> None:
    if any(row.images for row in examples):
        raise ConfigError("Futurix's text-only decision head does not accept images")


class ReadoutStrategy(ABC):
    """Prompts a ``DecisionModel``'s backbone and turns its outputs into option logits.

    A strategy reads the model's current ``backbone``, ``tokenizer``, and
    ``config`` at call time, so trainers may wrap or replace the backbone.
    """

    head: torch.nn.Module | None = None
    """A trainable readout on top of the backbone, or ``None`` if it reads LM logits."""

    def __init__(self, model: DecisionModel) -> None:
        self.model = model
        # Vocabulary rows this readout scores or marks; LoRA training keeps them trainable.
        self.token_ids: list[int] = []

    @abstractmethod
    def forward(
        self, examples: Sequence[DecisionInput], assignment_seed: int | None
    ) -> DecisionBatch:
        """One differentiable forward pass over a non-empty batch."""

    @abstractmethod
    def prompt_tokens(self, example: DecisionInput) -> int:
        """The longest token row ``example`` adds to a batch, for token-budget batching."""

    def initialize_new_head(self) -> None:
        """Prepare a head no checkpoint has trained yet. Only some readouts need this."""
        return None

    def load_head(self, path: Path) -> None:
        """Load the head weights saved with a checkpoint at ``path``, if this readout has a head."""
        return None

    def save_head(self, path: Path) -> None:
        """Save this readout's head weights into ``path``, if it has a head."""
        return None

    def _on_device(self, encoded: Mapping[str, torch.Tensor]) -> Encoded:
        limit = self.model.config.max_sequence_length
        if encoded["input_ids"].shape[1] > limit:
            raise ConfigError(
                f"prompt exceeds max_sequence_length={limit}; "
                "increase the limit instead of silently truncating candidates"
            )
        device = input_device(self.model.backbone)
        return {key: value.to(device) for key, value in with_position_ids(encoded).items()}


class PromptReadout(ReadoutStrategy):
    """A readout that renders each example as one text prompt."""

    @abstractmethod
    def prompt(self, example: DecisionInput, assignment_seed: int | None) -> tuple[str, list[int]]:
        """The prompt text and each original option's output slot."""

    def prompt_tokens(self, example: DecisionInput) -> int:
        prompt = self.prompt(example, None)[0]
        return len(self.model.tokenizer.encode(prompt, add_special_tokens=True))

    def _encode(
        self, examples: Sequence[DecisionInput], assignment_seed: int | None
    ) -> tuple[Encoded, list[list[int]]]:
        reject_images(examples)
        prompts = [self.prompt(row, assignment_seed) for row in examples]
        encoded = self.model.tokenizer(
            [item[0] for item in prompts],
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return self._on_device(encoded), [item[1] for item in prompts]


class TokenReadout(PromptReadout):
    """Score added decision-token rows in the LM's next-token logits (Futurix and Bosun v3)."""

    def __init__(self, model: DecisionModel) -> None:
        super().__init__(model)
        tokenizer, config = model.tokenizer, model.config
        self.bosun = config.prompt_format == "bosun-v3"
        if self.bosun and not getattr(tokenizer, "chat_template", None):
            if not config.bosun_template_fallback:
                raise ConfigError("Bosun prompt requires a tokenizer chat template")
            tokenizer.chat_template = BOSUN_CONVERSION_CHAT_TEMPLATE
        tokens = [decision_token(i) for i in range(config.decision_tokens)]
        self.token_ids = add_tokens(model, tokens, what="decision token")
        # Bosun reads the last position of a left-padded batch, as its release does.
        set_padding(tokenizer, "left" if self.bosun else "right")

    def prompt(self, example: DecisionInput, assignment_seed: int | None) -> tuple[str, list[int]]:
        config = self.model.config
        if self.bosun:
            return format_bosun_prompt(
                example,
                self.model.tokenizer,
                decision_tokens=config.decision_tokens,
                seed=config.seed,
                assignment_seed=assignment_seed,
                prompt_schema=config.prompt_schema,
                learn_abstain=config.learn_abstain,
            )
        return format_decision_prompt(
            example,
            decision_tokens=config.decision_tokens,
            seed=config.seed,
            assignment_seed=assignment_seed,
        )

    def forward(
        self, examples: Sequence[DecisionInput], assignment_seed: int | None
    ) -> DecisionBatch:
        encoded, slots = self._encode(examples, assignment_seed)
        device = encoded["input_ids"].device
        output = self.model.backbone(**encoded, **({"logits_to_keep": 1} if self.bosun else {}))
        rows = torch.arange(len(examples), device=device)
        last_positions = (
            torch.zeros(len(examples), dtype=torch.long, device=device)
            if output.logits.shape[1] == 1
            else (
                torch.full((len(examples),), output.logits.shape[1] - 1, device=device)
                if self.model.tokenizer.padding_side == "left"
                else encoded["attention_mask"].sum(dim=1) - 1
            )
        )
        token_ids = torch.tensor(self.token_ids, device=device)
        logits = output.logits[rows, last_positions].index_select(dim=-1, index=token_ids)
        return DecisionBatch(mask_unavailable_slots(logits, slots), slots)


class HeadReadout(ReadoutStrategy):
    """A readout with a registered ``torch.nn.Module`` head over backbone hidden states."""

    head: torch.nn.Module

    def _hidden_size(self) -> int:
        config = self.model.backbone.config
        hidden_size = getattr(getattr(config, "text_config", config), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise CheckpointError("causal LM config must expose a positive text hidden_size")
        return hidden_size

    def _attach(self, head: _Head) -> _Head:
        """Place ``head`` with the input embeddings unless those are not materialized yet."""
        embedding_device = self.model.backbone.get_input_embeddings().weight.device
        if embedding_device.type != "meta":
            head.to(embedding_device)
        return head

    def _head_on(self, device: torch.device) -> torch.nn.Module:
        """The head on ``device``. A head in training must already be there."""
        if next(self.head.parameters()).device != device:
            if self.model.backbone.training:
                raise ConfigError(
                    "training a device-dispatched decision head needs a single execution device"
                )
            self.head.to(device)
        return self.head

    def _base_hidden_states(self, encoded: Encoded) -> torch.Tensor:
        # Call the transformer's base directly: a full vocabulary projection
        # would allocate [batch, sequence, vocabulary] logits on many LMs.
        source = self.model.backbone
        if getattr(source, "peft_config", None):
            source = source.get_base_model()
        base = getattr(source, "base_model", None)
        if not isinstance(base, torch.nn.Module) or base is source:
            raise CheckpointError(
                "the causal LM must expose a base transformer for pointer decisions"
            )
        output = base(**encoded, return_dict=True, use_cache=False)
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("the causal LM base must return [batch, sequence, hidden] states")
        return hidden

    def load_head(self, path: Path) -> None:
        if not path.is_dir() or not (path / HEAD_WEIGHTS).is_file():
            raise CheckpointError(f"custom head checkpoint is missing {HEAD_WEIGHTS}")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise MissingDependencyError("custom head loading requires safetensors") from exc
        self.head.load_state_dict(load_file(str(path / HEAD_WEIGHTS)), strict=True)

    def save_head(self, path: Path) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise MissingDependencyError("custom head saving requires safetensors") from exc
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.head.state_dict().items()
        }
        save_file(state, str(path / HEAD_WEIGHTS))


class PointerReadout(HeadReadout):
    """Score each option's last token (``span-v1``) against the query's final hidden state."""

    def __init__(self, model: DecisionModel, head: torch.nn.Module | None) -> None:
        super().__init__(model)
        set_padding(model.tokenizer, "right")
        config = model.config
        settings = dict(config.head_config)
        if config.learn_abstain:
            settings["abstain"] = True
        registered = build_decision_head(config.head_type, self._hidden_size(), settings)
        if head is not None and type(head) is not type(registered):
            raise TypeError("decision_head must match the registered head factory")
        self.head = self._attach(head if head is not None else registered)

    def _prepare(
        self, example: DecisionInput, assignment_seed: int | None
    ) -> tuple[list[int], list[int], list[int]]:
        """Token IDs, each option's slot, and each option's last-token position."""
        config = self.model.config
        return format_tokenized_head_prompt(
            example,
            self.model.tokenizer,
            max_options=config.decision_tokens,
            seed=config.seed,
            assignment_seed=assignment_seed,
        )

    def prompt_tokens(self, example: DecisionInput) -> int:
        return len(self._prepare(example, None)[0])

    def forward(
        self, examples: Sequence[DecisionInput], assignment_seed: int | None
    ) -> DecisionBatch:
        reject_images(examples)
        prepared = [self._prepare(row, assignment_seed) for row in examples]
        pad = self.model.tokenizer.pad_token_id
        width = max(len(item[0]) for item in prepared)
        input_ids = torch.full((len(examples), width), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, (tokens, _, _) in enumerate(prepared):
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row, : len(tokens)] = 1
        encoded = self._on_device({"input_ids": input_ids, "attention_mask": attention_mask})
        hidden = self._base_hidden_states(encoded)
        head = self._head_on(hidden.device)
        sizes = [len(row.options) for row in examples]
        candidates = hidden.new_zeros((len(examples), max(sizes), hidden.shape[-1]))
        available = torch.zeros((len(examples), max(sizes)), dtype=torch.bool, device=hidden.device)
        for row, (count, (_, _, endpoints)) in enumerate(zip(sizes, prepared, strict=True)):
            candidates[row, :count] = hidden[row, torch.tensor(endpoints, device=hidden.device)]
            available[row, :count] = True
        last = encoded["attention_mask"].sum(dim=1).to(hidden.device) - 1
        state = hidden[torch.arange(len(examples), device=hidden.device), last]
        logits = head(state, candidates, available)
        # A head trained to abstain appends its abstain logit after the widest row's options.
        null_slot = max(sizes) if self.model.config.learn_abstain else None
        outputs = (
            available
            if null_slot is None
            else torch.cat((available, available.new_ones((len(examples), 1))), dim=1)
        )
        if logits.shape != outputs.shape:
            raise ValueError(
                "decision head must return one logit per supplied option"
                + (" followed by one abstain logit" if null_slot is not None else "")
            )
        if not torch.isfinite(logits[outputs]).all():
            raise ValueError("decision head produced non-finite logits")
        return DecisionBatch(
            logits.masked_fill(~outputs, float("-inf")), [i[1] for i in prepared], null_slot
        )


class LetterCodeReadout(HeadReadout, PromptReadout):
    """Project the final hidden state onto a fixed table of single-token answer codes."""

    head: LetterCodeHead

    def __init__(self, model: DecisionModel, head: torch.nn.Module | None) -> None:
        super().__init__(model)
        config = model.config
        codes, token_ids = answer_codes(model.tokenizer, config.decision_tokens)
        expected = {"codes": codes, "token_ids": token_ids}
        if config.head_config and config.head_config != expected:
            raise CheckpointError("saved letter-code answers differ from this tokenizer")
        config.head_config = expected
        # Left padding puts every row's complete prompt before the final position.
        set_padding(model.tokenizer, "left")
        registered = LetterCodeHead(self._hidden_size(), codes, token_ids)
        if head is None:
            head = registered
        elif not isinstance(head, LetterCodeHead):
            raise TypeError("decision_head must match the registered head factory")
        self.head = self._attach(head)
        if config.attention_mode == "noncausal_full_attention":
            enable_qwen_noncausal_attention(model.backbone)

    def initialize_new_head(self) -> None:
        self.head.initialize_from_lm_head(
            self.model.backbone.get_output_embeddings(), self.model.config.head_config["token_ids"]
        )

    def prompt(self, example: DecisionInput, assignment_seed: int | None) -> tuple[str, list[int]]:
        return format_letter_code_prompt(
            example, self.model.tokenizer, self.model.config.head_config["codes"]
        )

    def forward(
        self, examples: Sequence[DecisionInput], assignment_seed: int | None
    ) -> DecisionBatch:
        encoded, slots = self._encode(examples, assignment_seed)
        hidden = self._base_hidden_states(encoded)
        # The final hidden state sees the complete state, question, and code table.
        scores = self._head_on(hidden.device)(hidden[:, -1])
        available = (
            torch.arange(self.model.config.decision_tokens, device=scores.device)[None, :]
            < torch.tensor([len(row.options) for row in examples], device=scores.device)[:, None]
        )
        if not torch.isfinite(scores[available]).all():
            raise ValueError("letter-code readout produced non-finite candidate logits")
        return DecisionBatch(scores.masked_fill(~available, float("-inf")), slots)
