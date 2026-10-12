"""Hugging Face causal-LM backend for typed, vocabulary-logit decisions."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.errors import CheckpointError, ConfigError, MissingDependencyError
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
    choice_input,
)
from futurix_decision.inference.system_one import system_one_answers, system_one_inputs
from futurix_decision.models.batching import token_budget_batches
from futurix_decision.models.decision_config import LOADABLE, DecisionModelConfig
from futurix_decision.models.families.futurix import QWEN35
from futurix_decision.models.protocol import DecisionShortcuts
from futurix_decision.models.readouts import (
    DecisionBatch,
    LetterCodeReadout,
    PointerReadout,
    ReadoutStrategy,
    TokenReadout,
    input_device,
)
from futurix_decision.models.registry import registered_families, reject_native_checkpoint
from futurix_decision.models.sources import tokenizer_kwargs
from futurix_decision.utils.config import load_decision_config

DECISION_CONFIG = "decision_config.json"


class DecisionModel(DecisionShortcuts):
    """Wrap a decoder-only LM with a decision readout chosen by ``DecisionModelConfig``.

    ``DecisionModel(backbone, tokenizer, config)`` takes a config; keyword
    arguments replace its fields, so ``DecisionModel(lm, tokenizer,
    decision_tokens=16)`` also works. Both a plain LM's decision tokens and a
    newly attached head need training before their probabilities can be
    treated as calibrated decisions.
    """

    multimodal_backbone = False
    """Load the backbone with ``AutoModelForMultimodalLM`` (for example, JPT)."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        tokenizer: Any,
        config: DecisionModelConfig | None = None,
        *,
        decision_head: torch.nn.Module | None = None,
        **settings: Any,
    ) -> None:
        if config is None:
            config = DecisionModelConfig(**self._defaults())
        config = replace(config, **settings)
        if not self._reads(config.prompt_format):
            raise ConfigError(
                f"{type(self).__name__} cannot read prompt_format={config.prompt_format!r}; "
                "DecisionModel.from_pretrained picks the class that can"
            )
        # A Transformers causal LM. Typed Any: nn.Module's __getattr__ hides its methods.
        self.backbone: Any = backbone
        self.tokenizer = tokenizer
        self.config = config
        self.readout = self._make_readout(decision_head)
        if not config.train_backbone:
            self.backbone.requires_grad_(False)

    @classmethod
    def _defaults(cls) -> dict[str, Any]:
        """``DecisionModelConfig`` fields this class sets when the caller passes no config."""
        return {}

    @classmethod
    def _reads(cls, prompt_format: str) -> bool:
        """Whether this class implements ``prompt_format``'s readout."""
        return prompt_format in ("futurix", "bosun-v3")

    def _make_readout(self, decision_head: torch.nn.Module | None) -> ReadoutStrategy:
        if self.config.head_type == "tokens":
            if decision_head is not None:
                raise ConfigError("token readout does not take a custom decision head")
            return TokenReadout(self)
        if self.config.head_type == "letter_code":
            return LetterCodeReadout(self, decision_head)
        return PointerReadout(self, decision_head)

    @property
    def decision_head(self) -> torch.nn.Module | None:
        """The readout's trainable head, or ``None`` when it reads LM logits."""
        return self.readout.head

    @decision_head.setter
    def decision_head(self, head: torch.nn.Module | None) -> None:
        self.readout.head = head

    def to(self, device: str | torch.device | None) -> DecisionModel:
        """Move the backbone and head to ``device`` and return ``self``. ``None`` does nothing."""
        if device is not None:
            self.backbone.to(device)
            if self.decision_head is not None:
                self.decision_head.to(device)
        return self

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path, *, trainable_adapter: bool = False, **kwargs: Any
    ) -> DecisionModel:
        """Load a Futurix checkpoint, or start an untrained readout on a plain causal LM.

        Keywords named in ``LOADABLE`` set ``DecisionModelConfig`` fields; a
        saved checkpoint rejects values that change its trained readout. The
        rest go to Transformers' ``from_pretrained``. A saved ``prompt_format``
        selects the ``DecisionModel`` subclass that reads it.
        """
        path = Path(checkpoint)
        requested = {key: kwargs.pop(key) for key in LOADABLE if key in kwargs}
        hf_kwargs = kwargs
        saved = load_decision_config(checkpoint, **hf_kwargs)
        reject_native_checkpoint(checkpoint, saved)
        model_class = cls._class_for(saved.get("prompt_format", "futurix"))
        options = model_class._saved_options(saved)
        adapter = path.is_dir() and (path / "adapter_config.json").is_file()
        base_checkpoint = (
            saved.get("base_checkpoint")
            if adapter
            else str(path.resolve())
            if path.is_dir()
            else str(checkpoint)
        )
        if adapter and not base_checkpoint:
            raise CheckpointError(
                "adapter checkpoint is missing base_checkpoint in decision_config.json"
            )
        config = DecisionModelConfig.from_saved(
            saved,
            requested,
            base_checkpoint=base_checkpoint,
            source_checkpoint=saved.get("source_checkpoint", str(checkpoint)),
            source_revision=hf_kwargs.get("revision", saved.get("source_revision")),
            base_model_revision=saved.get("base_model_revision", hf_kwargs.get("revision")),
        )
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), **tokenizer_kwargs(hf_kwargs))
        backbone_kwargs = dict(hf_kwargs)
        if config.attention_mode == "noncausal_full_attention":
            if backbone_kwargs.get("attn_implementation", "sdpa") != "sdpa":
                raise ConfigError("noncausal letter-code attention requires SDPA")
            backbone_kwargs["attn_implementation"] = "sdpa"
        if adapter and saved.get("base_model_revision"):
            backbone_kwargs["revision"] = saved["base_model_revision"]
        multimodal = (
            model_class.multimodal_backbone
            or config.multimodal_loader
            or str(checkpoint) in QWEN35
            or saved.get("source_checkpoint") in QWEN35
        )
        backbone = _backbone_class(multimodal).from_pretrained(
            base_checkpoint if adapter else str(checkpoint), **backbone_kwargs
        )
        model = model_class(backbone, tokenizer, config, **options)
        if "head_type" not in saved:
            model.readout.initialize_new_head()
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise MissingDependencyError(
                    "adapter loading needs pip install futurix-decision[train]"
                ) from exc
            model.backbone = PeftModel.from_pretrained(
                model.backbone, str(checkpoint), is_trainable=trainable_adapter
            )
        if "head_type" in saved:
            model.readout.load_head(path)
        return model

    @classmethod
    def _class_for(cls, prompt_format: str) -> type[DecisionModel]:
        """``cls``, or the registered subclass of it that reads ``prompt_format``."""
        if cls._reads(prompt_format):
            return cls
        for adapter in registered_families():
            model_class = adapter.model_class()
            if (
                model_class is not None
                and issubclass(model_class, cls)
                and model_class._reads(prompt_format)
            ):
                return model_class
        raise ConfigError(f"{cls.__name__} cannot read prompt_format={prompt_format!r}")

    @classmethod
    def _saved_options(cls, saved: Mapping[str, Any]) -> dict[str, Any]:
        """Constructor keywords a subclass restores from ``decision_config.json``."""
        return {}

    def _metadata(self) -> dict[str, Any]:
        """The ``decision_config.json`` contents, except an adapter's ``base_checkpoint``."""
        return self.config.metadata()

    @classmethod
    def from_bosun(
        cls, checkpoint: str | Path = "Hanno-Labs/bosun-v3.1-1.7b", **kwargs: Any
    ) -> DecisionModel:
        """Load a pinned Bosun v3.1 release or a trusted local copy; see ``BosunAdapter.load``."""
        from futurix_decision.models.families.bosun import BosunAdapter
        from futurix_decision.models.sources import checkpoint_source

        source, local = checkpoint_source(checkpoint)
        adapter = BosunAdapter()
        if local:
            adapter.requires(Path(source))
        return adapter.load(source, **kwargs)

    @classmethod
    def from_decider(
        cls, checkpoint: str | Path = "Mapika/decider-4b", **hf_kwargs: Any
    ) -> DecisionModel:
        """Import the released 4B v2.1 letter-logit contract, without adding tokens."""
        from futurix_decision.models.decider import DeciderDecisionModel

        return DeciderDecisionModel.from_native(checkpoint, **hf_kwargs)

    @classmethod
    def from_jpt(cls, checkpoint: str | Path = "kirp/jpt-4b", **hf_kwargs: Any) -> DecisionModel:
        """Load JPT's full multimodal backbone for SDK text-decision scoring."""
        from futurix_decision.models.jpt import JPTDecisionModel

        return JPTDecisionModel.from_native(checkpoint, **hf_kwargs)

    @classmethod
    def from_eikos(
        cls, checkpoint: str | Path = "caiovicentino1/Eikos-4B", **hf_kwargs: Any
    ) -> DecisionModel:
        """Import Eikos' text SemIf prompt, letter rows and T=1 calibration."""
        from futurix_decision.models.eikos import EikosDecisionModel

        return EikosDecisionModel.from_native(checkpoint, **hf_kwargs)

    def save_pretrained(self, path: str | Path, *, safe_serialization: bool = True) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        adapter_base = None
        base_checkpoint = self.config.base_checkpoint
        if getattr(self.backbone, "peft_config", None):
            if getattr(self.backbone, "is_loaded_in_4bit", False):
                if not base_checkpoint:
                    raise ConfigError("a 4-bit adapter needs a recoverable base_checkpoint")
                if Path(base_checkpoint).is_dir() and Path(base_checkpoint).resolve() == (
                    path.resolve()
                ):
                    raise ConfigError("QLoRA adapter output cannot overwrite its base checkpoint")
                adapter_base = base_checkpoint
            else:
                # Merge unquantized LoRA and trainable decision-token rows into
                # a standalone Hugging Face model (no adapter dependency).
                self.backbone = self.backbone.merge_and_unload()
        self.backbone.save_pretrained(path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(path)
        self._save_decision_metadata(path, adapter_base=adapter_base)

    def _save_decision_metadata(self, path: str | Path, *, adapter_base: str | None = None) -> None:
        """Write head weights and metadata alongside an already saved backbone."""
        path = Path(path)
        self.readout.save_head(path)
        metadata = {
            **self._metadata(),
            **({"base_checkpoint": adapter_base} if adapter_base else {}),
        }
        (path / DECISION_CONFIG).write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        return self._infer([example], [confidence_threshold])[0]

    def batch_decide(
        self,
        examples: Sequence[DecisionInput],
        *,
        confidence_threshold: float | None = None,
        max_batch_tokens: int | None = None,
    ) -> list[DecisionResult]:
        """Decide ``examples`` in one forward pass, or in token-budgeted chunks.

        With ``max_batch_tokens``, rows go through the backbone in order, in
        chunks whose padded size (longest prompt times rows) stays within the
        budget, so peak memory no longer grows with ``len(examples)``.
        """
        if max_batch_tokens is None:
            return self._infer(examples, [confidence_threshold] * len(examples))
        results: list[DecisionResult] = []
        for chunk in token_budget_batches(
            examples, self.readout.prompt_tokens, max_batch_tokens=max_batch_tokens
        ):
            results.extend(self._infer(chunk, [confidence_threshold] * len(chunk)))
        return results

    def system_one(
        self, state: State, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Answer typed System One questions with this model's trained readout."""
        examples = system_one_inputs(state, questions)
        return system_one_answers(questions, examples, self.batch_decide(examples))

    def forward_decisions(
        self, examples: Sequence[DecisionInput], *, assignment_seed: int | None = None
    ) -> DecisionBatch:
        """Single LM forward for a batch; preserve gradients for all readouts.

        Change assignment_seed between training steps/epochs to randomize the
        candidate-to-slot mapping reproducibly. Inference uses a stable mapping.
        """
        if not examples:
            raise ConfigError("examples must not be empty")
        return self.readout.forward(examples, assignment_seed)

    def _infer(
        self, examples: Sequence[DecisionInput], thresholds: Sequence[float | None]
    ) -> list[DecisionResult]:
        if not examples:
            return []
        for threshold in thresholds:
            check_confidence_threshold(threshold)
        device = input_device(self.backbone)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        self.backbone.eval()
        if self.decision_head is not None:
            self.decision_head.eval()
        with torch.inference_mode():
            batch = self.forward_decisions(examples)
            logits = batch.logits.float()
            if batch.null_slot is not None:
                # A result reports probabilities over the options, so leave abstention out.
                null = torch.tensor([batch.null_slot], device=logits.device)
                logits = logits.index_fill(1, null, float("-inf"))
            probabilities = torch.softmax(logits / self.config.temperature, dim=-1)
            probabilities = probabilities.cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # Report the batch's wall time per row, as every backend does.
        latency_ms = (time.perf_counter() - started) * 1000 / len(examples)
        return [
            DecisionResult.from_probabilities(
                example,
                [probabilities[row, slot].item() for slot in slots],
                latency_ms=latency_ms,
                confidence_threshold=threshold,
            )
            for row, (example, slots, threshold) in enumerate(
                zip(examples, batch.slots, thresholds, strict=True)
            )
        ]

    def close(self) -> None:
        """Do nothing: an in-process model holds no connections. Part of ``DecisionBackend``."""

    def batch_choice(
        self,
        examples: Sequence[Mapping[str, Any] | DecisionInput],
        *,
        confidence_threshold: float | None = None,
    ) -> list[DecisionResult]:
        """Run all examples in one padded transformer forward pass."""
        inputs: list[DecisionInput] = []
        thresholds: list[float | None] = []
        for row in examples:
            if isinstance(row, DecisionInput):
                if row.kind != "choice":
                    raise ConfigError("batch_choice accepts only choice inputs")
                inputs.append(row)
                thresholds.append(confidence_threshold)
            else:
                if row.get("kind", "choice") != "choice":
                    raise ConfigError("batch_choice accepts only choice inputs")
                choices = row.get("choices", row.get("options", []))
                question = row.get("question", "Which option should be selected?")
                inputs.append(choice_input(row["state"], choices, question))
                thresholds.append(row.get("confidence_threshold", confidence_threshold))
        return self._infer(inputs, thresholds)


def _backbone_class(multimodal: bool) -> Any:
    if not multimodal:
        return AutoModelForCausalLM
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise MissingDependencyError(
            "JPT and Qwen3.5-9B require Transformers 5.17+; "
            "install futurix-decision[jpt] or futurix-decision[qwen3.5]"
        ) from exc
    return AutoModelForMultimodalLM
