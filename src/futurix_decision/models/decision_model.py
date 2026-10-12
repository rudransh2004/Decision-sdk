"""Hugging Face causal-LM backend for typed, vocabulary-logit decisions."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.inference.schemas import DecisionResult
from futurix_decision.models.bosun import (
    BOSUN_CONVERSION_CHAT_TEMPLATE,
    BOSUN_PROMPT_SCHEMA,
    format_bosun_prompt,
)
from futurix_decision.models.custom_head import (
    CANDIDATE_MARKER,
    HEAD_WEIGHTS,
    build_decision_head,
)
from futurix_decision.models.decision_head import decision_token, mask_unavailable_slots
from futurix_decision.models.families import reject_native_checkpoint
from futurix_decision.models.letter_code import (
    LetterCodeHead,
    answer_codes,
    enable_qwen_noncausal_attention,
    format_letter_code_prompt,
)
from futurix_decision.models.sources import checkpoint_source, require_checkpoint_files
from futurix_decision.models.tokenizer import (
    format_decision_prompt,
    format_head_prompt,
    format_tokenized_head_prompt,
)
from futurix_decision.utils.config import load_decision_config

DECISION_CONFIG = "decision_config.json"
BOSUN_RELEASES = {
    "Hanno-Labs/bosun-v3.1-1.7b": "1d8dc82a20e4a32ed60927a47272d6efff48eed2",
    "Hanno-Labs/bosun-v3.1-0.6b": "5e78a37c8d86860ebac9ef2d5327d6299498498e",
}


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


class DecisionModel:
    """Wrap a decoder-only LM with either decision-token or candidate-aware readout.

    Both a plain LM's decision tokens and a newly attached custom head need
    training before their probabilities can be treated as calibrated decisions.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        tokenizer: Any,
        *,
        decision_tokens: int = 256,
        temperature: float = 1.0,
        seed: int = 0,
        max_sequence_length: int = 1024,
        base_checkpoint: str | None = None,
        source_checkpoint: str | None = None,
        source_revision: str | None = None,
        base_model_revision: str | None = None,
        prompt_format: str = "futurix",
        prompt_schema: str = BOSUN_PROMPT_SCHEMA,
        head_type: str = "tokens",
        head_config: Mapping[str, Any] | None = None,
        decision_head: torch.nn.Module | None = None,
        head_prompt_version: str = "marker-v1",
        train_backbone: bool = True,
        bosun_template_fallback: bool = False,
        learn_abstain: bool = False,
        attention_mode: str = "causal",
        multimodal_loader: bool = False,
    ) -> None:
        if not 2 <= decision_tokens <= 256:
            raise ValueError("decision_tokens must be between 2 and 256")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if prompt_format not in (
            "futurix",
            "bosun-v3",
            "decider-v2.1",
            "jpt-llm2jev-chat",
            "eikos-letter-v1-semif",
        ):
            raise ValueError("unsupported decision prompt format")
        if prompt_format == "bosun-v3" and decision_tokens != 256:
            raise ValueError("Bosun v3 requires exactly 256 decision tokens")
        if head_type != "tokens" and prompt_format != "futurix":
            raise ValueError(
                "custom heads require prompt_format='futurix', not Bosun's token contract"
            )
        if prompt_format == "decider-v2.1" and type(self) is DecisionModel:
            raise ValueError(
                "decider-v2.1 requires DeciderDecisionModel to preserve its letter readout"
            )
        if prompt_format == "jpt-llm2jev-chat" and type(self) is DecisionModel:
            raise ValueError("JPT requires JPTDecisionModel to preserve its chat-letter readout")
        if prompt_format == "eikos-letter-v1-semif" and type(self) is DecisionModel:
            raise ValueError("Eikos requires EikosDecisionModel to preserve its SemIf readout")
        if head_type == "tokens" and (decision_head is not None or head_config):
            raise ValueError("token readout does not take a custom decision head")
        # Read old local exports but always write new metadata when re-exporting.
        if head_type == "pplx" and head_prompt_version == "pplx-v1":
            head_type, head_prompt_version = "letter_code", "letter-code-v1"
        if head_prompt_version not in ("marker-v1", "span-v1", "letter-code-v1"):
            raise ValueError("unsupported candidate prompt version")
        if head_type == "tokens" and head_prompt_version != "marker-v1":
            raise ValueError("candidate prompt versions require a custom decision head")
        if (head_type == "letter_code") != (head_prompt_version == "letter-code-v1"):
            raise ValueError(
                "letter-code readouts require head_type='letter_code' "
                "and head_prompt_version='letter-code-v1'"
            )
        if head_type == "letter_code" and decision_tokens > 255:
            raise ValueError("letter-code readout supports at most 255 options")
        if attention_mode not in ("causal", "noncausal_full_attention"):
            raise ValueError("attention_mode must be causal or noncausal_full_attention")
        if attention_mode != "causal" and head_type != "letter_code":
            raise ValueError("noncausal attention is only implemented for letter-code readouts")
        if not isinstance(multimodal_loader, bool):
            raise ValueError("multimodal_loader must be a boolean")
        if not isinstance(train_backbone, bool):
            raise ValueError("train_backbone must be a boolean")
        if not isinstance(bosun_template_fallback, bool) or (
            bosun_template_fallback and prompt_format != "bosun-v3"
        ):
            raise ValueError("Bosun chat-template fallback requires prompt_format='bosun-v3'")
        if not isinstance(learn_abstain, bool) or (
            learn_abstain
            and not (
                head_type == "pointer" or (head_type == "tokens" and prompt_format == "bosun-v3")
            )
        ):
            raise ValueError("learn_abstain requires a pointer head or Bosun token readout")
        if head_type == "pointer" and head_config and "abstain" in head_config:
            raise ValueError("configure pointer abstention with learn_abstain, not head_config")
        if prompt_format == "bosun-v3" and not getattr(tokenizer, "chat_template", None):
            if not bosun_template_fallback:
                raise ValueError("Bosun prompt requires a tokenizer chat template")
            tokenizer.chat_template = BOSUN_CONVERSION_CHAT_TEMPLATE
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.decision_tokens = decision_tokens
        self.temperature = temperature
        self.seed = seed
        self.max_sequence_length = max_sequence_length
        self.base_checkpoint = base_checkpoint
        self.source_checkpoint = source_checkpoint or base_checkpoint
        self.source_revision = source_revision
        self.base_model_revision = base_model_revision
        self.prompt_format = prompt_format
        self.prompt_schema = prompt_schema
        self.head_type = head_type
        self.head_config = dict(head_config or {})
        self.head_prompt_version = head_prompt_version
        self.train_backbone = train_backbone
        self.bosun_template_fallback = bosun_template_fallback
        self.learn_abstain = learn_abstain
        self.attention_mode = attention_mode
        self.multimodal_loader = multimodal_loader
        if head_type == "letter_code":
            codes, token_ids = answer_codes(tokenizer, decision_tokens)
            expected = {"codes": codes, "token_ids": token_ids}
            if self.head_config and self.head_config != expected:
                raise ValueError("saved letter-code answers differ from this tokenizer")
            self.head_config = expected
        # Store only simple, JSON-safe architecture parameters in checkpoints.
        json.dumps(self.head_config, allow_nan=False)
        if head_type == "tokens":
            self.decision_head = None
            self._token_ids = self._register_decision_tokens()
        else:
            marker_id = (
                self._register_candidate_marker() if head_prompt_version == "marker-v1" else None
            )
            if marker_id is None:
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token is None:
                        raise ValueError("tokenizer needs a pad or EOS token for batch inference")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left" if head_type == "letter_code" else "right"
            self._token_ids = []
            text_config = getattr(self.backbone.config, "text_config", self.backbone.config)
            hidden_size = getattr(text_config, "hidden_size", None)
            if not isinstance(hidden_size, int) or hidden_size < 1:
                raise ValueError("causal LM config must expose a positive text hidden_size")
            config = dict(self.head_config)
            if learn_abstain:
                config["abstain"] = True
            registered = build_decision_head(head_type, hidden_size, config)
            if decision_head is not None and type(decision_head) is not type(registered):
                raise TypeError("decision_head must match the registered head factory")
            self.decision_head = decision_head if decision_head is not None else registered
            embedding_device = self.backbone.get_input_embeddings().weight.device
            if embedding_device.type != "meta":
                self.decision_head.to(embedding_device)
            self._marker_id = marker_id
        if not train_backbone:
            self.backbone.requires_grad_(False)
        if attention_mode == "noncausal_full_attention":
            enable_qwen_noncausal_attention(self.backbone)

    def _register_candidate_marker(self) -> int:
        tokenizer = self.tokenizer
        is_new = CANDIDATE_MARKER not in tokenizer.get_vocab()
        existing = getattr(tokenizer, "additional_special_tokens", None)
        if existing is None:  # Transformers 5 uses extra_special_tokens.
            existing = getattr(tokenizer, "extra_special_tokens", [])
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(dict.fromkeys(existing + [CANDIDATE_MARKER]))}
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("tokenizer needs a pad or EOS token for batch inference")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        embeddings = self.backbone.get_input_embeddings()
        if embeddings.num_embeddings != len(tokenizer):
            if getattr(self.backbone, "is_loaded_in_4bit", False):
                raise ValueError(
                    "cannot resize a 4-bit checkpoint; add the candidate marker "
                    "and save the unquantized base before loading it with BitsAndBytesConfig"
                )
            self.backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        marker_id = tokenizer.convert_tokens_to_ids(CANDIDATE_MARKER)
        if tokenizer.encode(CANDIDATE_MARKER, add_special_tokens=False) != [marker_id]:
            raise ValueError("candidate marker must encode to exactly one token")
        if is_new:
            # Qwen may consume a pre-existing padded vocabulary row without a
            # resize. Initialize that row rather than treating it as trained.
            output_embeddings = self.backbone.get_output_embeddings()
            input_weight = self.backbone.get_input_embeddings().weight
            output_weight = output_embeddings.weight if output_embeddings is not None else None
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            std = float(getattr(self.backbone.config, "initializer_range", 0.02))
            with torch.no_grad():
                for weight in (
                    [input_weight]
                    if output_weight is None or output_weight is input_weight
                    else [input_weight, output_weight]
                ):
                    value = torch.randn(
                        weight.shape[1], generator=generator, dtype=torch.float32
                    ).mul_(std)
                    weight[marker_id].copy_(value.to(device=weight.device, dtype=weight.dtype))
        return marker_id

    def _register_decision_tokens(self) -> list[int]:
        tokens = [decision_token(i) for i in range(self.decision_tokens)]
        original_vocabulary = self.tokenizer.get_vocab()
        added_tokens = {token for token in tokens if token not in original_vocabulary}
        existing = getattr(self.tokenizer, "additional_special_tokens", None)
        if existing is None:  # Transformers 5 uses extra_special_tokens.
            existing = getattr(self.tokenizer, "extra_special_tokens", [])
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(dict.fromkeys(existing + tokens))}
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("tokenizer needs a pad or EOS token for batch inference")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" if self.prompt_format == "bosun-v3" else "right"
        embedding = self.backbone.get_input_embeddings()
        if embedding.num_embeddings != len(self.tokenizer):
            if getattr(self.backbone, "is_loaded_in_4bit", False):
                raise ValueError(
                    "cannot resize a 4-bit checkpoint; add decision tokens and "
                    "save the unquantized base before loading it with BitsAndBytesConfig"
                )
            # Independently initialize new decision rows; mean/covariance
            # resizing can produce near-identical logits with small vocabularies.
            self.backbone.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        if any(
            self.tokenizer.encode(token, add_special_tokens=False) != [id_]
            for token, id_ in zip(tokens, ids)
        ):
            raise ValueError("each decision token must encode to exactly one token ID")
        new_ids = [id_ for token, id_ in zip(tokens, ids) if token in added_tokens]
        if new_ids:
            # Some LMs (including Qwen3) ship more embedding rows than the
            # tokenizer has tokens. Adding decision tokens consumes those
            # *padded* rows, so resizing may only shrink the matrix and does
            # not initialize them. Explicitly seed independent input/output
            # rows without touching trained tokens in an existing checkpoint.
            input_weight = self.backbone.get_input_embeddings().weight
            output_embedding = self.backbone.get_output_embeddings()
            output_weight = output_embedding.weight if output_embedding is not None else None
            std = float(getattr(self.backbone.config, "initializer_range", 0.02))
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        decision_tokens: int | None = None,
        head_type: str | None = None,
        head_config: Mapping[str, Any] | None = None,
        head_prompt_version: str | None = None,
        train_backbone: bool | None = None,
        bosun_template_fallback: bool | None = None,
        learn_abstain: bool | None = None,
        attention_mode: str | None = None,
        multimodal_loader: bool | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_sequence_length: int | None = None,
        prompt_format: str | None = None,
        trainable_adapter: bool = False,
        **hf_kwargs: Any,
    ) -> DecisionModel:
        path = Path(checkpoint)
        decision_config = load_decision_config(checkpoint, **hf_kwargs)
        reject_native_checkpoint(checkpoint, decision_config)
        if cls is DecisionModel and decision_config.get("prompt_format") == "decider-v2.1":
            from futurix_decision.models.decider import DeciderDecisionModel

            cls = DeciderDecisionModel
        if cls is DecisionModel and decision_config.get("prompt_format") == "jpt-llm2jev-chat":
            from futurix_decision.models.jpt import JPTDecisionModel

            cls = JPTDecisionModel
        if cls is DecisionModel and decision_config.get("prompt_format") == "eikos-letter-v1-semif":
            from futurix_decision.models.eikos import EikosDecisionModel

            cls = EikosDecisionModel
        if (
            decision_config.get("prompt_format") == "decider-v2.1"
            and "type_temperatures" not in decision_config
        ):
            raise ValueError("decider-v2.1 checkpoint is missing type-specific calibration")
        stored_head_type = decision_config.get("head_type")
        if stored_head_type == "pplx":
            stored_head_type = "letter_code"
        if head_type == "pplx":
            head_type = "letter_code"
        if stored_head_type is not None and head_type is not None and head_type != stored_head_type:
            raise ValueError("cannot override the saved checkpoint's decision head type")
        if "head_config" in decision_config and head_config is not None:
            if dict(head_config) != decision_config["head_config"]:
                raise ValueError(
                    "cannot override the saved checkpoint's decision head configuration"
                )
        stored_prompt_version = decision_config.get("head_prompt_version")
        if stored_prompt_version == "pplx-v1":
            stored_prompt_version = "letter-code-v1"
        if head_prompt_version == "pplx-v1":
            head_prompt_version = "letter-code-v1"
        if (
            stored_prompt_version is not None
            and head_prompt_version is not None
            and head_prompt_version != stored_prompt_version
        ):
            raise ValueError("cannot override the saved checkpoint's candidate prompt version")
        stored_train_backbone = decision_config.get("train_backbone")
        if (
            stored_train_backbone is not None
            and train_backbone is not None
            and train_backbone != stored_train_backbone
        ):
            raise ValueError("cannot override the saved checkpoint's backbone training policy")
        stored_bosun_fallback = decision_config.get("bosun_template_fallback")
        if (
            stored_bosun_fallback is not None
            and bosun_template_fallback is not None
            and bosun_template_fallback != stored_bosun_fallback
        ):
            raise ValueError("cannot override the saved checkpoint's Bosun chat-template fallback")
        stored_learn_abstain = decision_config.get("learn_abstain", False)
        if (
            decision_config.get("architecture") == "futurix-decision"
            and learn_abstain is not None
            and learn_abstain != stored_learn_abstain
        ):
            raise ValueError("cannot override the saved checkpoint's abstain readout")
        for key, override in (
            ("attention_mode", attention_mode),
            ("multimodal_loader", multimodal_loader),
        ):
            if key in decision_config and override is not None and override != decision_config[key]:
                raise ValueError(f"cannot override the saved checkpoint's {key}")
        tokenizer_kwargs = {
            key: hf_kwargs[key]
            for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
            if key in hf_kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), **tokenizer_kwargs)
        adapter = path.is_dir() and (path / "adapter_config.json").is_file()
        base_checkpoint = (
            decision_config.get("base_checkpoint")
            if adapter
            else str(path.resolve())
            if path.is_dir()
            else str(checkpoint)
        )
        if adapter and not base_checkpoint:
            raise ValueError(
                "adapter checkpoint is missing base_checkpoint in decision_config.json"
            )
        qwen35_base = str(checkpoint) in (
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-9B-Base",
        ) or decision_config.get("source_checkpoint") in (
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-9B-Base",
        )
        use_multimodal = (
            multimodal_loader
            if multimodal_loader is not None
            else decision_config.get("multimodal_loader", False)
        )
        if (
            decision_config.get("prompt_format") == "jpt-llm2jev-chat"
            or qwen35_base
            or use_multimodal
        ):
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError as exc:
                raise ImportError(
                    "JPT and Qwen3.5-9B require Transformers 5.17+; "
                    "install futurix-decision[jpt] or futurix-decision[qwen3.5]"
                ) from exc
            backbone_cls = AutoModelForMultimodalLM
        else:
            backbone_cls = AutoModelForCausalLM
        backbone_kwargs = dict(hf_kwargs)
        resolved_attention = attention_mode or decision_config.get("attention_mode", "causal")
        if resolved_attention == "noncausal_full_attention":
            if backbone_kwargs.get("attn_implementation", "sdpa") != "sdpa":
                raise ValueError("noncausal letter-code attention requires SDPA")
            backbone_kwargs["attn_implementation"] = "sdpa"
        if adapter and decision_config.get("base_model_revision"):
            backbone_kwargs["revision"] = decision_config["base_model_revision"]
        backbone = backbone_cls.from_pretrained(
            base_checkpoint if adapter else str(checkpoint), **backbone_kwargs
        )
        model = cls(
            backbone,
            tokenizer,
            decision_tokens=decision_tokens
            if decision_tokens is not None
            else decision_config.get("decision_tokens", 256),
            temperature=temperature
            if temperature is not None
            else decision_config.get("temperature", 1.0),
            seed=seed if seed is not None else decision_config.get("seed", 0),
            max_sequence_length=max_sequence_length
            if max_sequence_length is not None
            else decision_config.get("max_sequence_length", 1024),
            base_checkpoint=base_checkpoint,
            source_checkpoint=decision_config.get("source_checkpoint", str(checkpoint)),
            source_revision=hf_kwargs.get("revision", decision_config.get("source_revision")),
            base_model_revision=decision_config.get(
                "base_model_revision", hf_kwargs.get("revision")
            ),
            prompt_format=prompt_format
            if prompt_format is not None
            else decision_config.get("prompt_format", "futurix"),
            prompt_schema=decision_config.get("prompt_schema", BOSUN_PROMPT_SCHEMA),
            head_type=head_type or stored_head_type or "tokens",
            head_config=head_config
            if head_config is not None
            else decision_config.get("head_config"),
            head_prompt_version=head_prompt_version or stored_prompt_version or "marker-v1",
            train_backbone=(
                train_backbone
                if train_backbone is not None
                else stored_train_backbone
                if stored_train_backbone is not None
                else True
            ),
            bosun_template_fallback=(
                bosun_template_fallback
                if bosun_template_fallback is not None
                else stored_bosun_fallback
                if stored_bosun_fallback is not None
                else False
            ),
            learn_abstain=(learn_abstain if learn_abstain is not None else stored_learn_abstain),
            attention_mode=resolved_attention,
            multimodal_loader=use_multimodal,
            **(
                {"type_temperatures": decision_config["type_temperatures"]}
                if decision_config.get("prompt_format") == "decider-v2.1"
                and "type_temperatures" in decision_config
                else {}
            ),
        )
        if model.head_type == "letter_code" and stored_head_type is None:
            assert isinstance(model.decision_head, LetterCodeHead)
            model.decision_head.initialize_from_lm_head(
                model.backbone.get_output_embeddings(), model.head_config["token_ids"]
            )
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "adapter loading needs pip install futurix-decision[train]"
                ) from exc
            model.backbone = PeftModel.from_pretrained(
                model.backbone, str(checkpoint), is_trainable=trainable_adapter
            )
        if stored_head_type is not None and stored_head_type != "tokens":
            if not path.is_dir() or not (path / HEAD_WEIGHTS).is_file():
                raise ValueError(f"custom head checkpoint is missing {HEAD_WEIGHTS}")
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError("custom head loading requires safetensors") from exc
            assert model.decision_head is not None
            model.decision_head.load_state_dict(load_file(str(path / HEAD_WEIGHTS)), strict=True)
        return model

    @classmethod
    def from_bosun(
        cls,
        checkpoint: str | Path = "Hanno-Labs/bosun-v3.1-1.7b",
        *,
        revision: str | None = None,
        max_sequence_length: int = 4096,
        learn_abstain: bool = False,
        **hf_kwargs: Any,
    ) -> DecisionModel:
        """Load a pinned Hub release or a trusted local Bosun copy with its readout.

        Bosun is not a plain LM repository: its custom loader reconstructs a
        Qwen base, a PEFT adapter, and two sets of trained decision-token rows.
        Local copies still need their referenced Qwen base available on disk
        or downloadable by Transformers.
        """
        source, local = checkpoint_source(checkpoint)
        revision = revision or BOSUN_RELEASES.get(source)
        if (not local and revision is None) or (
            revision is not None
            and (
                len(revision) != 40
                or any(digit not in "0123456789abcdefABCDEF" for digit in revision)
            )
        ):
            raise ValueError("pin the Bosun repository to a full 40-character commit revision")
        if local:
            require_checkpoint_files(Path(source), "bosun")
        package = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=True,
            **({"revision": revision} if revision else {}),
            **hf_kwargs,
        )
        if (
            getattr(package.config, "prompt_schema", None) != BOSUN_PROMPT_SCHEMA
            or getattr(package.config, "decision_token_count", None) != 256
            or getattr(package.config, "null_token_index", None) != 255
        ):
            raise ValueError(
                "checkpoint does not implement the supported Bosun v3 decision contract"
            )
        if package.tokenizer is None:
            raise ValueError("Bosun checkpoint did not load its tokenizer")
        return cls(
            package.model,
            package.tokenizer,
            decision_tokens=256,
            max_sequence_length=max_sequence_length,
            base_checkpoint=package.config.base_model_name_or_path,
            source_checkpoint=source,
            source_revision=revision,
            base_model_revision=package.config.base_model_revision,
            prompt_format="bosun-v3",
            prompt_schema=package.config.prompt_schema,
            learn_abstain=learn_abstain,
        )

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
        if getattr(self.backbone, "peft_config", None):
            if getattr(self.backbone, "is_loaded_in_4bit", False):
                if not self.base_checkpoint:
                    raise ValueError("a 4-bit adapter needs a recoverable base_checkpoint")
                if (
                    Path(self.base_checkpoint).is_dir()
                    and Path(self.base_checkpoint).resolve() == path.resolve()
                ):
                    raise ValueError("QLoRA adapter output cannot overwrite its base checkpoint")
                adapter_base = self.base_checkpoint
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
        if self.decision_head is not None:
            try:
                from safetensors.torch import save_file
            except ImportError as exc:
                raise ImportError("custom head saving requires safetensors") from exc
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in self.decision_head.state_dict().items()
            }
            save_file(state, str(path / HEAD_WEIGHTS))
        (path / DECISION_CONFIG).write_text(
            json.dumps(
                {
                    "architecture": "futurix-decision",
                    "head_type": self.head_type,
                    "head_config": self.head_config,
                    "head_prompt_version": self.head_prompt_version,
                    "train_backbone": self.train_backbone,
                    "bosun_template_fallback": self.bosun_template_fallback,
                    "learn_abstain": self.learn_abstain,
                    "attention_mode": self.attention_mode,
                    "multimodal_loader": self.multimodal_loader,
                    "decision_tokens": self.decision_tokens,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "max_sequence_length": self.max_sequence_length,
                    "prompt_format": self.prompt_format,
                    "prompt_schema": self.prompt_schema,
                    **(
                        {"source_checkpoint": self.source_checkpoint}
                        if self.source_checkpoint
                        else {}
                    ),
                    **({"source_revision": self.source_revision} if self.source_revision else {}),
                    **(
                        {"base_model_revision": self.base_model_revision}
                        if self.base_model_revision
                        else {}
                    ),
                    "supported_types": ["choice", "noul", "score"],
                    **({"base_checkpoint": adapter_base} if adapter_base else {}),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _prompt_and_slots(
        self, example: DecisionInput, assignment_seed: int | None
    ) -> tuple[str, list[int]]:
        if self.head_type == "letter_code":
            return format_letter_code_prompt(example, self.tokenizer, self.head_config["codes"])
        if self.decision_head is not None:
            return format_head_prompt(
                example,
                max_options=self.decision_tokens,
                seed=self.seed,
                assignment_seed=assignment_seed,
            )
        if self.prompt_format == "bosun-v3":
            return format_bosun_prompt(
                example,
                self.tokenizer,
                decision_tokens=self.decision_tokens,
                seed=self.seed,
                assignment_seed=assignment_seed,
                prompt_schema=self.prompt_schema,
                learn_abstain=self.learn_abstain,
            )
        return format_decision_prompt(
            example,
            decision_tokens=self.decision_tokens,
            seed=self.seed,
            assignment_seed=assignment_seed,
        )

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        return self._infer([example], [confidence_threshold])[0]

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]:
        return self._infer(examples, [confidence_threshold] * len(examples))

    def system_one(
        self, state: State, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Answer typed System One questions with this model's trained readout."""
        if not questions:
            raise ValueError("system_one needs at least one question")
        examples = []
        for question_id, question in questions.items():
            if (
                not isinstance(question_id, str)
                or not question_id
                or not isinstance(question, Mapping)
            ):
                raise ValueError("system_one questions must be named mappings")
            kind = question.get("type", "choice")
            criteria = question.get("criteria")
            if kind == "noul":
                if criteria is not None and (
                    not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}
                ):
                    raise ValueError("noul criteria must use false/true keys")
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
                        [str(i) for i in range(len(criteria))]
                        if kind == "score"
                        else list(criteria)
                    )
                    descriptions = list(criteria) if kind == "score" else [None] * len(names)
                else:
                    raise ValueError(f"{kind} question needs criteria options")
            else:
                raise ValueError(f"unsupported system_one question type: {kind!r}")
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
        answers = {}
        for question_id, example, result in zip(
            questions, examples, self.batch_decide(examples), strict=True
        ):
            answer: dict[str, Any] = {
                "type": example.kind,
                "probabilities": result.probabilities,
            }
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

    def _input_device(self) -> torch.device:
        device = self.backbone.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
        for target in getattr(self.backbone, "hf_device_map", {}).values():
            if isinstance(target, int):
                return torch.device("cuda", target)
            if isinstance(target, str) and target.startswith("cuda"):
                return torch.device(target)
        return torch.device("cpu")

    def _candidate_hidden_states(self, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # Call the transformer's base directly: a full vocabulary projection
        # would allocate [batch, sequence, vocabulary] logits on many LMs.
        source = self.backbone
        if getattr(source, "peft_config", None):
            source = source.get_base_model()
        base = getattr(source, "base_model", None)
        if not isinstance(base, torch.nn.Module) or base is source:
            raise ValueError("the causal LM must expose a base transformer for pointer decisions")
        output = base(**encoded, return_dict=True, use_cache=False)
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("the causal LM base must return [batch, sequence, hidden] states")
        return hidden

    def forward_decisions(
        self, examples: Sequence[DecisionInput], *, assignment_seed: int | None = None
    ) -> DecisionBatch:
        """Single LM forward for a batch; preserve gradients for all readouts.

        Change assignment_seed between training steps/epochs to randomize the
        candidate-to-slot mapping reproducibly. Inference uses a stable mapping.
        """
        if not examples:
            raise ValueError("examples must not be empty")
        if any(row.images for row in examples):
            raise ValueError("Futurix's text-only decision head does not accept images")
        endpoints: list[list[int]] | None = None
        if self.decision_head is not None and self.head_prompt_version == "span-v1":
            prepared = [
                format_tokenized_head_prompt(
                    row,
                    self.tokenizer,
                    max_options=self.decision_tokens,
                    seed=self.seed,
                    assignment_seed=assignment_seed,
                )
                for row in examples
            ]
            slots = [item[1] for item in prepared]
            endpoints = [item[2] for item in prepared]
            lengths = [len(item[0]) for item in prepared]
            width = max(lengths)
            input_ids = torch.full(
                (len(examples), width), self.tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, (tokens, _, _) in enumerate(prepared):
                input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                attention_mask[row, : len(tokens)] = 1
            encoded: Mapping[str, torch.Tensor] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            prompts_and_slots = [self._prompt_and_slots(row, assignment_seed) for row in examples]
            prompts = [item[0] for item in prompts_and_slots]
            slots = [item[1] for item in prompts_and_slots]
            encoded = self.tokenizer(
                prompts, padding=True, return_tensors="pt", add_special_tokens=True
            )
        if encoded["input_ids"].shape[1] > self.max_sequence_length:
            raise ValueError(
                f"prompt exceeds max_sequence_length={self.max_sequence_length}; "
                "increase the limit instead of silently truncating candidates"
            )
        device = self._input_device()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if self.head_type == "letter_code":
            assert isinstance(self.decision_head, LetterCodeHead)
            hidden = self._candidate_hidden_states(encoded)
            if next(self.decision_head.parameters()).device != hidden.device:
                if self.backbone.training:
                    raise ValueError(
                        "training a device-dispatched decision head needs a single execution device"
                    )
                self.decision_head.to(hidden.device)
            # Letter-code inputs are left-padded; the final hidden state sees
            # the complete state, question, and presented answer-code table.
            scores = self.decision_head(hidden[:, -1])
            available = (
                torch.arange(
                    self.decision_tokens,
                    device=scores.device,
                )[None, :]
                < torch.tensor([len(row.options) for row in examples], device=scores.device)[
                    :, None
                ]
            )
            if not torch.isfinite(scores[available]).all():
                raise ValueError("letter-code readout produced non-finite candidate logits")
            return DecisionBatch(scores.masked_fill(~available, float("-inf")), slots)
        if self.decision_head is not None:
            if self.head_prompt_version == "span-v1":
                hidden = self._candidate_hidden_states(encoded)
            else:
                # Preserve the published marker-v1 forward contract for
                # already exported pointer-head checkpoints.
                hidden = self.backbone(
                    **encoded,
                    output_hidden_states=True,
                    logits_to_keep=1,
                ).hidden_states[-1]
            if next(self.decision_head.parameters()).device != hidden.device:
                if self.backbone.training:
                    raise ValueError(
                        "training a device-dispatched pointer head needs a single execution device"
                    )
                self.decision_head.to(hidden.device)
            sizes = [len(row.options) for row in examples]
            max_options = max(sizes)
            candidates = hidden.new_zeros((len(examples), max_options, hidden.shape[-1]))
            available = torch.zeros(
                (len(examples), max_options), dtype=torch.bool, device=hidden.device
            )
            for row, count in enumerate(sizes):
                if endpoints is not None:
                    markers = torch.tensor(endpoints[row], device=hidden.device)
                else:
                    markers = (
                        encoded["input_ids"][row].eq(self._marker_id).nonzero(as_tuple=True)[0]
                    )
                    if markers.numel() != count:
                        raise ValueError("each option must have exactly one candidate marker")
                candidates[row, :count] = hidden[row, markers]
                available[row, :count] = True
            last = encoded["attention_mask"].sum(dim=1).to(hidden.device) - 1
            state = hidden[torch.arange(len(examples), device=hidden.device), last]
            decision_logits = self.decision_head(state, candidates, available)
            if decision_logits.shape != available.shape:
                raise ValueError("decision head must return one logit per supplied option")
            if not torch.isfinite(decision_logits[available]).all():
                raise ValueError("decision head produced non-finite logits")
            return DecisionBatch(decision_logits.masked_fill(~available, float("-inf")), slots)
        output = self.backbone(
            **encoded, **({"logits_to_keep": 1} if self.prompt_format == "bosun-v3" else {})
        )
        last_positions = (
            torch.zeros(len(examples), dtype=torch.long, device=device)
            if output.logits.shape[1] == 1
            else (
                torch.full((len(examples),), output.logits.shape[1] - 1, device=device)
                if self.tokenizer.padding_side == "left"
                else encoded["attention_mask"].sum(dim=1) - 1
            )
        )
        last_logits = output.logits[torch.arange(len(examples), device=device), last_positions]
        token_ids = torch.tensor(self._token_ids, device=device)
        decision_logits = last_logits.index_select(dim=-1, index=token_ids)
        return DecisionBatch(mask_unavailable_slots(decision_logits, slots), slots)

    def _infer(
        self, examples: Sequence[DecisionInput], thresholds: Sequence[float | None]
    ) -> list[DecisionResult]:
        if not examples:
            return []
        for threshold in thresholds:
            if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
                raise ValueError("confidence_threshold must be between 0 and 1")
        device = self._input_device()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        self.backbone.eval()
        if self.decision_head is not None:
            self.decision_head.eval()
        with torch.inference_mode():
            batch = self.forward_decisions(examples)
            probabilities = torch.softmax(batch.logits.float() / self.temperature, dim=-1)
            probabilities = probabilities.cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000
        results = []
        for row, (example, slots, threshold) in enumerate(zip(examples, batch.slots, thresholds)):
            values = [probabilities[row, slot].item() for slot in slots]
            winner = max(range(len(values)), key=values.__getitem__)
            confidence = values[winner]
            abstained = threshold is not None and confidence < threshold
            decision: str | int | bool | None = None if abstained else example.options[winner]
            if decision is not None and example.kind == "noul" and not example.preserve_order:
                decision = decision == "true"
            if decision is not None and example.kind == "score" and not example.preserve_order:
                decision = int(decision)
            results.append(
                DecisionResult(
                    decision=decision,
                    confidence=confidence,
                    probabilities=dict(zip(example.options, values)),
                    abstained=abstained,
                    latency_ms=latency_ms,
                )
            )
        return results

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        example = DecisionInput(
            kind="choice", state=state, question=question, options=list(choices)
        )
        return self._infer([example], [confidence_threshold])[0]

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        example = DecisionInput(
            kind="noul", state=state, question=question, options=["false", "true"]
        )
        return self._infer([example], [confidence_threshold])[0]

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        example = DecisionInput(
            kind="score",
            state=state,
            question=question,
            options=[str(level) for level in range(min_score, max_score + 1)],
        )
        return self._infer([example], [confidence_threshold])[0]

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
                    raise ValueError("batch_choice accepts only choice inputs")
                inputs.append(row)
                thresholds.append(confidence_threshold)
            else:
                if row.get("kind", "choice") != "choice":
                    raise ValueError("batch_choice accepts only choice inputs")
                inputs.append(
                    DecisionInput(
                        state=row["state"],
                        question=row.get("question", "Which option should be selected?"),
                        options=list(row.get("choices", row.get("options", []))),
                    )
                )
                thresholds.append(row.get("confidence_threshold", confidence_threshold))
        return self._infer(inputs, thresholds)
