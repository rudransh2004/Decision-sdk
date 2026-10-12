"""Bosun v3.1 releases: Hanno Labs' Qwen base, PEFT adapter, and decision-token rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.families.base import FamilyAdapter, reject_futurix_options
from futurix_decision.models.sources import (
    CheckpointFiles,
    checkpoint_source,
    optional_json,
    read_checkpoint_json,
)

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel

# The Hub commits the SDK verified; a Hub ID loads only at a pinned commit.
BOSUN_RELEASES = {
    "Hanno-Labs/bosun-v3.1-1.7b": "1d8dc82a20e4a32ed60927a47272d6efff48eed2",
    "Hanno-Labs/bosun-v3.1-0.6b": "5e78a37c8d86860ebac9ef2d5327d6299498498e",
}


class BosunAdapter(FamilyAdapter):
    name = "bosun"
    checkpoints = tuple(BOSUN_RELEASES)
    aliases = {
        "bosun-0.6b": "Hanno-Labs/bosun-v3.1-0.6b",
        "bosun-1.7b": "Hanno-Labs/bosun-v3.1-1.7b",
    }
    markers = ("decision_embeddings.safetensors",)
    extra = "bosun"
    trainable = True
    vllm = "export"

    def detect(self, directory: Path) -> bool:
        return (
            super().detect(directory)
            or optional_json(directory, "config.json").get("model_type") == "bosun"
        )

    def load(
        self,
        checkpoint: str,
        *,
        device: str | None = None,
        revision: str | None = None,
        max_sequence_length: int = 4096,
        learn_abstain: bool = False,
        **hf_kwargs: Any,
    ) -> DecisionModel:
        """Load a pinned Hub release or a trusted local Bosun copy with its readout.

        Bosun is not a plain LM repository: its custom loader reconstructs a
        Qwen base, a PEFT adapter, and two sets of trained decision-token rows.
        Local copies still need their referenced Qwen base available on disk
        or downloadable by Transformers. Callers check local files with
        ``requires`` first, as ``load_model`` does.
        """
        reject_futurix_options(hf_kwargs, self.name)
        from transformers import AutoModelForCausalLM

        from futurix_decision.models.bosun import BOSUN_PROMPT_SCHEMA
        from futurix_decision.models.decision_config import DecisionModelConfig
        from futurix_decision.models.decision_model import DecisionModel

        source, local = checkpoint_source(checkpoint)
        revision = revision or BOSUN_RELEASES.get(source)
        if (not local and revision is None) or (
            revision is not None
            and (
                len(revision) != 40
                or any(digit not in "0123456789abcdefABCDEF" for digit in revision)
            )
        ):
            raise ConfigError("pin the Bosun repository to a full 40-character commit revision")
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
            raise CheckpointError(
                "checkpoint does not implement the supported Bosun v3 decision contract"
            )
        if package.tokenizer is None:
            raise CheckpointError("Bosun checkpoint did not load its tokenizer")
        config = DecisionModelConfig(
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
        return DecisionModel(package.model, package.tokenizer, config).to(device)

    def requires(self, path: Path) -> None:
        files = CheckpointFiles(path)
        files.require("config.json")
        if files.has("config.json"):
            config = read_checkpoint_json(path / "config.json")
            files.require(
                str(config.get("decision_embeddings_file") or "decision_embeddings.safetensors")
            )
            adapter = str(config.get("adapter_subfolder") or "adapter")
            tokenizer = str(config.get("tokenizer_subfolder") or "tokenizer")
            files.require(f"{adapter}/adapter_config.json")
            files.one_of(f"{adapter}/adapter_model.safetensors", f"{adapter}/adapter_model.bin")
            files.one_of(f"{tokenizer}/tokenizer.json", f"{tokenizer}/vocab.json")
        files.check(self.name)

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        if section.get("head_type", "tokens") != "tokens":
            raise ConfigError("Bosun's prompt contract requires the token decision head")
        if method == "qlora":
            raise ConfigError("Bosun releases load unquantized; fine-tune them with full or lora")
        if "max_sequence_length" in section:
            return {"max_sequence_length": section["max_sequence_length"]}
        return {}
