"""Kev: Jared Palmer's pointer head over a Qwen3.5 LoRA, run through the ``kev`` package."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from futurix_decision.errors import MissingDependencyError
from futurix_decision.models.families.publisher import (
    PublisherAdapter,
    PublisherRuntime,
    require_hub_files,
)
from futurix_decision.models.sources import CheckpointFiles


class KevRuntime(PublisherRuntime):
    """``native`` is the ``(tokenizer, model)`` pair ``kev.checkpoint`` loads."""

    def system_one(self, payload: Mapping[str, Any]) -> Any:
        from kev.api import SystemOneRequest, to_answers, to_record
        from kev.model import admit

        tokenizer, model = self.native
        record, metadata = to_record(SystemOneRequest.model_validate(payload))
        encoded = admit(model, tokenizer, record)
        return {"answers": to_answers([prob.tolist() for prob in model.probs(encoded)], metadata)}


class KevAdapter(PublisherAdapter):
    name = "kev"
    extra = "kev-native"
    readout = "pointer"
    runtime_type = KevRuntime
    trainable = True
    max_options = 255

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        try:
            from kev.checkpoint import Checkpoint, LoadOptions
        except ImportError as exc:
            raise MissingDependencyError(
                "kev is required: install futurix-decision[kev-native]"
            ) from exc
        if not local:
            require_hub_files(name, self.name, revision, "head.pt")
        source = f"{name}@{revision}" if revision and not Path(name).is_dir() else name
        return Checkpoint(source).load(device, LoadOptions(**kwargs))

    def files(self, files: CheckpointFiles) -> None:
        files.require("head.pt")
        if files.has("adapter_config.json"):
            files.one_of("adapter_model.safetensors", "adapter_model.bin")
        else:
            files.require("config.json")
            files.weights()
