"""Clef: Cloudflare's joint-schema head, run through the release's pinned model code."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from futurix_decision.models.families.publisher import (
    PublisherAdapter,
    PublisherRuntime,
    load_images,
)
from futurix_decision.models.sources import CheckpointFiles


class ClefRuntime(PublisherRuntime):
    """``native`` is ``(module, model, processor)``: the release's code module and its loads."""

    def prepare_images(self, images: Sequence[Any]) -> list[Any]:
        return load_images(images)

    def system_one(self, payload: Mapping[str, Any]) -> Any:
        module, model, processor = self.native
        return module.systemone(model, processor, payload)


class ClefAdapter(PublisherAdapter):
    name = "clef"
    extra = "clef"
    readout = "joint-schema"
    runtime_type = ClefRuntime
    trainable = True
    images = True
    remote_code = True
    transformers = (5, 10)

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        directory = self.snapshot(
            name,
            local=local,
            revision=revision,
            heads=("joint_schema_model.py", "joint_head_config.json", "joint_head.safetensors"),
        )
        source = directory / "joint_schema_model.py"
        module_name = (
            "futurix_native_clef_" + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
        )
        spec = importlib.util.spec_from_file_location(module_name, source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        model, processor = module.load_release_model(directory, device=device, **kwargs)
        return (module, model, processor)

    def files(self, files: CheckpointFiles) -> None:
        files.require(
            "config.json",
            "joint_schema_model.py",
            "joint_head_config.json",
            "joint_head.safetensors",
        )
        files.tokenizer()
        files.weights()
