"""Clef: Cloudflare's joint-schema head, run through the release's pinned model code."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.datasets.systemone import native_question, option_keys
from futurix_decision.errors import CheckpointError, InvalidOutputError
from futurix_decision.models.families.publisher import (
    NativeExport,
    PublisherAdapter,
    PublisherRuntime,
    load_images,
)
from futurix_decision.models.sources import CheckpointFiles

if TYPE_CHECKING:
    from futurix_decision.datasets.schema import DecisionExample
    from futurix_decision.models.native import NativeDecisionModel


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

    def train(
        self,
        model: NativeDecisionModel,
        rows: Iterable[DecisionExample],
        *,
        data: Path,
        output: Path,
        config: Any | None,
        **kwargs: Any,
    ) -> NativeExport:
        """Fit the joint head (and backbone, if asked) on soft targets; export both heads."""
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import save_file

        from futurix_decision.trainers.native import NativeSFTConfig, sft_rows

        examples = sft_rows(self.name, rows, config, kwargs)
        settings = config or NativeSFTConfig(train_backbone=False)
        module, native, processor = model.runtime
        source = Path(module.__file__).parent
        for filename in ("joint_schema_model.py", "joint_head_config.json"):
            if not (source / filename).is_file():
                raise CheckpointError(f"Clef cannot export without its native {filename}")
        native.language_model.requires_grad_(settings.train_backbone)
        native.head.requires_grad_(True)
        native.train()
        if not settings.train_backbone:
            native.language_model.eval()
        optimizer = torch.optim.AdamW(
            (p for p in native.parameters() if p.requires_grad),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        losses: list[float] = []
        for _epoch in range(settings.epochs):
            for offset in range(0, len(examples), settings.batch_size):
                batch_rows = examples[offset : offset + settings.batch_size]
                records = [
                    module.encode_record(
                        processor.tokenizer,
                        {
                            "id": row.id,
                            "state": row.state,
                            "questions": {"decision": native_question(row)},
                            **({"images": load_images(row.images)} if row.images else {}),
                        },
                        processor=processor,
                    )
                    for row in batch_rows
                ]
                device = next(native.parameters()).device
                batch = module.collate_records(records, processor.tokenizer.pad_token_id, device)
                logits = native(batch)
                if len(logits) != len(batch_rows) or any(len(question) != 1 for question in logits):
                    raise InvalidOutputError("Clef returned unexpected joint-schema logits")
                terms = []
                for i, (question, row) in enumerate(zip(logits, batch_rows, strict=True)):
                    prediction = question[0].float()
                    if len(prediction) != len(row.options):
                        raise InvalidOutputError("Clef returned a different option count")
                    # The native encoder may sort choice IDs. Align targets to the
                    # option IDs actually encoded instead of assuming user order.
                    keys = option_keys(row)
                    distribution = row.target_distribution()
                    target = torch.tensor(
                        [
                            distribution[keys.index(option)]
                            for option in records[i].questions[0].option_ids
                        ],
                        device=prediction.device,
                        dtype=torch.float32,
                    )
                    terms.append(-(target * functional.log_softmax(prediction, dim=-1)).sum())
                loss = torch.stack(terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        native.eval()
        output.mkdir(parents=True, exist_ok=True)
        native.language_model.save_pretrained(output, safe_serialization=True)
        processor.save_pretrained(output)
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in native.head.state_dict().items()
            },
            str(output / "joint_head.safetensors"),
        )
        shutil.copyfile(source / "joint_schema_model.py", output / "joint_schema_model.py")
        shutil.copyfile(source / "joint_head_config.json", output / "joint_head_config.json")
        return NativeExport(len(examples), {"train_items": len(examples), "losses": losses}, output)
