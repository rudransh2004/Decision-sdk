"""Kev: Jared Palmer's pointer head over a Qwen3.5 LoRA, run through the ``kev`` package."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError, MissingDependencyError
from futurix_decision.models.families.publisher import (
    NativeExport,
    PublisherAdapter,
    PublisherRuntime,
    require_hub_files,
)
from futurix_decision.models.sources import CheckpointFiles

if TYPE_CHECKING:
    from futurix_decision.datasets.schema import DecisionExample
    from futurix_decision.models.native import NativeDecisionModel


def _records(rows: Iterable[DecisionExample]) -> Iterable[dict[str, Any]]:
    """Grouped records whose questions carry the hard ``label`` and soft ``target``."""
    from futurix_decision.trainers.native import grouped_records

    for record in grouped_records(rows):
        questions = {}
        for qid, question in record["questions"].items():
            probs = record["gold"][qid]["probabilities"]
            if len(probs) > 255:
                raise ConfigError("Kev native pointer head supports at most 255 options")
            winner = max(probs, key=probs.__getitem__)
            questions[qid] = {
                **question,
                "label": (
                    winner == "true"
                    if question["type"] == "noul"
                    else int(winner)
                    if question["type"] == "score"
                    else winner
                ),
                "target": probs,
            }
        yield {"state": record["state"], "questions": questions}


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
        """Run ``python -m kev.train``; ``config`` and ``kwargs`` become its CLI flags."""
        from futurix_decision.trainers.native import text_only, write_rows

        if config is not None and not isinstance(config, Mapping):
            raise TypeError("Kev CLI config must be a mapping of CLI option names to values")
        flags = dict(config or {})
        flags.update(kwargs)
        if flags.keys() & {"data", "out", "init_from", "base", "base_revision"}:
            raise ConfigError("Kev data/out/init_from/base/base_revision are managed by the SDK")
        count = write_rows(data, _records(text_only(rows, "Kev")))
        try:
            from kev.checkpoint import Checkpoint
        except ImportError as exc:
            raise MissingDependencyError(
                "install futurix-decision[kev-native] to train Kev"
            ) from exc
        source = (
            f"{model.checkpoint}@{model.source_revision}"
            if model.source_revision and not Path(model.checkpoint).is_dir()
            else model.checkpoint
        )
        release = Checkpoint(source)
        command = [
            sys.executable,
            "-m",
            "kev.train",
            "--data",
            str(data),
            "--init_from",
            source,
            "--base",
            release.meta.base,
            "--out",
            str(output),
        ]
        if getattr(release.meta, "base_revision", None):
            command.extend(["--base_revision", release.meta.base_revision])
        for key, value in flags.items():
            if not key.replace("_", "").isalnum() or key.startswith("_"):
                raise ConfigError(f"invalid Kev training option: {key}")
            command.extend([f"--{key}", str(int(value)) if isinstance(value, bool) else str(value)])
        subprocess.run(command, check=True)
        return NativeExport(count, {"train_items": count}, output)
