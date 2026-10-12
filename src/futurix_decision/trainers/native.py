"""Publisher fine-tuning with complete, reloadable native artifacts.

``finetune_native`` owns what every family shares: the output directory, a
scratch data file, the ``futurix_native.json`` manifest, and reloading the
export. Each family's ``PublisherAdapter.train`` runs the publisher's trainer.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from futurix_decision.datasets.schema import DecisionExample
from futurix_decision.datasets.systemone import native_question, option_keys
from futurix_decision.errors import ConfigError
from futurix_decision.models.native import NativeDecisionModel
from futurix_decision.trainers.sft import DatasetSource, _canonical_rows


@dataclass(frozen=True)
class NativeSFTConfig:
    """Common smoke-training controls; publisher configs remain available for advanced tuning."""

    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    train_backbone: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ConfigError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigError("weight_decay must be non-negative and finite")


def _write_manifest(path: Path, model: NativeDecisionModel) -> None:
    (path / "futurix_native.json").write_text(
        json.dumps(
            {
                "family": model.family,
                "base_checkpoint": model.checkpoint,
                "source_revision": model.source_revision,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def grouped_records(rows: Iterable[DecisionExample]) -> Iterable[dict[str, Any]]:
    """Restore multi-question records when the examples share a grouped state."""
    current: dict[str, Any] | None = None
    last_group: str | None = None
    for row in rows:
        group = row.group_id or row.id
        if group != last_group:
            if current is not None:
                yield current
            current = {"state": row.state, "questions": {}, "gold": {}}
            last_group = group
        assert current is not None
        if current["state"] != row.state:
            raise ConfigError(f"group {group!r} contains different states")
        question_id = row.question_id or "decision"
        if question_id in current["questions"]:
            raise ConfigError(f"group {group!r} repeats question ID {question_id!r}")
        current["questions"][question_id] = native_question(row)
        current["gold"][question_id] = {
            "probabilities": dict(zip(option_keys(row), row.target_distribution(), strict=True))
        }
    if current is not None:
        yield current


def text_only(rows: Iterable[DecisionExample], family: str) -> Iterable[DecisionExample]:
    for row in rows:
        if row.images:
            raise ConfigError(f"{family} native training accepts text/JSON decisions, not images")
        yield row


def write_rows(destination: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    if not count:
        raise ConfigError("training dataset contains no labeled decisions")
    return count


def sft_rows(
    family: str, rows: Iterable[DecisionExample], config: Any | None, kwargs: Mapping[str, Any]
) -> list[DecisionExample]:
    """Check the options of a ``NativeSFTConfig`` run and read all of its rows."""
    if kwargs:
        raise TypeError(f"unsupported {family} training options: {sorted(kwargs)}")
    examples = list(rows)
    if not examples:
        raise ConfigError("training dataset contains no labeled decisions")
    if config is not None and not isinstance(config, NativeSFTConfig):
        raise TypeError(f"{family} fine-tuning uses NativeSFTConfig")
    return examples


def finetune_native(
    model: NativeDecisionModel,
    dataset: DatasetSource,
    output_dir: str | Path,
    *,
    config: Any | None = None,
    **kwargs: Any,
) -> NativeDecisionModel:
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing native checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter = model.adapter
    if not adapter.trainable:
        raise NotImplementedError(f"{model.family} has no verified native training/export path")
    with tempfile.TemporaryDirectory(prefix="futurix-native-data-") as temporary:
        export = adapter.train(
            model,
            _canonical_rows(dataset),
            data=Path(temporary) / "train.jsonl",
            output=path,
            config=config,
            **kwargs,
        )
    if not export.path.is_dir():
        raise RuntimeError(f"publisher trainer did not export a native checkpoint at {export.path}")
    _write_manifest(export.path, model)
    trained = NativeDecisionModel.from_pretrained(
        export.path,
        trust_remote_code=adapter.remote_code,
        revision=model.source_revision,
    )
    trained.training_report = {"train_items": export.items, "publisher_report": export.report}
    return trained
