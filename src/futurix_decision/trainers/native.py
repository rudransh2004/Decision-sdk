"""Publisher-specific fine-tuning with complete, reloadable native artifacts."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from futurix_decision.datasets.schema import DecisionExample
from futurix_decision.datasets.systemone import native_question, option_keys
from futurix_decision.errors import (
    CheckpointError,
    ConfigError,
    InvalidOutputError,
    MissingDependencyError,
)
from futurix_decision.models.families.publisher import load_images
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


def _grouped_records(rows: Iterable[DecisionExample]) -> Iterable[dict[str, Any]]:
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


def _text_only(rows: Iterable[DecisionExample], family: str) -> Iterable[DecisionExample]:
    for row in rows:
        if row.images:
            raise ConfigError(f"{family} native training accepts text/JSON decisions, not images")
        yield row


def _gliner_record(row: DecisionExample, *, target_policy: str) -> dict[str, Any]:
    if row.images:
        raise ConfigError("GLiNER's encoder cannot train on image decisions")
    target = row.target_distribution()
    if target_policy not in ("error", "argmax"):
        raise ConfigError("target_policy must be 'error' or 'argmax'")
    if max(target) < 1 - 1e-7 and target_policy != "argmax":
        raise ConfigError(
            "GLiNER's native classifier only accepts hard labels; use "
            "target_policy='argmax' to explicitly discard soft probabilities"
        )
    winner = row.options[max(range(len(target)), key=target.__getitem__)]
    labels = (
        dict(zip(row.options, row.candidate_descriptions, strict=True))
        if row.candidate_descriptions
        else None
    )
    text = row.state if isinstance(row.state, str) else json.dumps(row.state, ensure_ascii=False)
    task: dict[str, Any] = {
        "task": row.question_id or "decision",
        "labels": row.options,
        "true_label": [winner],
        "prompt": row.question,
    }
    if labels:
        task["label_descriptions"] = {
            name: description for name, description in labels.items() if description
        }
    return {"input": text, "output": {"classifications": [task]}}


def _write_rows(destination: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    if not count:
        raise ConfigError("training dataset contains no labeled decisions")
    return count


def _kev_records(rows: Iterable[DecisionExample]) -> Iterable[dict[str, Any]]:
    for record in _grouped_records(rows):
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


def _train_pplx(
    model: NativeDecisionModel,
    rows: list[DecisionExample],
    path: Path,
    config: NativeSFTConfig,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    native = model.runtime
    native.requires_grad_(True)
    if not config.train_backbone:
        native.backbone.requires_grad_(False)
    native.train(True)
    optimizer = torch.optim.AdamW(
        (p for p in native.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    for _epoch in range(config.epochs):
        for offset in range(0, len(rows), config.batch_size):
            examples = rows[offset : offset + config.batch_size]
            if any(len(row.options) > 255 for row in examples):
                raise ConfigError("PPLX native readout supports at most 255 options")
            records = [
                {
                    "state": row.state,
                    "question": native_question(row),
                    **({"images": row.images} if row.images else {}),
                }
                for row in examples
            ]
            batch = native.prepare(records)
            logits = native(batch)
            if logits.shape[0] != len(examples):
                raise InvalidOutputError("PPLX returned the wrong batch size")
            terms = []
            for prediction, row in zip(logits, examples, strict=True):
                if prediction.shape[0] < len(row.options):
                    raise InvalidOutputError("PPLX returned fewer logits than decision options")
                target = torch.tensor(
                    row.target_distribution(), device=prediction.device, dtype=torch.float32
                )
                terms.append(
                    -(
                        target
                        * functional.log_softmax(
                            prediction[: len(row.options)].float(),
                            dim=-1,
                        )
                    ).sum()
                )
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    native.eval()
    native.save(path)
    return {"train_items": len(rows), "losses": losses}


def _train_clef(
    model: NativeDecisionModel,
    rows: list[DecisionExample],
    path: Path,
    config: NativeSFTConfig,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import save_file

    module, native, processor = model.runtime
    source = Path(module.__file__).parent
    for filename in ("joint_schema_model.py", "joint_head_config.json"):
        if not (source / filename).is_file():
            raise CheckpointError(f"Clef cannot export without its native {filename}")
    native.language_model.requires_grad_(config.train_backbone)
    native.head.requires_grad_(True)
    native.train()
    if not config.train_backbone:
        native.language_model.eval()
    optimizer = torch.optim.AdamW(
        (p for p in native.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    for _epoch in range(config.epochs):
        for offset in range(0, len(rows), config.batch_size):
            examples = rows[offset : offset + config.batch_size]
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
                for row in examples
            ]
            device = next(native.parameters()).device
            batch = module.collate_records(records, processor.tokenizer.pad_token_id, device)
            logits = native(batch)
            if len(logits) != len(examples) or any(len(question) != 1 for question in logits):
                raise InvalidOutputError("Clef returned unexpected joint-schema logits")
            terms = []
            for i, (question, row) in enumerate(zip(logits, examples, strict=True)):
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
    path.mkdir(parents=True, exist_ok=True)
    native.language_model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in native.head.state_dict().items()},
        str(path / "joint_head.safetensors"),
    )
    shutil.copyfile(source / "joint_schema_model.py", path / "joint_schema_model.py")
    shutil.copyfile(source / "joint_head_config.json", path / "joint_head_config.json")
    return {"train_items": len(rows), "losses": losses}


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
    if model.family not in ("laya", "gliner", "kev", "clef", "pplx-decider"):
        raise NotImplementedError(f"{model.family} has no verified native training/export path")
    with tempfile.TemporaryDirectory(prefix="futurix-native-data-") as temporary:
        data = Path(temporary) / "train.jsonl"
        if model.family == "laya":
            if kwargs.keys() - {"device"}:
                raise TypeError(
                    f"unsupported Laya training options: {sorted(kwargs.keys() - {'device'})}"
                )
            count = _write_rows(
                data,
                _grouped_records(_text_only(_canonical_rows(dataset), "Laya")),
            )
            try:
                from laya.train import finetune
            except ImportError as exc:
                raise MissingDependencyError(
                    "install futurix-decision[laya] to train Laya"
                ) from exc
            if isinstance(config, NativeSFTConfig):
                from laya.train import TrainConfig

                config = TrainConfig(
                    epochs=config.epochs,
                    micro_batch=config.batch_size,
                    grad_accum=1,
                    encoder_lr=config.learning_rate,
                    head_lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    freeze_encoder=not config.train_backbone,
                )
            report = finetune(
                str(data),
                model.checkpoint,
                str(path),
                config=config,
                device=kwargs.get("device", "auto"),
            )
        elif model.family == "gliner":
            target_policy = kwargs.pop("target_policy", "error")
            if kwargs:
                raise TypeError(f"unsupported GLiNER training options: {sorted(kwargs)}")
            count = _write_rows(
                data,
                (
                    _gliner_record(row, target_policy=target_policy)
                    for row in _canonical_rows(dataset)
                ),
            )
            try:
                from gliner2 import AutoExtractor
                from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
            except ImportError as exc:
                raise MissingDependencyError(
                    "install futurix-decision[gliner-train] to train GLiNER"
                ) from exc
            if config is None:
                config = TrainingConfig(output_dir=str(path))
            elif isinstance(config, NativeSFTConfig):
                if not config.train_backbone:
                    raise ConfigError(
                        "GLiNER's native trainer does not expose readout-only training"
                    )
                config = TrainingConfig(
                    output_dir=str(path),
                    num_epochs=config.epochs,
                    batch_size=config.batch_size,
                    encoder_lr=config.learning_rate,
                    task_lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    num_workers=0,
                )
            elif Path(config.output_dir) != path:
                raise ConfigError("GLiNER config.output_dir must equal output_dir")
            extractor = AutoExtractor.from_pretrained(model.checkpoint)
            report = ExtractorTrainer(extractor, config).train(train_data=str(data))
            path = path / "final"  # Native GLiNER trainer writes a reloadable release here.
        elif model.family == "kev":
            if config is not None and not isinstance(config, Mapping):
                raise TypeError("Kev CLI config must be a mapping of CLI option names to values")
            flags = dict(config or {})
            flags.update(kwargs)
            if flags.keys() & {"data", "out", "init_from", "base", "base_revision"}:
                raise ConfigError(
                    "Kev data/out/init_from/base/base_revision are managed by the SDK"
                )
            count = _write_rows(data, _kev_records(_text_only(_canonical_rows(dataset), "Kev")))
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
                str(path),
            ]
            if getattr(release.meta, "base_revision", None):
                command.extend(["--base_revision", release.meta.base_revision])
            for key, value in flags.items():
                if not key.replace("_", "").isalnum() or key.startswith("_"):
                    raise ConfigError(f"invalid Kev training option: {key}")
                command.extend(
                    [f"--{key}", str(int(value)) if isinstance(value, bool) else str(value)]
                )
            subprocess.run(command, check=True)
            report = {"train_items": count}
        else:
            if kwargs:
                raise TypeError(f"unsupported {model.family} training options: {sorted(kwargs)}")
            rows = list(_canonical_rows(dataset))
            if not rows:
                raise ConfigError("training dataset contains no labeled decisions")
            if config is not None and not isinstance(config, NativeSFTConfig):
                raise TypeError(f"{model.family} fine-tuning uses NativeSFTConfig")
            if model.family == "clef":
                report = _train_clef(
                    model,
                    rows,
                    path,
                    config or NativeSFTConfig(train_backbone=False),
                )
            else:
                report = _train_pplx(model, rows, path, config or NativeSFTConfig())
            count = len(rows)
    if not path.is_dir():
        raise RuntimeError(f"publisher trainer did not export a native checkpoint at {path}")
    _write_manifest(path, model)
    trained = NativeDecisionModel.from_pretrained(
        path,
        trust_remote_code=model.family in ("pplx-decider", "clef"),
        revision=model.source_revision,
    )
    trained.training_report = {"train_items": count, "publisher_report": report}
    return trained
