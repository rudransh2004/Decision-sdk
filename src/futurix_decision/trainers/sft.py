"""Decision SFT over dynamically assigned decision-token logits."""

from __future__ import annotations

import hashlib
import heapq
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, IterableDataset

from futurix_decision.datasets import DecisionExample, load_decisions, normalize_decision
from futurix_decision.datasets.systemone import expand_systemone_row
from futurix_decision.errors import ConfigError, MissingDependencyError
from futurix_decision.models import DecisionBatch, DecisionModel
from futurix_decision.trainers.losses import LossMode, decision_sft_loss
from futurix_decision.utils.logging import TrainingLogger

FineTuneMethod = Literal["full", "lora", "qlora"]
DatasetSource = str | Path | Mapping[str, Any] | Iterable[DecisionExample | Mapping[str, Any]]


@dataclass(frozen=True)
class SFTConfig:
    """Settings for decision SFT.

    ``method`` is ``"full"`` (train every unfrozen parameter), ``"lora"``, or
    ``"qlora"`` (a pre-quantized 4-bit base). ``loss_mode`` picks hard or soft
    cross-entropy; ``"auto"`` follows the targets. ``label_smoothing`` mixes
    targets with a uniform distribution over the available options.
    """

    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    method: FineTuneMethod = "full"
    loss_mode: LossMode = "auto"
    label_smoothing: float = 0.0
    precision: Literal["no", "bf16", "fp16"] = "no"
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str | list[str] = "all-linear"
    seed: int = 42
    log_stdout: bool = False
    log_jsonl: str | None = None
    wandb_project: str | None = None
    max_examples: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ConfigError(
                "epochs, batch_size, and gradient_accumulation_steps must be positive"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ConfigError("max_grad_norm must be positive and finite")
        if not 0 <= self.label_smoothing < 1:
            raise ConfigError("label_smoothing must be in [0, 1)")
        if self.method not in ("full", "lora", "qlora"):
            raise ConfigError("method must be full, lora, or qlora")
        if self.loss_mode not in ("auto", "hard", "soft"):
            raise ConfigError("loss_mode must be auto, hard, or soft")
        if self.precision not in ("no", "bf16", "fp16"):
            raise ConfigError("precision must be no, bf16, or fp16")
        if self.lora_rank < 1 or self.lora_alpha < 1 or not 0 <= self.lora_dropout < 1:
            raise ConfigError("invalid LoRA rank, alpha, or dropout")
        if self.max_examples is not None and self.max_examples < 1:
            raise ConfigError("max_examples must be positive when set")


def _canonical_rows(source: DatasetSource) -> Iterator[DecisionExample]:
    if isinstance(source, (str, Path, Mapping)):
        yield from load_decisions(source, streaming=True)
        return
    for index, row in enumerate(source):
        if isinstance(row, DecisionExample):
            yield row
        elif isinstance(row, Mapping):
            if "questions" in row:
                for question in expand_systemone_row(row, index=index):
                    yield normalize_decision(question, index=index)
            else:
                yield normalize_decision(row, index=index)
        else:
            raise ConfigError(f"row {index}: expected a DecisionExample or mapping")


def _representative_rows(source: DatasetSource, *, count: int, seed: int) -> list[DecisionExample]:
    """Choose an order-independent, reproducible subset across the entire split."""
    heap: list[tuple[int, int, DecisionExample]] = []
    for index, row in enumerate(_canonical_rows(source)):
        priority = int.from_bytes(
            hashlib.blake2b(f"{seed}:{row.id}".encode(), digest_size=8).digest(), "big"
        )
        item = (-priority, index, row)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for item in sorted(heap, key=lambda item: (item[0], item[2].id))]
    random.Random(seed).shuffle(selected)
    return selected


class _StreamingDecisions(IterableDataset):
    def __init__(self, source: DatasetSource, max_examples: int | None = None, seed: int = 42):
        self.source = source
        self.max_examples = max_examples
        self.seed = seed

    def __iter__(self) -> Iterator[DecisionExample]:
        if self.max_examples is None:
            yield from _canonical_rows(self.source)
        else:
            yield from _representative_rows(self.source, count=self.max_examples, seed=self.seed)


def _collate(rows: list[DecisionExample]) -> list[DecisionExample]:
    return rows


class SFTTrainer:
    """Accelerate-backed SFT; full, LoRA, and pre-quantized 4-bit QLoRA."""

    def __init__(self, model: DecisionModel, *, config: SFTConfig | None = None) -> None:
        self.model = model
        self.config = config or SFTConfig()
        self.history: list[dict[str, float | int]] = []
        self._trained = False
        self._accelerator: Any | None = None
        self._assignment_seed = 0
        self._step_metrics: dict[str, float] = {}

    def _configure_backbone(self) -> None:
        backbone = self.model.backbone
        method = self.config.method
        quantized = getattr(backbone, "is_loaded_in_4bit", False)
        if method == "full" and quantized:
            raise ConfigError("full fine-tuning cannot update quantized weights")
        if method == "lora" and quantized:
            raise ConfigError("4-bit checkpoint detected; use method='qlora'")
        if method == "qlora" and not quantized:
            raise ConfigError(
                "QLoRA requires a 4-bit checkpoint loaded with BitsAndBytesConfig; "
                "marker-based heads must be prepared before quantizing the base"
            )
        device_map = getattr(backbone, "hf_device_map", None)
        if device_map:
            placements = {str(value) for value in device_map.values()}
            if len(placements) > 1 or "cpu" in placements or "disk" in placements:
                raise ConfigError(
                    "training a device-dispatched or offloaded backbone is not supported; "
                    "use a single-device model or a distributed training setup"
                )
        existing_adapter = bool(getattr(backbone, "peft_config", None))
        if (
            self.model.config.prompt_format == "bosun-v3"
            and existing_adapter
            and not getattr(self.model, "_bosun_continuation_prepared", False)
        ):
            if quantized:
                raise ConfigError("merge the original Bosun adapter before 4-bit continuation")
            # Bosun ships a frozen inference adapter and separately trained
            # input/output rows. Merge its adapter into the already-restored
            # base, then train a fresh LoRA and both decision-token surfaces.
            # Neither the original adapter nor its decision rows are reset.
            backbone = backbone.merge_and_unload()
            self.model.backbone = backbone
            self.model._bosun_continuation_prepared = True
            existing_adapter = False
        if existing_adapter and method == "full":
            raise ConfigError(
                "cannot full-fine-tune a PEFT adapter; use its current LoRA/QLoRA method"
            )
        if existing_adapter and not any(param.requires_grad for param in backbone.parameters()):
            raise ConfigError(
                "the loaded PEFT adapter is frozen; reload with "
                "DecisionModel.from_pretrained(..., trainable_adapter=True)"
            )
        if self.config.gradient_checkpointing:
            getattr(backbone.config, "text_config", backbone.config).use_cache = False
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if method == "full" or existing_adapter:
            return
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise MissingDependencyError(
                "LoRA/QLoRA needs pip install futurix-decision[train]"
            ) from exc
        if method == "qlora":
            backbone = prepare_model_for_kbit_training(
                backbone,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        input_embedding = backbone.get_input_embeddings()
        output_embedding = backbone.get_output_embeddings()
        if output_embedding is None:
            raise ConfigError("the causal LM must expose output embeddings")
        tied = input_embedding.weight is output_embedding.weight
        modules_to_save = None
        if not tied and self.model.decision_head is None:
            if not isinstance(output_embedding, torch.nn.Linear):
                raise ConfigError("untied output embeddings must be a linear module for LoRA")
            names = [
                name for name, module in backbone.named_modules() if module is output_embedding
            ]
            if len(names) != 1:
                raise ConfigError("could not identify the LM's output projection")
            # The new decision rows in an untied output projection must be
            # learnable and must be retained with the adapter.
            modules_to_save = names
        trainable_indices = self.model.readout.token_ids or None
        adapter = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            trainable_token_indices=trainable_indices,
            ensure_weight_tying=tied,
            modules_to_save=modules_to_save,
        )
        self.model.backbone = get_peft_model(backbone, adapter)
        if self.config.gradient_checkpointing and method == "lora":
            self.model.backbone.enable_input_require_grads()

    def _loader(self, dataset: DatasetSource) -> DataLoader:
        if isinstance(dataset, Sequence) and not isinstance(dataset, (str, bytes)):
            rows = list(_canonical_rows(dataset))
            if self.config.max_examples is not None:
                rows = _representative_rows(
                    rows, count=self.config.max_examples, seed=self.config.seed
                )
            if not rows:
                raise ConfigError("SFT dataset is empty")
            generator = torch.Generator().manual_seed(self.config.seed)
            return DataLoader(
                rows,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=generator,
                collate_fn=_collate,
            )
        if self.config.epochs > 1 and isinstance(dataset, Iterator):
            raise ConfigError(
                "one-shot iterators cannot be replayed across epochs; "
                "pass a dataset path, HF ID, or re-iterable collection instead"
            )
        return DataLoader(
            _StreamingDecisions(dataset, self.config.max_examples, self.config.seed),
            batch_size=self.config.batch_size,
            collate_fn=_collate,
        )

    def compute_loss(
        self, batch: DecisionBatch, examples: Sequence[DecisionExample]
    ) -> torch.Tensor:
        return decision_sft_loss(
            batch,
            examples,
            mode=self.config.loss_mode,
            label_smoothing=self.config.label_smoothing,
        )

    def train(self, dataset: DatasetSource) -> list[dict[str, float | int]]:
        """Train on ``dataset`` and return the per-step history."""
        try:
            from accelerate import Accelerator, DataLoaderConfiguration
            from accelerate.utils import set_seed
        except ImportError as exc:
            raise MissingDependencyError("SFT needs pip install futurix-decision[train]") from exc
        if self._trained:
            raise RuntimeError("this trainer has already trained; create a new SFTTrainer")
        set_seed(self.config.seed)
        cache_config = getattr(
            self.model.backbone.config, "text_config", self.model.backbone.config
        )
        previous_use_cache = cache_config.use_cache
        self._configure_backbone()
        accelerator = Accelerator(
            mixed_precision=self.config.precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # The batches contain validated objects, not tensors. Shard the
            # iterable across ranks rather than broadcasting/slicing objects.
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        )
        params = [param for param in self.model.backbone.parameters() if param.requires_grad]
        if self.model.decision_head is not None:
            params.extend(
                param for param in self.model.decision_head.parameters() if param.requires_grad
            )
        if not params:
            raise ConfigError("the backbone and decision head have no trainable parameters")
        optimizer = torch.optim.AdamW(
            params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        loader = self._loader(dataset)
        if self.model.decision_head is None:
            prepared, optimizer, loader = accelerator.prepare(
                self.model.backbone, optimizer, loader
            )
            prepared_head = None
        else:
            prepared, prepared_head, optimizer, loader = accelerator.prepare(
                self.model.backbone, self.model.decision_head, optimizer, loader
            )
            self.model.decision_head = prepared_head
            self.model.decision_head.train()
        self.model.backbone = prepared
        self.model.backbone.train()
        step = 0
        microstep = 0

        logger = TrainingLogger(
            stdout=self.config.log_stdout if accelerator.is_main_process else False,
            jsonl=self.config.log_jsonl if accelerator.is_main_process else None,
            wandb_project=self.config.wandb_project if accelerator.is_main_process else None,
        )
        try:
            for epoch in range(self.config.epochs):
                seen = 0
                accumulated_loss = 0.0
                accumulated_batches = 0
                for examples in loader:
                    models = (
                        (self.model.backbone,)
                        if self.model.decision_head is None
                        else (self.model.backbone, self.model.decision_head)
                    )
                    with accelerator.accumulate(*models):
                        assignment_seed = self.config.seed + microstep
                        self._assignment_seed = assignment_seed
                        batch = self.model.forward_decisions(
                            examples, assignment_seed=assignment_seed
                        )
                        loss = self.compute_loss(batch, examples)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(params, self.config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                    seen += len(examples)
                    microstep += 1
                    accumulated_loss += loss.detach().float().item()
                    accumulated_batches += 1
                    if accelerator.sync_gradients:
                        step += 1
                        mean = torch.tensor(
                            accumulated_loss / accumulated_batches, device=accelerator.device
                        )
                        mean = accelerator.reduce(mean, reduction="mean")
                        entry = {
                            "epoch": epoch + 1,
                            "step": step,
                            "loss": mean.item(),
                            **self._step_metrics,
                        }
                        self.history.append(entry)
                        logger.log(entry)
                        accumulated_loss = 0.0
                        accumulated_batches = 0
                if seen == 0:
                    raise ConfigError("SFT dataset is empty")
        finally:
            logger.close()
            self.model.backbone = accelerator.unwrap_model(prepared)
            if prepared_head is not None:
                self.model.decision_head = accelerator.unwrap_model(prepared_head)
            if self.config.gradient_checkpointing:
                self.model.backbone.gradient_checkpointing_disable()
                getattr(
                    self.model.backbone.config, "text_config", self.model.backbone.config
                ).use_cache = previous_use_cache
        self._trained = True
        self._accelerator = accelerator
        return self.history

    def save_pretrained(self, path: str | Path, *, safe_serialization: bool = True) -> None:
        """Save the trained model to ``path``; call after ``train``."""
        if not self._trained:
            raise RuntimeError("train before saving a checkpoint")
        assert self._accelerator is not None
        self._accelerator.wait_for_everyone()
        if self._accelerator.is_main_process:
            self.model.save_pretrained(path, safe_serialization=safe_serialization)
        self._accelerator.wait_for_everyone()
