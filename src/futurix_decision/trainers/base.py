"""Train the SFT, calibration, and RLCD stages in sequence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from huggingface_hub import HfApi

from futurix_decision.datasets import load_decisions
from futurix_decision.errors import ConfigError
from futurix_decision.evaluation import evaluate, fit_temperature
from futurix_decision.models import DecisionModel
from futurix_decision.rewards import ProperScoringRule
from futurix_decision.trainers.calibration import CalibrationConfig, CalibrationTrainer
from futurix_decision.trainers.rlcd import RLCDConfig, RLCDTrainer
from futurix_decision.trainers.sft import DatasetSource, SFTConfig, SFTTrainer


class DecisionTrainer:
    def __init__(
        self,
        *,
        model: DecisionModel,
        dataset: DatasetSource,
        sft_config: SFTConfig | None = None,
        calibration_config: CalibrationConfig | None = None,
        rlcd_config: RLCDConfig | None = None,
        calibration_dataset: DatasetSource | None = None,
        rewards: Sequence[ProperScoringRule] | None = None,
        reference_model: DecisionModel | None = None,
        checkpoint_dir: str | Path | None = None,
        validation_dataset: DatasetSource | None = None,
        validation_max_rows: int | None = None,
    ) -> None:
        self.model = model
        self.sft_config = sft_config or SFTConfig()
        self.calibration_config = calibration_config
        self.rlcd_config = rlcd_config
        self.calibration_dataset = calibration_dataset
        self.rewards = rewards
        self.reference_model = reference_model
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.validation_dataset = validation_dataset
        if validation_max_rows is not None and validation_max_rows < 1:
            raise ConfigError("validation_max_rows must be positive")
        self.validation_max_rows = validation_max_rows
        self.validation_metrics: dict[str, dict] = {}
        self.selected_stage: str | None = None
        self.dataset_revision: str | None = None
        if isinstance(dataset, str) and not Path(dataset).exists():
            # Record the Hub commit, rather than a mutable branch name, for
            # later interpretation of a long-running training/benchmark run.
            self.dataset_revision = HfApi().dataset_info(dataset).sha
        self.sft = SFTTrainer(model, config=self.sft_config)
        self.dataset = dataset
        self._active: SFTTrainer | None = None
        self.history: dict[str, list[dict[str, float | int]]] = {}

    def _heldout(self) -> DatasetSource | None:
        if self.calibration_dataset is not None:
            return self.calibration_dataset
        if isinstance(self.dataset, str) and not Path(self.dataset).exists():
            return load_decisions(self.dataset, split="calibration", streaming=True)
        return None

    def _validation(self) -> DatasetSource | None:
        if self.validation_dataset is not None:
            return self.validation_dataset
        if isinstance(self.dataset, str) and not Path(self.dataset).exists():
            return load_decisions(self.dataset, split="validation", streaming=True)
        return None

    def train(
        self, *, stages: Sequence[str] = ("decision_sft",)
    ) -> list[dict[str, float | int]] | dict[str, list[dict[str, float | int]]]:
        stages = list(stages)
        if (
            not stages
            or len(set(stages)) != len(stages)
            or any(stage not in ("decision_sft", "calibration", "rlcd") for stage in stages)
        ):
            raise ConfigError("stages must be unique entries from decision_sft, calibration, rlcd")
        if len(stages) > 1 and isinstance(self.dataset, Iterator):
            raise ConfigError("multi-stage training requires a re-iterable dataset or path")
        if (
            "calibration" in stages
            and "rlcd" in stages
            and isinstance(self.calibration_dataset, Iterator)
        ):
            raise ConfigError("calibration_dataset must be re-iterable for post-RLCD refitting")
        if len(stages) > 1 and isinstance(self.validation_dataset, Iterator):
            raise ConfigError("validation_dataset must be re-iterable across stages")
        if "calibration" in stages and self._heldout() is None:
            raise ConfigError(
                "provide a separate calibration_dataset for held-out temperature fitting"
            )
        for stage in stages:
            if stage == "decision_sft":
                current = self.sft
                result = current.train(self.dataset)
            elif stage == "calibration":
                heldout = self._heldout()
                assert heldout is not None
                config = self.calibration_config or CalibrationConfig(
                    method=self.sft_config.method,
                    precision=self.sft_config.precision,
                    gradient_checkpointing=self.sft_config.gradient_checkpointing,
                    batch_size=self.sft_config.batch_size,
                    log_stdout=self.sft_config.log_stdout,
                    log_jsonl=self.sft_config.log_jsonl,
                    wandb_project=self.sft_config.wandb_project,
                )
                current = CalibrationTrainer(self.model, config=config)
                result = current.train(self.dataset, calibration_dataset=heldout)
            else:
                config = self.rlcd_config or RLCDConfig(
                    method=self.sft_config.method,
                    precision=self.sft_config.precision,
                    gradient_checkpointing=self.sft_config.gradient_checkpointing,
                    batch_size=self.sft_config.batch_size,
                    log_stdout=self.sft_config.log_stdout,
                    log_jsonl=self.sft_config.log_jsonl,
                    wandb_project=self.sft_config.wandb_project,
                )
                reference_model = self.reference_model
                if (
                    reference_model is None
                    and config.kl_beta
                    and self.checkpoint_dir
                    and self.history
                ):
                    # Reload the completed previous stage instead of
                    # deepcopying a model with PEFT/Accelerate device hooks.
                    # PyTorch 2.11 cannot deepcopy some of those hook handles.
                    previous = self.checkpoint_dir / next(reversed(self.history))
                    if previous.is_dir():
                        device = next(self.model.backbone.parameters()).device
                        load_kwargs = (
                            {"device_map": {"": device.index or 0}} if device.type == "cuda" else {}
                        )
                        reference_model = DecisionModel.from_pretrained(previous, **load_kwargs)
                current = RLCDTrainer(
                    self.model, config=config, rewards=self.rewards, reference_model=reference_model
                )
                result = current.train(self.dataset)
                # RLCD changes the policy, so the pre-RLCD temperature is
                # stale. Refit on held-out data or reset to uncalibrated T=1.
                heldout = self._heldout()
                self.model.config.temperature = 1.0
                if heldout is not None:
                    self.model.config.temperature = fit_temperature(
                        self.model, heldout, batch_size=config.batch_size
                    )
            self.history[stage] = result
            self._active = current
            validation = self._validation()
            if validation is not None:
                if self.validation_max_rows is not None:
                    validation = islice(validation, self.validation_max_rows)
                self.validation_metrics[stage] = evaluate(
                    self.model, validation, batch_size=current.config.batch_size
                )
            if self.checkpoint_dir is not None:
                self._checkpoint(stage)
        return self.history[stages[0]] if len(stages) == 1 else self.history

    def _checkpoint(self, stage: str) -> None:
        """Save a complete reloadable stage before marking it completed."""
        assert self._active is not None and self._active._accelerator is not None
        accelerator = self._active._accelerator
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            assert self.checkpoint_dir is not None
            root = self.checkpoint_dir
            root.mkdir(parents=True, exist_ok=True)
            destination = root / stage
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing stage checkpoint: {destination}"
                )
            with tempfile.TemporaryDirectory(dir=root, prefix=f".{stage}-") as temporary:
                self.model.save_pretrained(temporary)
                os.replace(temporary, destination)
            state = {
                "dataset": str(self.dataset)
                if isinstance(self.dataset, (str, Path))
                else "<in-memory>",
                "dataset_revision": self.dataset_revision,
                "completed_stages": list(self.history),
                "checkpoint": str(destination.resolve()),
                "history": self.history,
                "validation_metrics": self.validation_metrics,
                "prompt_format": self.model.config.prompt_format,
                "head_type": self.model.config.head_type,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=root,
                prefix=".progress-",
                suffix=".json",
                encoding="utf-8",
                delete=False,
            ) as output:
                json.dump(state, output, indent=2, allow_nan=False)
                output.write("\n")
                temporary_manifest = output.name
            os.replace(temporary_manifest, root / "progress.json")
        accelerator.wait_for_everyone()

    def save_pretrained(self, path: str | Path, *, safe_serialization: bool = True) -> None:
        if self._active is None:
            raise RuntimeError("train before saving")
        self._active.save_pretrained(path, safe_serialization=safe_serialization)

    def select_best(self, *, metric: str = "accuracy") -> str:
        """Choose a saved stage on validation only; never use held-out test rows."""
        if metric not in ("accuracy", "nll"):
            raise ConfigError("selection metric must be accuracy or nll")
        if not self.validation_metrics or self.checkpoint_dir is None:
            raise ConfigError("selection needs validation metrics and stage checkpoints")
        self.selected_stage = max(
            self.validation_metrics,
            key=lambda stage: (
                (
                    self.validation_metrics[stage]["accuracy"],
                    -self.validation_metrics[stage]["nll"],
                )
                if metric == "accuracy"
                else (
                    -self.validation_metrics[stage]["nll"],
                    self.validation_metrics[stage]["accuracy"],
                )
            ),
        )
        if self.selected_stage != list(self.history)[-1]:
            self.model = DecisionModel.from_pretrained(self.checkpoint_dir / self.selected_stage)
        return self.selected_stage

    def export(
        self, path: str | Path, *, merge_lora: bool = True, safetensors: bool = True
    ) -> None:
        """Export a full HF checkpoint; QLoRA is merged into a fresh full-precision base."""
        if self._active is None or self._active._accelerator is None:
            raise RuntimeError("train before exporting")
        accelerator = self._active._accelerator
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output = Path(path)
            quantized = getattr(self.model.backbone, "is_loaded_in_4bit", False)
            if quantized and merge_lora:
                with tempfile.TemporaryDirectory() as directory:
                    self.model.save_pretrained(directory, safe_serialization=safetensors)
                    merged = DecisionModel.from_pretrained(directory)
                    merged.save_pretrained(output, safe_serialization=safetensors)
            elif (
                getattr(self.model.backbone, "peft_config", None)
                and not merge_lora
                and not quantized
            ):
                if not self.model.config.base_checkpoint:
                    raise ConfigError("unmerged LoRA needs a recoverable base checkpoint")
                self.model.backbone.save_pretrained(output, safe_serialization=safetensors)
                self.model.tokenizer.save_pretrained(output)
                self.model._save_decision_metadata(
                    output, adapter_base=self.model.config.base_checkpoint
                )
            else:
                self.model.save_pretrained(output, safe_serialization=safetensors)
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(self.dataset)
                if isinstance(self.dataset, (str, Path))
                else "<in-memory>",
                "dataset_revision": self.dataset_revision,
                "model_source": self.model.config.source_checkpoint,
                "source_revision": self.model.config.source_revision,
                "base_model_revision": self.model.config.base_model_revision,
                "prompt_format": self.model.config.prompt_format,
                "head_type": self.model.config.head_type,
                "head_config": self.model.config.head_config,
                "stages": list(self.history),
                "sft_config": asdict(self.sft_config),
                "calibration_config": asdict(self.calibration_config)
                if self.calibration_config
                else None,
                "rlcd_config": asdict(self.rlcd_config) if self.rlcd_config else None,
                "history": self.history,
                "validation_metrics": self.validation_metrics,
                "selected_stage": self.selected_stage,
            }
            (output / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        accelerator.wait_for_everyone()
