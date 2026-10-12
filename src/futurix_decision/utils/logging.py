"""Local metric sinks; Weights & Biases is strictly opt-in."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from futurix_decision.errors import MissingDependencyError

logger = logging.getLogger(__name__)


class TrainingLogger:
    def __init__(
        self, *, stdout: bool = False, jsonl: str | None = None, wandb_project: str | None = None
    ) -> None:
        self.stdout = stdout
        self._stream = None
        self._wandb: Any = None
        if jsonl:
            path = Path(jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("a", encoding="utf-8")
        if wandb_project:
            try:
                import wandb
            except ImportError as exc:
                self.close()
                raise MissingDependencyError(
                    "W&B logging requires pip install futurix-decision[wandb]"
                ) from exc
            self._wandb = wandb.init(project=wandb_project)

    def log(self, values: dict) -> None:
        line = json.dumps(values, allow_nan=False)
        if self.stdout:
            logger.info("%s", line)
        else:
            logger.debug("%s", line)
        if self._stream:
            self._stream.write(line + "\n")
            self._stream.flush()
        if self._wandb:
            self._wandb.log(values, step=values["step"])

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
        if self._wandb:
            self._wandb.finish()
            self._wandb = None
