"""The interface each decision-model family implements. Imports without torch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families import FAMILIES
from futurix_decision.models.protocol import DecisionBackend


class FamilyAdapter(ABC):
    """Loads, checks, and optionally fine-tunes one family of decision checkpoints.

    ``load_model`` and the CLI find adapters by ``name`` in the registry; see
    ``futurix_decision.models.registry.register_family``. Subclasses set the
    three attributes and implement ``load``. ``requires``, ``finetune``, and
    ``training_kwargs`` have working defaults.
    """

    name: str
    """Registry key: the ``family=`` argument and the CLI ``--model-format`` value."""

    checkpoints: tuple[str, ...] = ()
    """Hub IDs ``load_model`` recognizes by name. Empty means any Hub ID, when the
    caller passes ``family=name``."""

    extra: str | None = None
    """The pip extra that installs this family's dependencies, for example ``"bosun"``."""

    @abstractmethod
    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionBackend:
        """Load a Hub ID or local directory and return a ready backend.

        ``device`` places the model, for example ``"cuda:0"``. ``kwargs`` are
        the family's own options, passed through from ``load_model`` or the CLI.
        """

    def requires(self, path: Path) -> None:
        """Raise ``CheckpointError`` if a local checkpoint can't be loaded.

        ``load_model`` calls this before ``load`` for local directories. The
        default checks nothing.
        """
        return None

    def finetune(
        self, model: DecisionBackend, dataset: Any, output_dir: str | Path, **kwargs: Any
    ) -> DecisionBackend:
        """Train ``model`` in the family's own format, save it, and return it reloaded.

        Only families with their own trainer implement this. The others train
        through ``DecisionTrainer``.
        """
        raise NotImplementedError(f"{self.name} has no native fine-tuning; use DecisionTrainer")

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        """Check a training config's ``model:`` section and return its ``load`` options.

        ``method`` is the fine-tuning method: ``"full"``, ``"lora"``, or
        ``"qlora"``. The default accepts a section that sets no family options.
        """
        options = sorted(set(section) & _FAMILY_OPTIONS)
        if options:
            raise ConfigError(f"{self.name} does not accept model options {options}")
        return {}


# Keys of a training config's ``model:`` section that adapters interpret. The CLI
# itself reads ``name``, ``format``, ``revision``, ``strategy``, and ``attention_mode``.
_FAMILY_OPTIONS = frozenset(
    {
        "decision_tokens",
        "head_config",
        "head_type",
        "max_sequence_length",
        "prompt_format",
        "seed",
        "temperature",
    }
)


def reject_futurix_options(kwargs: Mapping[str, Any], family: str) -> None:
    """Refuse ``load_model`` options that only configure Futurix readouts."""
    for key in ("head_type", "train_backbone"):
        if key in kwargs:
            raise ConfigError(
                f"{key} applies to Futurix checkpoints and convert=True, not {family} releases"
            )


def supported_checkpoints(family: str) -> tuple[str, ...]:
    """The Hub IDs the family table marks as implemented for ``family``."""
    return next(entry.supported_checkpoints for entry in FAMILIES if entry.name == family)
