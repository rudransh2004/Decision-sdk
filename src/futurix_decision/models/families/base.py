"""The interface each decision-model family implements. Imports without torch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from futurix_decision.errors import ConfigError
from futurix_decision.models.protocol import DecisionBackend

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel


class FamilyAdapter(ABC):
    """Loads, checks, and optionally fine-tunes one family of decision checkpoints.

    ``load_model`` and the CLI find adapters by ``name`` in the registry; see
    ``futurix_decision.models.registry.register_family``. Subclasses set
    ``name`` and implement ``load``; every other attribute and method has a
    working default.

    The class attributes are the family's only record of its releases and
    capabilities: the registry derives Hub ID lookup, aliases, and local
    detection from them, and a test checks the README support matrix against them.
    """

    name: str
    """Registry key: the ``family=`` argument and the CLI ``--model-format`` value."""

    checkpoints: tuple[str, ...] = ()
    """Hub IDs ``load_model`` recognizes by name and loads. Empty means any Hub ID,
    when the caller passes ``family=name``."""

    unsupported_checkpoints: tuple[str, ...] = ()
    """Published Hub IDs of this family that the SDK cannot load yet. Loaders
    recognize them only to refuse them, so no loader mistakes one for a plain LM."""

    aliases: Mapping[str, str] = MappingProxyType({})
    """Short names for Hub IDs, for example ``{"kev": "jaredpalmer/kev-9b"}``."""

    markers: tuple[str, ...] = ()
    """Files, relative to a checkpoint directory, that only this family's releases
    contain. ``detect`` returns ``True`` when any of them exists."""

    extra: str | None = None
    """The pip extra that installs this family's dependencies, for example ``"bosun"``."""

    trainable: bool = False
    """Whether the SDK trains this family on labeled decision data, through
    ``DecisionTrainer`` or the family's own ``finetune``."""

    vllm: Literal["direct", "export", "server"] | None = None
    """How the family reaches vLLM: ``"direct"`` checkpoints load in ``VLLMEngine``;
    ``"export"`` ones do after ``futurix-decision export``; ``"server"`` means the
    publisher's own vLLM-backed System One server. ``None`` means no vLLM path."""

    @abstractmethod
    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionBackend:
        """Load a Hub ID or local directory and return a ready backend.

        ``device`` places the model, for example ``"cuda:0"``. ``kwargs`` are
        the family's own options, passed through from ``load_model`` or the CLI.
        """

    def detect(self, directory: Path) -> bool:
        """Whether the local ``directory`` holds one of this family's checkpoints.

        ``load_model`` asks each registered family in turn when the caller
        passes no ``family``. The default looks for ``markers``; override it to
        read a config file instead. Return ``False`` when unsure.
        """
        return any((directory / marker).is_file() for marker in self.markers)

    def model_class(self) -> type[DecisionModel] | None:
        """The ``DecisionModel`` subclass that reads this family's saved checkpoints.

        ``DecisionModel.from_pretrained`` asks each family for it to find the
        class whose prompt format a saved checkpoint names. ``None`` means the
        family loads no ``DecisionModel`` subclass of its own.
        """
        return None

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
