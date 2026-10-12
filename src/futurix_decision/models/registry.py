"""Name -> ``FamilyAdapter``. ``load_model`` and the CLI dispatch through it. Imports without torch.

Register a third-party family with ``register_family(MyAdapter())``; then
``load_model(checkpoint, family="my-family")`` and
``futurix-decision evaluate --model-format my-family`` both reach it.

Every per-family lookup here (Hub IDs, aliases, local detection) reads the
registered adapters' own attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.families.base import FamilyAdapter
from futurix_decision.models.families.bosun import BosunAdapter
from futurix_decision.models.families.clef import ClefAdapter
from futurix_decision.models.families.futurix import FuturixAdapter
from futurix_decision.models.families.gliner import GlinerAdapter
from futurix_decision.models.families.kev import KevAdapter
from futurix_decision.models.families.laya import LayaAdapter
from futurix_decision.models.families.letter import DeciderAdapter, EikosAdapter, JPTAdapter
from futurix_decision.models.families.liquid import LiquidAdapter
from futurix_decision.models.families.nox_lux import LuxAdapter, NoxAdapter
from futurix_decision.models.families.openjev import OpenJevAdapter
from futurix_decision.models.families.pplx import PplxAdapter
from futurix_decision.models.sources import read_checkpoint_json

# "standard" and "native" are CLI --model-format values with their own meaning.
_RESERVED = frozenset({"standard", "native"})
_FAMILIES: dict[str, FamilyAdapter] = {}

# Files of decision releases no registered family loads. A directory holding one
# is never loaded as a plain LM.
FOREIGN_DECISION_FILES = ("judge_config.json", "quyet_config.json")


def register_family(adapter: FamilyAdapter) -> None:
    """Make ``adapter`` loadable by its name. Names, Hub IDs, and aliases must be new."""
    if not isinstance(adapter, FamilyAdapter):
        raise ConfigError("register_family needs a FamilyAdapter instance")
    name = getattr(adapter, "name", None)
    if not isinstance(name, str) or not name or name in _RESERVED:
        raise ConfigError(f"invalid family name: {name!r}")
    if name in _FAMILIES:
        raise ConfigError(f"family {name!r} is already registered")
    for checkpoint in (*adapter.checkpoints, *adapter.unsupported_checkpoints):
        owner = family_for_checkpoint(checkpoint)
        if owner is not None:
            raise ConfigError(f"{checkpoint} already belongs to family {owner.name!r}")
    for alias, checkpoint in adapter.aliases.items():
        if adapter.checkpoints and checkpoint not in adapter.checkpoints:
            raise ConfigError(f"alias {alias!r} names {checkpoint}, which {name} does not load")
        if resolve_alias(alias) != alias:
            raise ConfigError(f"alias {alias!r} is already registered")
    _FAMILIES[name] = adapter


def get_family(name: str) -> FamilyAdapter:
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ConfigError(
            f"unknown decision-model family {name!r}; registered: {', '.join(_FAMILIES)}"
        ) from None


def registered_families() -> tuple[FamilyAdapter, ...]:
    return tuple(_FAMILIES.values())


def family_for_checkpoint(checkpoint: str) -> FamilyAdapter | None:
    """The adapter that lists ``checkpoint`` among its supported or unsupported Hub IDs."""
    return next(
        (
            adapter
            for adapter in _FAMILIES.values()
            if checkpoint in adapter.checkpoints or checkpoint in adapter.unsupported_checkpoints
        ),
        None,
    )


def resolve_alias(name: str) -> str:
    """The Hub ID a registered alias such as ``"kev"`` stands for, or ``name`` itself."""
    for adapter in _FAMILIES.values():
        if name in adapter.aliases:
            return adapter.aliases[name]
    return name


def infer_local_family(directory: Path) -> str | None:
    """Recognize publisher copies and SDK exports without depending on their folder name.

    An SDK export's ``futurix_native.json`` names its family. Otherwise the first
    registered family whose ``detect`` accepts ``directory`` wins, and failing
    that, a family whose Hub ID ends in the directory's name.
    """
    manifest = directory / "futurix_native.json"
    if manifest.is_file():
        family = read_checkpoint_json(manifest).get("family")
        if family not in _FAMILIES:
            raise CheckpointError(f"unknown native family in {manifest}: {family!r}")
        return str(family)
    detected = _detect(directory)
    if detected is not None:
        return detected.name
    return next(
        (
            adapter.name
            for adapter in _FAMILIES.values()
            if any(
                directory.name == checkpoint.rsplit("/", 1)[-1]
                for checkpoint in adapter.checkpoints
            )
        ),
        None,
    )


def _detect(directory: Path) -> FamilyAdapter | None:
    return next((adapter for adapter in _FAMILIES.values() if adapter.detect(directory)), None)


def reject_native_checkpoint(checkpoint: str | Path, metadata: dict[str, Any]) -> None:
    """Prevent a seemingly successful load which destroys a native decision head.

    ``metadata`` is the checkpoint's ``decision_config.json``, or ``{}``.
    """
    if metadata.get("architecture") == "futurix-decision":
        return
    family = family_for_checkpoint(str(checkpoint).rstrip("/"))
    if family is not None:
        raise CheckpointError(
            f"{checkpoint} is a {family.name} decision checkpoint; use "
            f"load_model(..., family={family.name!r}) to keep the publisher's native readout. "
            "The generic LM loader would replace its trained decision contract."
        )
    if metadata:
        raise CheckpointError(
            "decision_config.json belongs to a non-Futurix checkpoint; "
            "the generic loader cannot preserve its native decision readout"
        )
    path = Path(checkpoint)
    if not path.is_dir():
        return
    detected = _detect(path)
    if detected is not None:
        raise CheckpointError(
            f"local checkpoint is a {detected.name} release with a native decision readout; "
            f"load it with load_model(..., family={detected.name!r}). The generic LM loader "
            "would replace its trained decision contract"
        )
    if any((path / filename).is_file() for filename in FOREIGN_DECISION_FILES):
        raise CheckpointError(
            "local checkpoint contains a native decision configuration or head; "
            "the generic LM loader would replace its trained decision contract"
        )


for _adapter in (
    FuturixAdapter(),
    BosunAdapter(),
    DeciderAdapter(),
    JPTAdapter(),
    EikosAdapter(),
    KevAdapter(),
    LayaAdapter(),
    GlinerAdapter(),
    ClefAdapter(),
    PplxAdapter(),
    OpenJevAdapter(),
    NoxAdapter(),
    LuxAdapter(),
    LiquidAdapter(),
):
    register_family(_adapter)
