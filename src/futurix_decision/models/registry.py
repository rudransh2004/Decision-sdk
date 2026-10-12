"""Name -> ``FamilyAdapter``. ``load_model`` and the CLI dispatch through it. Imports without torch.

Register a third-party family with ``register_family(MyAdapter())``; then
``load_model(checkpoint, family="my-family")`` and
``futurix-decision evaluate --model-format my-family`` both reach it.
"""

from __future__ import annotations

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import FamilyAdapter
from futurix_decision.models.families.bosun import BosunAdapter
from futurix_decision.models.families.clef import ClefAdapter
from futurix_decision.models.families.futurix import FuturixAdapter
from futurix_decision.models.families.gliner import GlinerAdapter
from futurix_decision.models.families.kev import KevAdapter
from futurix_decision.models.families.laya import LayaAdapter
from futurix_decision.models.families.letter import LetterAdapter
from futurix_decision.models.families.liquid import LiquidAdapter
from futurix_decision.models.families.nox_lux import NoxLuxAdapter
from futurix_decision.models.families.openjev import OpenJevAdapter
from futurix_decision.models.families.pplx import PplxAdapter

# "standard" and "native" are CLI --model-format values with their own meaning.
_RESERVED = frozenset({"standard", "native"})
_FAMILIES: dict[str, FamilyAdapter] = {}


def register_family(adapter: FamilyAdapter) -> None:
    """Make ``adapter`` loadable by its name. Names must be new."""
    if not isinstance(adapter, FamilyAdapter):
        raise ConfigError("register_family needs a FamilyAdapter instance")
    name = getattr(adapter, "name", None)
    if not isinstance(name, str) or not name or name in _RESERVED:
        raise ConfigError(f"invalid family name: {name!r}")
    if name in _FAMILIES:
        raise ConfigError(f"family {name!r} is already registered")
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
    """The adapter that lists ``checkpoint`` among its Hub IDs, if any."""
    return next(
        (adapter for adapter in _FAMILIES.values() if checkpoint in adapter.checkpoints), None
    )


for _adapter in (
    FuturixAdapter(),
    BosunAdapter(),
    LetterAdapter(
        "decider",
        "futurix_decision.models.decider:DeciderDecisionModel",
        markers=("decider_config.json",),
    ),
    LetterAdapter("jpt", "futurix_decision.models.jpt:JPTDecisionModel", markers=("README.md",)),
    LetterAdapter(
        "eikos",
        "futurix_decision.models.eikos:EikosDecisionModel",
        markers=("decision_config.json", "calib.json"),
    ),
    KevAdapter(),
    LayaAdapter(),
    GlinerAdapter(),
    ClefAdapter(),
    PplxAdapter(),
    OpenJevAdapter(),
    NoxLuxAdapter("nox"),
    NoxLuxAdapter("lux"),
    LiquidAdapter(),
):
    register_family(_adapter)
