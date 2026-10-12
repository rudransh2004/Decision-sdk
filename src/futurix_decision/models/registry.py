"""Name -> ``FamilyAdapter``. ``load_model`` and the CLI dispatch through it. Imports without torch.

Register a third-party family with ``register_family(MyAdapter())``; then
``load_model(checkpoint, family="my-family")`` and
``futurix-decision evaluate --model-format my-family`` both reach it.
"""

from __future__ import annotations

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import FamilyAdapter
from futurix_decision.models.families.bosun import BosunAdapter
from futurix_decision.models.families.futurix import FuturixAdapter
from futurix_decision.models.families.letter import LetterAdapter
from futurix_decision.models.families.publisher import PublisherAdapter

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
    LetterAdapter("decider", "futurix_decision.models.decider:DeciderDecisionModel"),
    LetterAdapter("jpt", "futurix_decision.models.jpt:JPTDecisionModel"),
    LetterAdapter("eikos", "futurix_decision.models.eikos:EikosDecisionModel"),
    PublisherAdapter("kev", extra="kev-native"),
    PublisherAdapter("laya", extra="laya"),
    PublisherAdapter("gliner", extra="gliner"),
    PublisherAdapter("clef", extra="clef"),
    PublisherAdapter("pplx-decider", extra="pplx-decider"),
    PublisherAdapter("openjev", extra=None, local_files=False),
    PublisherAdapter("nox", extra="nox-lux"),
    PublisherAdapter("lux", extra="nox-lux"),
    PublisherAdapter("liquid-d1", extra="liquid-d1"),
):
    register_family(_adapter)
