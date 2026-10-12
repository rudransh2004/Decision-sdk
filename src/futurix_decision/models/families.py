"""Known upstream decision checkpoints and their *implemented* SDK contracts.

An upstream decision checkpoint must never be treated as a plain LM: doing so
silently replaces its trained readout with freshly initialized Futurix tokens.
This list is deliberately explicit; unknown plain LMs remain valid bases for
training a new Futurix model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFamily:
    """Flags apply to implemented variants within the family, not every Hub ID."""
    name: str
    checkpoints: tuple[str, ...]
    supported_checkpoints: tuple[str, ...] = ()
    inference: bool = False
    training: bool = False
    vllm_export: bool = False

    def supports(self, checkpoint: str, capability: str) -> bool:
        if capability not in ("inference", "training", "vllm_export"):
            raise ValueError(f"unknown decision capability: {capability}")
        return checkpoint in self.supported_checkpoints and getattr(self, capability)


FAMILIES = (
    ModelFamily(
        "bosun",
        ("Hanno-Labs/bosun-v3.1-1.7b", "Hanno-Labs/bosun-v3.1-0.6b"),
        supported_checkpoints=("Hanno-Labs/bosun-v3.1-1.7b", "Hanno-Labs/bosun-v3.1-0.6b"),
        inference=True, training=True, vllm_export=True,
    ),
    ModelFamily("kev", (
        "jaredpalmer/kev-27b", "jaredpalmer/kev-9b",
        "jaredpalmer/kev-4b", "jaredpalmer/kev-0.8b",
    ), supported_checkpoints=("jaredpalmer/kev-9b",), inference=True, training=True),
    ModelFamily("clef", (
        "Cloudflare/clef", "Cloudflare/clef-flash", "Cloudflare/clef-omni",
    ), supported_checkpoints=("Cloudflare/clef", "Cloudflare/clef-flash"),
                inference=True, training=True),
    ModelFamily("pplx-decider", ("perplexity-ai/pplx-decider-v1.1-27b",),
                supported_checkpoints=("perplexity-ai/pplx-decider-v1.1-27b",),
                inference=True, training=True),
    ModelFamily(
        "jpt", ("kirp/jpt-9b", "kirp/jpt-4b", "kirp/jpt-35b-a3b"),
        supported_checkpoints=("kirp/jpt-4b", "kirp/jpt-9b"),
        inference=True, training=True, vllm_export=True,
    ),
    ModelFamily(
        "eikos", ("caiovicentino1/Eikos-27B", "caiovicentino1/Eikos-4B"),
        supported_checkpoints=("caiovicentino1/Eikos-4B", "caiovicentino1/Eikos-27B"),
        inference=True, training=True, vllm_export=True,
    ),
    ModelFamily("openjev", ("openjev/openjev", "openjev/OpenJev-Flash-9B"),
                supported_checkpoints=("openjev/openjev", "openjev/OpenJev-Flash-9B"),
                inference=True),
    ModelFamily(
        "decider", ("Mapika/decider-4b",),
        supported_checkpoints=("Mapika/decider-4b",),
        inference=True, training=True, vllm_export=True,
    ),
    ModelFamily("liquid-d1", ("LiquidAI/d1-3B", "LiquidAI/d1-omni-600M"),
                supported_checkpoints=("LiquidAI/d1-3B",), inference=True),
    ModelFamily("nox", ("vllm-sr/Decision-1.0-Nox-4B", "vllm-sr/Decision-2.0-Nox-4B"),
                supported_checkpoints=("vllm-sr/Decision-1.0-Nox-4B",
                                       "vllm-sr/Decision-2.0-Nox-4B"), inference=True),
    ModelFamily("lux", ("vllm-sr/Decision-1.0-Lux-9B", "vllm-sr/Decision-2.0-Lux-9B"),
                supported_checkpoints=("vllm-sr/Decision-1.0-Lux-9B",
                                       "vllm-sr/Decision-2.0-Lux-9B"), inference=True),
    ModelFamily("encoder", (
        "convaiinnovations/laya", "convaiinnovations/laya-multilingual",
        "fastino/GLiNER2.5-Decide",
    ), supported_checkpoints=(
        "convaiinnovations/laya", "convaiinnovations/laya-multilingual",
        "fastino/GLiNER2.5-Decide",
    ), inference=True, training=True),
)


def identify_family(checkpoint: str | Path) -> ModelFamily | None:
    """Match a Hub ID or a local clone named after a known upstream release."""
    value = str(checkpoint).rstrip("/")
    for family in FAMILIES:
        if value in family.checkpoints:
            return family
    return None


def reject_native_checkpoint(checkpoint: str | Path, metadata: dict) -> None:
    """Prevent a seemingly successful load which destroys a native decision head."""
    if metadata.get("architecture") == "futurix-decision":
        return
    family = identify_family(checkpoint)
    if family is not None:
        entrypoint = {
            "bosun": "DecisionModel.from_bosun",
            "decider": "DecisionModel.from_decider",
            "jpt": "DecisionModel.from_jpt (4B/9B)",
            "eikos": "DecisionModel.from_eikos",
        }.get(family.name, "load_model (the publisher's native decision runtime)")
        raise ValueError(
            f"{checkpoint} is a {family.name} decision checkpoint; use {entrypoint}. "
            "The generic LM loader would replace its trained decision contract."
        )
    if metadata:
        raise ValueError(
            "decision_config.json belongs to a non-Futurix checkpoint; "
            "the generic loader cannot preserve its native decision readout"
        )
    path = Path(checkpoint)
    if path.is_dir() and any(
        (path / filename).is_file()
        for filename in (
            "decider_config.json", "joint_head.safetensors", "readout.safetensors",
            "head.pt", "judge_config.json", "rl_agent_config.json", "quyet_config.json",
        )
    ):
        raise ValueError(
            "local checkpoint contains a native decision configuration or head; "
            "the generic LM loader would replace its trained decision contract"
        )
    card = path / "README.md"
    if path.is_dir() and card.is_file() and "# JPT-" in card.read_text(
        encoding="utf-8", errors="replace"
    )[:4096]:
        raise ValueError(
            "local JPT checkpoint requires DecisionModel.from_jpt to preserve its letter readout"
        )
