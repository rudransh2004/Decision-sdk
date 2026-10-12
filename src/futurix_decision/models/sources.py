"""Resolve local checkpoints and reject incomplete decision-model artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from futurix_decision.errors import CheckpointError
from futurix_decision.models.families import FAMILIES

# Short names that ``load_model`` and ``NativeDecisionModel`` accept for Hub IDs.
ALIASES = {
    "bosun-0.6b": "Hanno-Labs/bosun-v3.1-0.6b",
    "bosun-1.7b": "Hanno-Labs/bosun-v3.1-1.7b",
    "kev": "jaredpalmer/kev-9b",
    "laya": "convaiinnovations/laya",
    "laya-multilingual": "convaiinnovations/laya-multilingual",
    "gliner": "fastino/GLiNER2.5-Decide",
    "clef": "Cloudflare/clef",
    "clef-flash": "Cloudflare/clef-flash",
    "pplx-decider": "perplexity-ai/pplx-decider-v1.1-27b",
    "openjev": "openjev/openjev",
    "openjev-flash": "openjev/OpenJev-Flash-9B",
    "nox": "vllm-sr/Decision-2.0-Nox-4B",
    "lux": "vllm-sr/Decision-2.0-Lux-9B",
    "liquid-d1": "LiquidAI/d1-3B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


def checkpoint_source(checkpoint: str | Path) -> tuple[str, bool]:
    """Keep Hub IDs as IDs, but never send an explicit missing disk path to the Hub."""
    value = str(checkpoint)
    path = Path(value).expanduser()
    if path.is_dir():
        return str(path), True
    if path.exists():
        raise NotADirectoryError(f"checkpoint must be a directory: {path}")
    if isinstance(checkpoint, Path) or value.startswith(("/", "./", "../", "~/")):
        raise FileNotFoundError(f"local checkpoint directory does not exist: {path}")
    return value, False


def tokenizer_kwargs(hf_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The ``from_pretrained`` options that also apply to tokenizers and configs."""
    keys = ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
    return {key: hf_kwargs[key] for key in keys if key in hf_kwargs}


def hub_kwargs(hf_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The ``from_pretrained`` options that ``hf_hub_download`` accepts."""
    keys = ("revision", "token", "cache_dir", "local_files_only")
    return {key: hf_kwargs[key] for key in keys if key in hf_kwargs}


def read_checkpoint_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid checkpoint metadata: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint metadata must be an object: {path}")
    return value


def infer_local_family(directory: Path) -> str | None:
    """Recognize publisher copies and SDK exports without depending on their folder name."""
    manifest = directory / "futurix_native.json"
    if manifest.is_file():
        family = read_checkpoint_json(manifest).get("family")
        if family not in {
            "kev",
            "laya",
            "gliner",
            "clef",
            "pplx-decider",
            "openjev",
            "nox",
            "lux",
            "liquid-d1",
        }:
            raise CheckpointError(f"unknown native family in {manifest}: {family!r}")
        return family
    decision_config = directory / "decision_config.json"
    if decision_config.is_file():
        metadata = read_checkpoint_json(decision_config)
        if metadata.get("architecture") == "futurix-decision":
            return "futurix"
        if metadata.get("prompt_version") == "letter-v1-semif":
            return "eikos"
        if metadata.get("format_version") == 1 and "attention_mode" in metadata:
            return "pplx-decider"
    config = directory / "config.json"
    if config.is_file():
        release = read_checkpoint_json(config)
        if release.get("model_type") == "bosun":
            return "bosun"
        if release.get("decision_format") == "vllm-sr-decision":
            name = release.get("model_name", "")
            if isinstance(name, str):
                if "-Nox-" in name:
                    return "nox"
                if "-Lux-" in name:
                    return "lux"
        auto_map = release.get("auto_map")
        if isinstance(auto_map, dict) and auto_map.get("AutoModel") == "modeling_d1.D1Model":
            return "liquid-d1"
    for marker, family in (
        ("joint_head.safetensors", "clef"),
        ("joint_schema_model.py", "clef"),
        ("joint_head_config.json", "clef"),
        ("readout.safetensors", "pplx-decider"),
        ("decider_config.json", "decider"),
        ("decision_embeddings.safetensors", "bosun"),
        ("head.pt", "kev"),
        ("rl_agent_config.json", "laya"),
        ("encoder_config/config.json", "gliner"),
        ("decision_head.safetensors", "futurix"),
    ):
        if (directory / marker).is_file():
            return family
    card = directory / "README.md"
    if card.is_file() and "# JPT-" in card.read_text(encoding="utf-8", errors="replace")[:4096]:
        return "jpt"
    for family in FAMILIES:
        if any(
            directory.name == checkpoint.rsplit("/", 1)[-1]
            for checkpoint in family.supported_checkpoints
        ):
            return (
                family.name
                if family.name != "encoder"
                else ("gliner" if directory.name == "GLiNER2.5-Decide" else "laya")
            )
    return None


class CheckpointFiles:
    """Collect the files a local checkpoint directory is missing, then raise once.

    Adapters' ``requires`` methods call ``require``, ``one_of``, ``tokenizer``,
    and ``weights`` for each file their loader reads, then ``check``. This runs
    on disk paths and on downloaded Hub snapshots; publisher loaders remain
    responsible for architecture-specific config and tensor validation.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.missing: list[str] = []

    def _inside(self, relative: str) -> None:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise CheckpointError(f"checkpoint file must be inside its directory: {relative!r}")

    def has(self, relative: str) -> bool:
        return (self.directory / relative).is_file()

    def require(self, *relatives: str) -> None:
        for relative in relatives:
            self._inside(relative)
            if not self.has(relative):
                self.missing.append(relative)

    def one_of(self, *alternatives: str) -> None:
        for relative in alternatives:
            self._inside(relative)
        if not any(self.has(relative) for relative in alternatives):
            self.missing.append(" or ".join(alternatives))

    def tokenizer(self) -> None:
        self.one_of("tokenizer.json", "tokenizer.model", "vocab.json")

    def weights(self) -> None:
        """Every shard a sharded index names, or one unsharded weight file."""
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = self.directory / index_name
            if index.is_file():
                files = read_checkpoint_json(index).get("weight_map")
                if (
                    not isinstance(files, dict)
                    or not files
                    or not all(
                        isinstance(filename, str) and filename for filename in files.values()
                    )
                ):
                    raise CheckpointError(f"invalid weight_map in {index}")
                self.require(*sorted(set(files.values())))
                return
        self.one_of("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")

    def check(self, family: str, *, source: str | None = None) -> None:
        if self.missing:
            raise CheckpointError(
                f"incomplete {family} checkpoint {source or str(self.directory)!r}; "
                f"missing: {', '.join(self.missing)}"
            )
