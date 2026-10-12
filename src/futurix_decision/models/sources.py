"""Resolve local checkpoints and reject incomplete decision-model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.families import FAMILIES


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


def require_checkpoint_files(
    directory: Path,
    family: str,
    *,
    source: str | None = None,
) -> None:
    """Check the native readout and complete local weight set before loading.

    This runs on disk paths and on downloaded Hub snapshots. Publisher loaders
    remain responsible for architecture-specific config and tensor validation.
    """
    missing: list[str] = []

    def safe(relative: str) -> None:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise CheckpointError(f"checkpoint file must be inside its directory: {relative!r}")

    def require(relative: str) -> None:
        safe(relative)
        if not (directory / relative).is_file():
            missing.append(relative)

    def one_of(*alternatives: str) -> None:
        for filename in alternatives:
            safe(filename)
        if not any((directory / name).is_file() for name in alternatives):
            missing.append(" or ".join(alternatives))

    def weights() -> None:
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = directory / index_name
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
                for filename in sorted(set(files.values())):
                    require(filename)
                return
        one_of("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")

    if family == "clef":
        for filename in (
            "config.json",
            "joint_schema_model.py",
            "joint_head_config.json",
            "joint_head.safetensors",
        ):
            require(filename)
        one_of("tokenizer.json", "tokenizer.model", "vocab.json")
        weights()
    elif family == "pplx-decider":
        for filename in ("config.json", "decision_config.json", "readout.safetensors"):
            require(filename)
        one_of("tokenizer.json", "tokenizer.model", "vocab.json")
        weights()
    elif family == "kev":
        require("head.pt")
        if (directory / "adapter_config.json").is_file():
            one_of("adapter_model.safetensors", "adapter_model.bin")
        else:
            require("config.json")
            weights()
    elif family == "laya":
        require("rl_agent_config.json")
        require("model.safetensors")
    elif family == "gliner":
        require("config.json")
        require("encoder_config/config.json")
        one_of("model.safetensors", "pytorch_model.bin")
    elif family == "bosun":
        require("config.json")
        if (directory / "config.json").is_file():
            config = read_checkpoint_json(directory / "config.json")
            require(
                str(config.get("decision_embeddings_file") or "decision_embeddings.safetensors")
            )
            adapter = str(config.get("adapter_subfolder") or "adapter")
            tokenizer = str(config.get("tokenizer_subfolder") or "tokenizer")
            require(f"{adapter}/adapter_config.json")
            one_of(f"{adapter}/adapter_model.safetensors", f"{adapter}/adapter_model.bin")
            one_of(f"{tokenizer}/tokenizer.json", f"{tokenizer}/vocab.json")
    elif family == "decider":
        require("decider_config.json")
        require("config.json")
        one_of("tokenizer.json", "tokenizer.model", "vocab.json")
        weights()
    elif family == "eikos":
        require("decision_config.json")
        require("calib.json")
        require("config.json")
        one_of("tokenizer.json", "tokenizer.model", "vocab.json")
        weights()
    elif family == "jpt":
        require("README.md")
        require("config.json")
        one_of("tokenizer.json", "tokenizer.model", "vocab.json")
        weights()
    elif family == "futurix":
        require("decision_config.json")
        if (directory / "decision_config.json").is_file():
            metadata = read_checkpoint_json(directory / "decision_config.json")
            if metadata.get("architecture") != "futurix-decision":
                raise CheckpointError(f"not a Futurix decision checkpoint: {directory}")
            if metadata.get("head_type", "tokens") != "tokens":
                require("decision_head.safetensors")
        if (directory / "adapter_config.json").is_file():
            one_of("adapter_model.safetensors", "adapter_model.bin")
        else:
            require("config.json")
            weights()
        one_of("tokenizer.json", "tokenizer.model", "vocab.json")
    elif family in ("nox", "lux"):
        require("config.json")
        if (directory / "config.json").is_file():
            config = read_checkpoint_json(directory / "config.json")
            if config.get("decision_format") != "vllm-sr-decision":
                raise CheckpointError(f"{directory} is not a vLLM Semantic Router decision release")
            version = config.get("format_version")
            if version not in (1, 2):
                raise CheckpointError(
                    f"unsupported Decision release format {version!r}: {directory}"
                )
            for field in ("model_config", "manifest") if version == 2 else ("model_config",):
                filename = config.get(field)
                if not isinstance(filename, str) or not filename:
                    raise CheckpointError(
                        f"Decision {version} checkpoint is missing {field}: {directory}"
                    )
                require(filename)
            head = config.get("decision_weights")
            backbone = config.get("backbone")
            tokenizer = config.get("tokenizer")
            if (
                not isinstance(head, dict)
                or not isinstance(backbone, dict)
                or not isinstance(tokenizer, dict)
            ):
                raise CheckpointError(
                    f"Decision {version} checkpoint is missing "
                    f"head/backbone/tokenizer metadata: {directory}"
                )
            for field, section in (
                ("decision_head", head),
                ("config", backbone),
                ("index", backbone),
                ("json", tokenizer),
                ("config", tokenizer),
                ("chat_template", tokenizer),
            ):
                filename = section.get(field)
                if not isinstance(filename, str) or not filename:
                    raise CheckpointError(
                        f"Decision {version} checkpoint is missing {field} metadata: {directory}"
                    )
                require(filename)
            filenames = backbone.get("weights")
            if (
                not isinstance(filenames, list)
                or not filenames
                or not all(isinstance(filename, str) and filename for filename in filenames)
            ):
                raise CheckpointError(
                    f"Decision {version} checkpoint has no backbone shard list: {directory}"
                )
            for filename in filenames:
                require(filename)
            index = directory / backbone["index"]
            if index.is_file():
                mapping = read_checkpoint_json(index).get("weight_map")
                if not isinstance(mapping, dict) or not mapping:
                    raise CheckpointError(f"invalid backbone weight_map in {index}")
                for filename in set(mapping.values()):
                    if not isinstance(filename, str) or not filename:
                        raise CheckpointError(f"invalid backbone shard in {index}")
                    relative = Path(filename)
                    if relative.parts[:1] != Path(backbone["index"]).parts[:1]:
                        relative = Path(backbone["index"]).parent / relative
                    require(str(relative))
            calibration = config.get("calibration")
            if isinstance(calibration, dict) and calibration.get("temperature_file"):
                filename = calibration["temperature_file"]
                if not isinstance(filename, str):
                    raise CheckpointError(f"Decision calibration filename is invalid: {directory}")
                require(filename)
    elif family == "liquid-d1":
        for filename in (
            "config.json",
            "modeling_d1.py",
            "lfm2_vl.py",
            "runner.py",
            "hybrid.py",
            "api.py",
            "prompt.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "chat_template.jinja",
        ):
            require(filename)
        weights()
    else:
        raise ConfigError(f"unsupported decision-model family: {family}")
    if missing:
        raise CheckpointError(
            f"incomplete {family} checkpoint {source or str(directory)!r}; "
            f"missing: {', '.join(missing)}"
        )
