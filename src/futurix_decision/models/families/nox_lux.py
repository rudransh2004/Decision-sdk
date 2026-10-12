"""Nox and Lux: vLLM Semantic Router's candidate heads, run through the release's remote code."""

from __future__ import annotations

from pathlib import Path

from futurix_decision.errors import CheckpointError
from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles, read_checkpoint_json


class NoxLuxAdapter(PublisherAdapter):
    """``name`` is ``"nox"`` or ``"lux"``; both read the ``vllm-sr-decision`` package format."""

    extra = "nox-lux"
    readout = "candidate-head"
    remote_code = True
    transformers = (5, 17)

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()

    def files(self, files: CheckpointFiles) -> None:
        files.require("config.json")
        if not files.has("config.json"):
            return
        directory = files.directory
        config = read_checkpoint_json(directory / "config.json")
        if config.get("decision_format") != "vllm-sr-decision":
            raise CheckpointError(f"{directory} is not a vLLM Semantic Router decision release")
        version = config.get("format_version")
        if version not in (1, 2):
            raise CheckpointError(f"unsupported Decision release format {version!r}: {directory}")
        for field in ("model_config", "manifest") if version == 2 else ("model_config",):
            filename = config.get(field)
            if not isinstance(filename, str) or not filename:
                raise CheckpointError(
                    f"Decision {version} checkpoint is missing {field}: {directory}"
                )
            files.require(filename)
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
            files.require(filename)
        filenames = backbone.get("weights")
        if (
            not isinstance(filenames, list)
            or not filenames
            or not all(isinstance(filename, str) and filename for filename in filenames)
        ):
            raise CheckpointError(
                f"Decision {version} checkpoint has no backbone shard list: {directory}"
            )
        files.require(*filenames)
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
                files.require(str(relative))
        calibration = config.get("calibration")
        if isinstance(calibration, dict) and calibration.get("temperature_file"):
            filename = calibration["temperature_file"]
            if not isinstance(filename, str):
                raise CheckpointError(f"Decision calibration filename is invalid: {directory}")
            files.require(filename)
