"""Resolve local checkpoints and reject incomplete decision-model artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from futurix_decision.errors import CheckpointError


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


def optional_json(directory: Path, filename: str) -> dict[str, Any]:
    """``filename`` in ``directory`` as an object, or ``{}`` when the file is absent."""
    path = directory / filename
    return read_checkpoint_json(path) if path.is_file() else {}


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
