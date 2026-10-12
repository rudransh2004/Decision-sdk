"""Local/Hub checkpoint metadata and YAML pipeline settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from futurix_decision.errors import ConfigError, MissingDependencyError


def load_decision_config(checkpoint: str | Path, **hub_kwargs: Any) -> dict[str, Any]:
    path = Path(checkpoint)
    metadata = path / "decision_config.json"
    if metadata.is_file():
        return json.loads(metadata.read_text(encoding="utf-8"))
    if path.exists():
        return {}
    try:
        filename = hf_hub_download(
            repo_id=str(checkpoint),
            filename="decision_config.json",
            **{
                key: value
                for key, value in hub_kwargs.items()
                if key in ("revision", "token", "cache_dir", "local_files_only")
            },
        )
    except EntryNotFoundError:
        return {}
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise MissingDependencyError("YAML configuration requires pip install pyyaml") from exc
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ConfigError("configuration must be a YAML mapping")
    return parsed
