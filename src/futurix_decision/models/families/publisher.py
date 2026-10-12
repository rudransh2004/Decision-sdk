"""Publisher runtimes: each family runs its publisher's own code through ``NativeDecisionModel``.

A ``PublisherAdapter`` subclass per family says how to load the publisher's
model (``open``), which files a local copy needs (``files``), and how to ask it
questions (its ``runtime_type``). Publisher packages are imported only inside
those methods, so this module imports without torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from futurix_decision._lazy import require_transformers
from futurix_decision.client._transport import DEFAULT_MAX_RETRIES, build_client, check_settings
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.families.base import FamilyAdapter, reject_futurix_options
from futurix_decision.models.protocol import DecisionBackend
from futurix_decision.models.sources import CheckpointFiles, checkpoint_source, read_checkpoint_json

if TYPE_CHECKING:
    from futurix_decision.datasets.schema import DecisionExample, DecisionInput
    from futurix_decision.models.native import NativeDecisionModel

# Hub ID -> publisher runtime. Laya and GLiNER share the "encoder" row of the
# family table but run different code.
PUBLISHER_CHECKPOINTS = {
    "jaredpalmer/kev-9b": "kev",
    "convaiinnovations/laya": "laya",
    "convaiinnovations/laya-multilingual": "laya",
    "fastino/GLiNER2.5-Decide": "gliner",
    "Cloudflare/clef": "clef",
    "Cloudflare/clef-flash": "clef",
    "perplexity-ai/pplx-decider-v1.1-27b": "pplx-decider",
    "openjev/openjev": "openjev",
    "openjev/OpenJev-Flash-9B": "openjev",
    "vllm-sr/Decision-1.0-Nox-4B": "nox",
    "vllm-sr/Decision-2.0-Nox-4B": "nox",
    "vllm-sr/Decision-1.0-Lux-9B": "lux",
    "vllm-sr/Decision-2.0-Lux-9B": "lux",
    "LiquidAI/d1-3B": "liquid-d1",
}

BatchAnswers = list[tuple[Mapping[str, Any], float]]
"""``(answer, per-row latency ms)`` for each row of a batch, in input order."""


class PublisherRuntime:
    """Asks one loaded publisher model System One questions.

    ``native`` is the publisher's object, which ``NativeDecisionModel.runtime``
    exposes. A payload is ``{"model", "state", "questions"}`` plus ``"images"``
    when the caller passed any. The default suits a Transformers remote-code
    model with a keyword-only ``system_one`` method: Nox, Lux, and Liquid d1.
    """

    def __init__(self, native: Any) -> None:
        self.native = native

    def prepare_images(self, images: Sequence[Any]) -> list[Any]:
        """Turn caller images (paths, URLs, or loaded images) into what ``native`` reads."""
        return list(images)

    def system_one(self, payload: Mapping[str, Any]) -> Any:
        """Answer every question in ``payload``; return ``{"answers": {qid: answer}}``."""
        return self.native.system_one(
            state=payload["state"],
            questions=payload["questions"],
            **({"images": payload["images"]} if "images" in payload else {}),
        )

    def decide(self, example: DecisionInput, payload: Mapping[str, Any]) -> Any:
        """Answer ``payload``'s one ``"decision"`` question, which encodes ``example``."""
        return self.system_one(payload)

    def batch(self, examples: Sequence[DecisionInput]) -> BatchAnswers | None:
        """Answer many rows in fewer calls as ``(answer, per-row latency ms)`` pairs.

        ``None`` means the runtime has no batch path and rows go one at a time.
        """
        return None


class NativeExport(NamedTuple):
    """What a publisher trainer reports: rows trained, its own report, and the export path."""

    items: int
    report: Any
    path: Path


class PublisherAdapter(FamilyAdapter):
    """One publisher runtime. Subclasses set ``name``, ``extra``, and ``readout``.

    The class attributes describe what the publisher's runtime supports;
    ``NativeDecisionModel`` enforces them for local and HTTP backends alike.
    """

    readout: str
    """How the release reads out a decision; ``NativeCapabilities.native_readout``."""
    runtime_type: type[PublisherRuntime] = PublisherRuntime
    trainable = False
    """Whether ``train`` is implemented."""
    images = False
    """Whether the publisher's decision path accepts images."""
    max_options: int | None = None
    remote_code = False
    """Whether loading runs publisher code: it needs ``trust_remote_code`` and a pinned revision."""
    transformers: tuple[int, int] | None = None
    """The oldest Transformers release the publisher's code runs on."""
    http_error: str | None = None
    """Set when the publisher has no System One server, as the reason ``base_url`` fails."""
    vllm_server = False
    """Whether the publisher's System One server runs on vLLM."""

    def __init__(self) -> None:
        self.checkpoints = tuple(
            hub_id for hub_id, family in PUBLISHER_CHECKPOINTS.items() if family == self.name
        )

    def load(
        self, checkpoint: str, *, device: str | None = None, **kwargs: Any
    ) -> NativeDecisionModel:
        reject_futurix_options(kwargs, self.name)
        name, local = checkpoint_source(checkpoint)
        # load_model already ran ``requires`` on a local copy.
        return self.open_model(
            name, local=local, check_files=False, device=device or "cpu", **kwargs
        )

    def open_model(
        self,
        name: str,
        *,
        local: bool,
        check_files: bool,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        timeout: float | None = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        device: str = "cpu",
        revision: str | None = None,
        trust_remote_code: bool = False,
        api_key: str | None = None,
        verify_server: bool = True,
        **native_kwargs: Any,
    ) -> NativeDecisionModel:
        """Connect to the publisher's System One server at ``base_url``, or load it locally.

        ``NativeDecisionModel.from_pretrained`` documents the options.
        """
        from futurix_decision.models.native import NativeDecisionModel

        manifest = Path(name) / "futurix_native.json"
        if revision is None and manifest.is_file():
            revision = read_checkpoint_json(manifest).get("source_revision")
        if base_url is not None:
            if self.http_error is not None:
                raise ConfigError(self.http_error)
            if client is not None and api_key is not None:
                raise ConfigError("configure api_key headers on your supplied client")
            check_settings(timeout, max_retries)
            runtime = client or build_client(base_url, api_key=api_key, timeout=timeout)
            model = NativeDecisionModel(
                name,
                self.name,
                runtime,
                "http",
                max_retries=max_retries,
                owns_runtime=client is None,
            )
            if verify_server:
                try:
                    self.verify_server(runtime, name, max_retries)
                except BaseException:
                    model.close()
                    raise
            return model
        if client is not None:
            raise ConfigError("client needs base_url for a native System One server")
        if api_key is not None:
            raise ConfigError("api_key is only for a native System One server")
        if local and check_files:
            self.requires(Path(name), source=name)
        if self.remote_code:
            pinned_remote(revision, trust_remote_code, self.name, local=local)
        if self.transformers is not None:
            require_transformers(self.extra or self.name, self.transformers)
        native = self.open(name, local=local, device=device, revision=revision, **native_kwargs)
        return NativeDecisionModel(name, self.name, native, "local", source_revision=revision)

    def open(
        self, name: str, *, local: bool, device: str, revision: str | None, **kwargs: Any
    ) -> Any:
        """Load the publisher's model object, after the remote-code and version checks.

        The default loads a Transformers remote-code ``AutoModel`` from a
        complete snapshot, as Nox, Lux, and Liquid d1 publish.
        """
        from transformers import AutoModel

        # Inspect the complete, immutable publisher package before remote
        # code can silently load an incomplete head/backbone as a base LM.
        directory = self.snapshot(name, local=local, revision=revision)
        return AutoModel.from_pretrained(str(directory), trust_remote_code=True, **kwargs).to(
            device
        )

    def snapshot(
        self, name: str, *, local: bool, revision: str | None, heads: tuple[str, ...] = ()
    ) -> Path:
        """The checkpoint directory: a local copy as is, or a complete Hub snapshot.

        ``heads`` are small files fetched first, so a release missing its
        decision head fails before the large download.
        """
        if local:
            return Path(name)
        require_hub_files(name, self.name, revision, *heads)
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(name, revision=revision))
        self.requires(directory, source=name)
        return directory

    def verify_server(self, client: httpx.Client, checkpoint: str, max_retries: int) -> None:
        """Raise ``BackendError`` if the server at ``client`` does not serve ``checkpoint``."""

    def requires(self, path: Path, *, source: str | None = None) -> None:
        """Check a local copy or Hub snapshot; ``source`` names the Hub ID in the error."""
        files = CheckpointFiles(path)
        self.files(files)
        files.check(self.name, source=source)

    def files(self, files: CheckpointFiles) -> None:
        """Mark each file the publisher loader reads. The default needs none."""

    def finetune(
        self, model: DecisionBackend, dataset: Any, output_dir: str | Path, **kwargs: Any
    ) -> NativeDecisionModel:
        from futurix_decision.models.native import NativeDecisionModel

        if not isinstance(model, NativeDecisionModel) or model.family != self.name:
            raise ConfigError(f"{self.name} fine-tuning needs a {self.name} NativeDecisionModel")
        return model.finetune(dataset, output_dir, **kwargs)

    def train(
        self,
        model: NativeDecisionModel,
        rows: Iterable[DecisionExample],
        *,
        data: Path,
        output: Path,
        config: Any | None,
        **kwargs: Any,
    ) -> NativeExport:
        """Run the publisher's trainer on ``rows`` and export a reloadable checkpoint.

        ``data`` is a scratch file path for training records; ``output`` is the
        empty directory to export into. ``finetune_native`` calls this, then
        writes the manifest into ``NativeExport.path`` and reloads it.
        """
        raise NotImplementedError(f"{self.name} has no verified native training/export path")


def pinned_remote(
    revision: str | None,
    trust_remote_code: bool,
    family: str,
    *,
    local: bool = False,
) -> None:
    """Refuse publisher code unless the caller trusts it and, on the Hub, pins a commit."""
    if not trust_remote_code or (
        not local
        and (
            revision is None
            or len(revision) != 40
            or any(digit not in "0123456789abcdefABCDEF" for digit in revision)
        )
    ):
        raise ConfigError(
            f"{family} runs publisher code: pass trust_remote_code=True"
            + ("" if local else " and an immutable 40-character revision from the model card")
            + ". Review that code before loading."
        )


def require_hub_files(name: str, family: str, revision: str | None, *filenames: str) -> None:
    """Check small publisher metadata/heads before a large snapshot is downloaded."""
    if not filenames:
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for filename in filenames:
        try:
            hf_hub_download(name, filename, revision=revision)
        except EntryNotFoundError as exc:
            raise CheckpointError(
                f"incomplete {family} Hub checkpoint {name!r}; missing: {filename}"
            ) from exc


def load_images(images: Sequence[Any]) -> list[Any]:
    """Load image paths and URLs with Transformers; pass loaded images through."""
    from transformers.image_utils import load_image

    return [load_image(image) if isinstance(image, str) else image for image in images]
