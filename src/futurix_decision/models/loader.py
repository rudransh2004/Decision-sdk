"""``load_model``: find a checkpoint's family in the registry and load it, or convert a raw LM."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args, overload

from futurix_decision._lazy import require_transformers, torch_required
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.families import identify_family, reject_native_checkpoint
from futurix_decision.models.families.futurix import QWEN35, on_device
from futurix_decision.models.families.publisher import pinned_remote
from futurix_decision.models.protocol import DecisionBackend
from futurix_decision.models.registry import family_for_checkpoint, get_family
from futurix_decision.models.sources import ALIASES, checkpoint_source, infer_local_family

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel

Strategy = Literal["pointer", "bosun", "letter_code"]


@overload
def load_model(
    checkpoint: str | Path,
    *,
    family: str | None = None,
    convert: Literal[True],
    strategy: Strategy | None = None,
    head_type: str | None = None,
    train_backbone: bool | None = None,
    revision: str | None = None,
    device: str | None = None,
    **family_kwargs: Any,
) -> DecisionModel: ...


@overload
def load_model(
    checkpoint: str | Path,
    *,
    family: str | None = None,
    convert: bool = False,
    strategy: Strategy | None = None,
    head_type: str | None = None,
    train_backbone: bool | None = None,
    revision: str | None = None,
    device: str | None = None,
    **family_kwargs: Any,
) -> DecisionBackend: ...


def load_model(
    checkpoint: str | Path,
    *,
    family: str | None = None,
    convert: bool = False,
    strategy: Strategy | None = None,
    head_type: str | None = None,
    train_backbone: bool | None = None,
    revision: str | None = None,
    device: str | None = None,
    **family_kwargs: Any,
) -> DecisionBackend:
    """Load a known Hub release on demand or an existing local decision checkpoint.

    For a local copy with no recognizable publisher metadata, specify ``family``.
    This never converts a trained native decision readout into an untrained LM.

    - ``convert=True`` turns a raw causal LM into an untrained ``DecisionModel``
      with the readout that ``strategy`` names: ``"pointer"`` (default),
      ``"bosun"`` tokens, or a fixed ``"letter_code"`` table. Only Qwen3.5 has
      a noncausal adapter.
    - ``head_type`` and ``train_backbone`` configure Futurix checkpoints and
      conversions; a saved checkpoint rejects values that differ from its own.
    - ``revision`` pins the Hub commit. ``device`` places the model, for example
      ``"cuda:0"``; publisher runtimes default to ``"cpu"``.
    - ``family_kwargs`` go to the family's loader, for example ``base_url`` and
      ``trust_remote_code`` for publisher runtimes, or ``max_sequence_length``
      and Transformers ``from_pretrained`` arguments for ``DecisionModel``.

    Raises ``MissingDependencyError`` when the checkpoint needs torch and it is
    not installed.
    """
    if strategy is not None and strategy not in get_args(Strategy):
        raise ConfigError("strategy must be 'pointer', 'bosun', or 'letter_code'")
    if strategy is not None and not convert:
        raise ConfigError("strategy is only for converting a plain causal LM")
    if device is not None and "device_map" in family_kwargs:
        raise ConfigError("pass either device or device_map, not both")
    kwargs = dict(family_kwargs)
    for key, value in (
        ("revision", revision),
        ("head_type", head_type),
        ("train_backbone", train_backbone),
    ):
        if value is not None:
            kwargs[key] = value
    with torch_required(f"loading {checkpoint}"):
        return _load_model(
            checkpoint, convert=convert, family=family, strategy=strategy, device=device, **kwargs
        )


def _load_model(
    checkpoint: str | Path,
    *,
    convert: bool,
    family: str | None,
    strategy: Strategy | None,
    device: str | None,
    **kwargs: Any,
) -> DecisionBackend:
    original, _ = checkpoint_source(checkpoint)
    name, local = checkpoint_source(ALIASES.get(original, original))
    path = Path(name)
    registered = None if local else identify_family(name)
    if local:
        identified = infer_local_family(path)
    else:
        known = family_for_checkpoint(name)
        identified = known.name if known else (registered.name if registered else None)
    if family is not None and identified is not None and family != identified:
        raise CheckpointError(f"{name} belongs to {identified}, not {family}")
    family = family or identified
    if convert:
        known_as = family or (registered.name if registered else None)
        if known_as is not None:
            raise ConfigError(
                f"{name} belongs to the known decision-model family {known_as!r} and already "
                "contains a trained decision head; convert=True would discard its native "
                "readout. Load it without convert"
            )
        return on_device(_convert(name, local=local, strategy=strategy, **kwargs), device)
    if name in QWEN35:
        raise ConfigError(
            "Qwen3.5-9B is not decision-trained; pass convert=True and train before inference"
        )
    from futurix_decision.utils.config import load_decision_config

    if family == "futurix" and not local:
        metadata = load_decision_config(name, **kwargs)
        if not metadata:
            raise CheckpointError(
                f"incomplete Futurix Hub checkpoint {name!r}; missing: decision_config.json"
            )
        if metadata.get("architecture") != "futurix-decision":
            raise CheckpointError(f"{name} does not contain a Futurix decision checkpoint")
    if family is None and not local and name.count("/") == 1:
        if load_decision_config(name, **kwargs).get("architecture") == "futurix-decision":
            family = "futurix"
    if family is None:
        if local:
            raise CheckpointError(
                f"cannot identify local decision checkpoint at {name!r}; "
                "pass the correct family=... or include the publisher's native config/head"
            )
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    adapter = get_family(family)
    if local:
        adapter.requires(path)
    elif adapter.checkpoints and name not in adapter.checkpoints:
        raise ConfigError(f"unknown selected decision checkpoint: {name}")
    return adapter.load(name, device=device, **kwargs)


def _convert(name: str, *, local: bool, strategy: Strategy | None, **kwargs: Any) -> DecisionModel:
    """Turn a plain causal LM into an untrained ``DecisionModel`` with ``strategy``'s readout."""
    from futurix_decision.models.decision_model import DecisionModel

    if name in QWEN35:
        require_transformers("qwen3.5", (5, 17))
        if strategy == "bosun":
            if kwargs.pop("head_type", "tokens") != "tokens":
                raise ConfigError("strategy='bosun' requires head_type='tokens'")
            if kwargs.get("decision_tokens", 256) != 256:
                raise ConfigError("strategy='bosun' requires exactly 256 decision tokens")
            quantization = kwargs.get("quantization_config")
            if getattr(quantization, "load_in_4bit", False) or getattr(
                quantization, "load_in_8bit", False
            ):
                raise ConfigError("Bosun token rows must be added before quantizing the base")
            return DecisionModel.from_pretrained(
                name,
                head_type="tokens",
                prompt_format="bosun-v3",
                bosun_template_fallback=True,
                train_backbone=kwargs.pop("train_backbone", True),
                **kwargs,
            )
        if strategy == "letter_code":
            if kwargs.pop("head_type", "letter_code") != "letter_code":
                raise ConfigError("strategy='letter_code' requires head_type='letter_code'")
            return DecisionModel.from_pretrained(
                name,
                head_type="letter_code",
                head_prompt_version="letter-code-v1",
                decision_tokens=kwargs.pop("decision_tokens", 255),
                train_backbone=kwargs.pop("train_backbone", True),
                attention_mode=kwargs.pop("attention_mode", "noncausal_full_attention"),
                **kwargs,
            )
        if strategy == "pointer" and kwargs.get("head_type", "pointer") != "pointer":
            raise ConfigError("strategy='pointer' requires head_type='pointer'")
        return DecisionModel.from_pretrained(
            name, head_type=kwargs.pop("head_type", "pointer"), **kwargs
        )
    if not local and name.count("/") != 1:
        raise ConfigError(
            "convert=True needs an existing local directory or a Hugging Face model ID"
        )
    from futurix_decision.utils.config import load_decision_config

    metadata = load_decision_config(name, **kwargs)
    if metadata.get("architecture") == "futurix-decision":
        raise ConfigError(
            "this is already a trained Futurix decision checkpoint; omit convert=True"
        )
    reject_native_checkpoint(name, metadata)
    if not local and kwargs.get("trust_remote_code"):
        pinned_remote(kwargs.get("revision"), True, "conversion")
    from transformers import AutoConfig

    config_kwargs = {
        key: kwargs[key]
        for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
        if key in kwargs
    }
    config = AutoConfig.from_pretrained(name, **config_kwargs)
    architectures = getattr(config, "architectures", None) or ()
    selected = strategy or "pointer"
    qwen_multimodal = (
        selected == "letter_code"
        and getattr(config, "model_type", None) == "qwen3_5"
        and getattr(config, "vision_config", None) is not None
    )
    if getattr(config, "is_encoder_decoder", False) or (
        getattr(config, "vision_config", None) and not qwen_multimodal
    ):
        raise CheckpointError(
            "conversion supports text-only causal LMs, not seq2seq or multimodal models"
        )
    if (
        architectures
        and not any(
            architecture.endswith(("ForCausalLM", "LMHeadModel")) for architecture in architectures
        )
        and not (qwen_multimodal and "Qwen3_5ForConditionalGeneration" in architectures)
    ):
        raise CheckpointError(
            f"{name} does not declare a text causal LM architecture: {architectures}"
        )
    required_head = {"bosun": "tokens", "pointer": "pointer", "letter_code": "letter_code"}[
        selected
    ]
    head_type = kwargs.pop("head_type", required_head)
    if head_type != required_head:
        raise ConfigError(f"strategy={selected!r} requires head_type={required_head!r}")
    if selected == "bosun":
        if kwargs.get("decision_tokens", 256) != 256:
            raise ConfigError("strategy='bosun' requires exactly 256 decision tokens")
        quantization = kwargs.get("quantization_config")
        if getattr(quantization, "load_in_4bit", False) or getattr(
            quantization, "load_in_8bit", False
        ):
            raise ConfigError("Bosun token rows must be added before quantizing the base")
    train_backbone = kwargs.pop("train_backbone", selected != "pointer")
    if not isinstance(train_backbone, bool):
        raise ConfigError("train_backbone must be a boolean")
    if selected == "letter_code":
        if qwen_multimodal:
            require_transformers("qwen3.5", (5, 17))
        attention_mode = kwargs.pop(
            "attention_mode",
            "noncausal_full_attention"
            if getattr(config, "model_type", None) == "qwen3_5"
            else "causal",
        )
        if attention_mode != "causal" and getattr(config, "model_type", None) != "qwen3_5":
            raise ConfigError("noncausal letter-code attention needs a Qwen3.5-family backbone")
        if kwargs.get("decision_tokens", 255) > 255:
            raise ConfigError("letter-code readout supports at most 255 answer codes")
        return DecisionModel.from_pretrained(
            name,
            head_type="letter_code",
            head_prompt_version="letter-code-v1",
            decision_tokens=kwargs.pop("decision_tokens", 255),
            attention_mode=attention_mode,
            multimodal_loader=qwen_multimodal,
            train_backbone=train_backbone,
            **kwargs,
        )
    return DecisionModel.from_pretrained(
        name,
        head_type=head_type,
        **(
            {"prompt_format": "bosun-v3", "bosun_template_fallback": True}
            if selected == "bosun"
            else {"head_prompt_version": "span-v1"}
        ),
        train_backbone=train_backbone,
        **kwargs,
    )
