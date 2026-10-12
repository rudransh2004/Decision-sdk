"""futurix-decision train | evaluate | export | serve."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from futurix_decision import DecisionModel, DecisionTrainer, evaluate, load_model
from futurix_decision.datasets import load_decisions
from futurix_decision.rewards import BrierScore, LogScore, RankedProbabilityScore, SphericalScore
from futurix_decision.trainers import CalibrationConfig, RLCDConfig, SFTConfig
from futurix_decision.utils.config import load_config


def _source(path: str, *, split: str = "train", revision: str | None = None):
    return load_decisions(path, split=split, streaming=True, revision=revision)


def _config_for(cls, config: dict[str, Any], stage: str, overrides: dict[str, Any] | None = None):
    section = dict(config.get("training", {}).get(stage, {}))
    section.pop("enabled", None)
    section.update(overrides or {})
    return cls(**section)


def _rewards(config: dict[str, Any]):
    weights = config.get("rewards")
    if weights is None:
        return None
    constructors = {
        "log": LogScore,
        "brier": BrierScore,
        "spherical": SphericalScore,
        "ranked": RankedProbabilityScore,
    }
    unknown = set(weights) - set(constructors)
    if unknown:
        raise ValueError(f"unknown reward rules: {sorted(unknown)}")
    return [constructors[name](weight=weight) for name, weight in weights.items()]


def _select_gpu(index: int | None) -> dict[str, Any]:
    """Restrict this SDK process to one physical GPU before CUDA initializes."""
    if index is None:
        return {}
    import torch

    if index < 0 or torch.cuda.is_initialized():
        raise ValueError("--gpu must be non-negative and selected before CUDA initializes")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    if not torch.cuda.is_available():
        raise ValueError(f"GPU {index} is not available")
    return {"dtype": torch.bfloat16, "device_map": {"": 0}}


def _train(args: argparse.Namespace) -> None:
    device_kwargs = _select_gpu(args.gpu)
    config = load_config(args.config) if args.config else {}
    model_section = config.get("model", {})
    checkpoint = args.model or model_section.get("name")
    dataset = args.dataset or config.get("dataset")
    output = args.output or config.get("output")
    if not all((checkpoint, dataset, output)):
        raise ValueError("train requires model, dataset and output (arguments or YAML)")
    if args.method:
        stages = [{"sft": "decision_sft"}.get(args.method, args.method)]
    elif args.stages:
        stages = args.stages.split(",")
    else:
        stages = [
            name
            for name, key in (
                ("decision_sft", "sft"),
                ("calibration", "calibration"),
                ("rlcd", "rlcd"),
            )
            if config.get("training", {}).get(key, {}).get("enabled", False)
        ] or ["decision_sft"]
    checkpoint_dir = args.checkpoint_dir or config.get("checkpoint_dir")
    previous_history: dict[str, list[dict[str, float | int]]] = {}
    if args.resume:
        if not checkpoint_dir:
            raise ValueError("--resume requires --checkpoint-dir (or checkpoint_dir in config)")
        progress = json.loads((Path(checkpoint_dir) / "progress.json").read_text(encoding="utf-8"))
        if progress["dataset"] != dataset:
            raise ValueError("resume dataset differs from the stage checkpoint")
        completed = progress["completed_stages"]
        if completed != stages[: len(completed)]:
            raise ValueError("resume stages differ from the completed stage prefix")
        previous_history = progress["history"]
        checkpoint = progress["checkpoint"]
        stages = stages[len(completed) :]
        if not stages:
            raise ValueError("all configured stages have already completed")
    selected_method = args.finetune_method or next(
        (
            config.get("training", {}).get(stage, {}).get("method")
            for stage in ("sft", "calibration", "rlcd")
            if config.get("training", {}).get(stage, {}).get("enabled")
            and config.get("training", {}).get(stage, {}).get("method")
        ),
        "full",
    )
    load_kwargs = {
        key: model_section[key]
        for key in ("decision_tokens", "max_sequence_length", "seed", "head_type", "head_config")
        if key in model_section
    }
    if args.decision_tokens is not None:
        load_kwargs["decision_tokens"] = args.decision_tokens
    load_kwargs.update(device_kwargs)
    revision = args.revision or model_section.get("revision")
    source_format = args.model_format or model_section.get("format", "standard")
    conversion = args.convert_strategy or model_section.get("strategy")
    if conversion and (args.resume or source_format != "standard"):
        raise ValueError("raw-model conversion requires a fresh run with --model-format standard")
    if source_format == "native":
        if args.resume or stages != ["decision_sft"] or args.finetune_method:
            raise ValueError(
                "native fine-tuning supports a fresh decision_sft run; "
                "use publisher config for other recipes"
            )
        if args.calibration_dataset or args.validation_dataset or args.log_jsonl or checkpoint_dir:
            raise ValueError(
                "native trainers manage their own checkpoints/calibration; "
                "use the Python API for customization"
            )
        native = load_model(
            checkpoint,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
            device="cuda:0" if args.gpu is not None else "cpu",
            family=args.native_family,
        )
        rows = (
            islice(_source(dataset), args.max_examples)
            if args.max_examples is not None
            else dataset
        )
        tuned = native.finetune(rows, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "family": tuned.family,
                    "train_items": tuned.training_report["train_items"],
                }
            )
        )
        return
    if args.resume:
        model = DecisionModel.from_pretrained(checkpoint, trainable_adapter=True, **device_kwargs)
        if model.prompt_format != progress["prompt_format"]:
            raise ValueError("resumed checkpoint prompt format differs from progress metadata")
        if model.head_type != progress.get("head_type", "tokens"):
            raise ValueError("resumed checkpoint decision head differs from progress metadata")
        if model_section.get("head_type", model.head_type) != model.head_type:
            raise ValueError("resumed checkpoint decision head differs from training config")
    elif source_format == "bosun":
        if model_section.get("head_type", "tokens") != "tokens":
            raise ValueError("Bosun's prompt contract requires the token decision head")
        bosun_kwargs = dict(device_kwargs)
        if revision:
            bosun_kwargs["revision"] = revision
        if "max_sequence_length" in model_section:
            bosun_kwargs["max_sequence_length"] = model_section["max_sequence_length"]
        model = DecisionModel.from_bosun(checkpoint, **bosun_kwargs)
    else:
        if source_format in ("decider", "jpt", "eikos"):
            if model_section.get("head_type", "tokens") != "tokens" or model_section.get(
                "head_config"
            ):
                raise ValueError(
                    "native letter models use their trained readout, not a new decision head"
                )
            expected_format = {
                "decider": "decider-v2.1",
                "jpt": "jpt-llm2jev-chat",
                "eikos": "eikos-letter-v1-semif",
            }[source_format]
            if source_format == "eikos" and "decision_tokens" in model_section:
                raise ValueError("Eikos one-pass width comes from the release metadata")
            if source_format != "eikos" and model_section.get("decision_tokens", 255) != 255:
                raise ValueError("native letter models support 255 existing option labels")
            if model_section.get("prompt_format", expected_format) != expected_format:
                raise ValueError(f"{source_format} requires its native prompt")
            load_kwargs = {
                key: load_kwargs[key]
                for key in ("max_sequence_length", "seed")
                if key in load_kwargs
            } | device_kwargs
            if source_format == "jpt" and "temperature" in model_section:
                load_kwargs["temperature"] = model_section["temperature"]
        if revision:
            load_kwargs["revision"] = revision
        if source_format not in ("decider", "jpt", "eikos") and model_section.get("prompt_format"):
            load_kwargs["prompt_format"] = model_section["prompt_format"]
        if selected_method == "qlora":
            import torch
            from transformers import BitsAndBytesConfig

            load_kwargs.pop("dtype", None)
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                device_map={
                    "": 0 if args.gpu is not None else int(os.environ.get("LOCAL_RANK", 0))
                },
            )
        if conversion:
            if args.trust_remote_code:
                load_kwargs["trust_remote_code"] = True
            if "attention_mode" in model_section:
                load_kwargs["attention_mode"] = model_section["attention_mode"]
            model = load_model(checkpoint, convert=True, strategy=conversion, **load_kwargs)
        else:
            model = (
                DecisionModel.from_decider(checkpoint, **load_kwargs)
                if source_format == "decider"
                else DecisionModel.from_jpt(checkpoint, **load_kwargs)
                if source_format == "jpt"
                else DecisionModel.from_eikos(checkpoint, **load_kwargs)
                if source_format == "eikos"
                else DecisionModel.from_pretrained(checkpoint, **load_kwargs)
            )
    method = args.finetune_method
    overrides = {"method": method} if method else {}
    if args.max_examples is not None:
        overrides["max_examples"] = args.max_examples
    sft = _config_for(SFTConfig, config, "sft", overrides)
    calibration = _config_for(CalibrationConfig, config, "calibration", overrides)
    rlcd = _config_for(RLCDConfig, config, "rlcd", overrides)
    if args.log_jsonl:
        sft = replace(sft, log_jsonl=args.log_jsonl)
        calibration = replace(calibration, log_jsonl=args.log_jsonl)
        rlcd = replace(rlcd, log_jsonl=args.log_jsonl)
    trainer = DecisionTrainer(
        model=model,
        dataset=dataset,
        calibration_dataset=args.calibration_dataset,
        sft_config=sft,
        calibration_config=calibration,
        rlcd_config=rlcd,
        rewards=_rewards(config),
        checkpoint_dir=checkpoint_dir,
        validation_dataset=args.validation_dataset,
        validation_max_rows=args.validation_max_rows,
    )
    if args.resume and progress.get("dataset_revision") != trainer.dataset_revision:
        raise ValueError("resume dataset revision differs from the stage checkpoint")
    trainer.history.update(previous_history)
    if args.resume:
        trainer.validation_metrics.update(progress.get("validation_metrics", {}))
    trainer.train(stages=stages)
    if args.selection_metric != "last" and trainer.validation_metrics and checkpoint_dir:
        trainer.select_best(metric=args.selection_metric)
    trainer.export(output, merge_lora=True, safetensors=True)
    if trainer._active and trainer._active._accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "stages": list(trainer.history),
                    "selected_stage": trainer.selected_stage,
                    "temperature": trainer.model.temperature,
                }
            )
        )


def _evaluate(args: argparse.Namespace) -> None:
    device_kwargs = _select_gpu(args.gpu)
    if args.max_sequence_length is not None:
        device_kwargs["max_sequence_length"] = args.max_sequence_length
    if args.revision:
        device_kwargs["revision"] = args.revision
    dataset = _source(args.dataset, split=args.split, revision=args.dataset_revision)
    if args.skip_rows:
        if args.skip_rows < 0:
            raise ValueError("--skip-rows must be non-negative")
        dataset = islice(dataset, args.skip_rows, None)
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("--max-rows must be positive")
        dataset = islice(dataset, args.max_rows)
    if args.backend == "vllm":
        if args.model_format == "native":
            raise ValueError(
                "native decision heads need the publisher runtime; OpenJev uses --native-base-url"
            )
        if args.revision:
            raise ValueError(
                "--revision is for PyTorch upstream loading; export a pinned model for vLLM"
            )
        if args.model_format == "bosun":
            raise ValueError("export the pinned Bosun checkpoint before vLLM evaluation")
        from futurix_decision.vllm import VLLMEngine

        model = VLLMEngine(
            args.model,
            max_model_len=args.max_sequence_length or 4096,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
        if args.max_sequence_length is not None:
            model.max_sequence_length = args.max_sequence_length
    else:
        if args.model_format == "native":
            model = load_model(
                args.model,
                device="cuda:0" if args.gpu is not None else "cpu",
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                base_url=args.native_base_url,
                family=args.native_family,
            )
        elif args.model_format == "bosun":
            model = DecisionModel.from_bosun(args.model, **device_kwargs)
        elif args.model_format == "decider":
            model = DecisionModel.from_decider(args.model, **device_kwargs)
        elif args.model_format == "jpt":
            model = DecisionModel.from_jpt(args.model, **device_kwargs)
        elif args.model_format == "eikos":
            model = DecisionModel.from_eikos(args.model, **device_kwargs)
        else:
            model = DecisionModel.from_pretrained(args.model, **device_kwargs)
    result = evaluate(
        model,
        dataset,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        predictions_output=args.predictions_output,
        resume=args.resume_evaluation,
        run_id=json.dumps(
            {
                "model": args.model,
                "model_format": args.model_format,
                "dataset": args.dataset,
                "split": args.split,
                "max_rows": args.max_rows,
                **({"revision": args.revision} if args.revision else {}),
                **({"dataset_revision": args.dataset_revision} if args.dataset_revision else {}),
                **({"backend": args.backend} if args.backend != "transformers" else {}),
                **({"skip_rows": args.skip_rows} if args.skip_rows else {}),
            },
            sort_keys=True,
        ),
        max_batch_tokens=None if args.model_format == "native" else args.max_batch_tokens,
    )
    if args.metrics_output:
        destination = Path(args.metrics_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _export(args: argparse.Namespace) -> None:
    if args.model_format == "native":
        raise ValueError(
            "native publisher artifacts are exported by load_model(...).finetune(dataset, output)"
        )
    device_kwargs = _select_gpu(args.gpu)
    if args.model_format == "bosun":
        model = DecisionModel.from_bosun(args.model, revision=args.revision, **device_kwargs)
    elif args.model_format == "decider":
        kwargs = {"revision": args.revision} if args.revision else {}
        model = DecisionModel.from_decider(args.model, **kwargs, **device_kwargs)
    elif args.model_format == "jpt":
        kwargs = {"revision": args.revision} if args.revision else {}
        model = DecisionModel.from_jpt(args.model, **kwargs, **device_kwargs)
    elif args.model_format == "eikos":
        kwargs = {"revision": args.revision} if args.revision else {}
        model = DecisionModel.from_eikos(args.model, **kwargs, **device_kwargs)
    else:
        model = DecisionModel.from_pretrained(args.model, **device_kwargs)
    output = Path(args.output)
    model.save_pretrained(output, safe_serialization=True)
    previous = Path(args.model) / "training_metadata.json"
    if previous.is_file():
        shutil.copyfile(previous, output / "training_metadata.json")
    else:
        (output / "training_metadata.json").write_text(
            json.dumps(
                {"exported_from": args.model, "created_at": datetime.now(timezone.utc).isoformat()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(str(output))


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("serve requires pip install futurix-decision[serve]") from exc
    from futurix_decision.vllm.api import create_app

    if args.backend == "vllm":
        if args.model_format == "native":
            raise ValueError(
                "native decision heads need the publisher runtime, not the generic vLLM engine"
            )
        from futurix_decision.vllm import VLLMEngine

        engine = VLLMEngine(args.model)
    else:
        if args.model_format == "native":
            import torch

            native_device = (
                ("cuda" if torch.cuda.is_available() else "cpu")
                if args.device == "auto"
                else args.device
            )
            engine = load_model(
                args.model,
                base_url=args.native_base_url,
                family=args.native_family,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                device=native_device,
            )
        else:
            engine = (
                DecisionModel.from_decider(args.model)
                if args.model_format == "decider"
                else DecisionModel.from_jpt(args.model)
                if args.model_format == "jpt"
                else DecisionModel.from_eikos(args.model)
                if args.model_format == "eikos"
                else DecisionModel.from_pretrained(args.model)
            )
        if args.device != "cpu" and args.model_format != "native":
            import torch

            if torch.cuda.is_available():
                engine.backbone.to("cuda")
            elif args.device == "cuda":
                raise ValueError("CUDA requested but not available")
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="futurix-decision")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run SFT, calibration, and/or RLCD")
    train.add_argument("--config")
    train.add_argument("--model")
    train.add_argument("--dataset")
    train.add_argument("--calibration-dataset")
    train.add_argument("--validation-dataset")
    train.add_argument("--validation-max-rows", type=int)
    train.add_argument("--output")
    train.add_argument("--method", choices=("sft", "decision_sft", "calibration", "rlcd"))
    train.add_argument("--stages", help="Comma-separated ordered stage names")
    train.add_argument("--finetune-method", choices=("full", "lora", "qlora"))
    train.add_argument("--log-jsonl")
    train.add_argument(
        "--model-format", choices=("standard", "bosun", "decider", "jpt", "eikos", "native")
    )
    train.add_argument(
        "--convert-strategy",
        choices=("pointer", "bosun", "letter_code"),
        type=lambda value: "letter_code" if value == "pplx" else value,
        help="Convert a raw model before training; letter_code supports RLCD",
    )
    train.add_argument(
        "--decision-tokens",
        type=int,
        help="Number of available decision codes/slots for a raw conversion",
    )
    train.add_argument("--revision", help="Pinned upstream model revision")
    train.add_argument("--trust-remote-code", action="store_true")
    train.add_argument("--native-family", help="Native family for a local publisher checkpoint")
    train.add_argument("--gpu", type=int, help="Physical GPU index; isolates this SDK process")
    train.add_argument("--checkpoint-dir", help="Save a reloadable checkpoint after each stage")
    train.add_argument(
        "--resume", action="store_true", help="Continue from checkpoint-dir/progress.json"
    )
    train.add_argument("--max-examples", type=int, help="Limit training rows per stage for a pilot")
    train.add_argument(
        "--selection-metric", choices=("accuracy", "nll", "last"), default="accuracy"
    )
    train.set_defaults(run=_train)
    evaluation = commands.add_parser("evaluate", help="Report held-out decision metrics")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--dataset", default="SargeDev/jev-distill-corpus-v3")
    evaluation.add_argument("--split", default="test_set_30k")
    evaluation.add_argument("--dataset-revision", help="Pinned Hugging Face dataset commit")
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.add_argument(
        "--model-format",
        choices=("standard", "bosun", "decider", "jpt", "eikos", "native"),
        default="standard",
    )
    evaluation.add_argument(
        "--revision", help="Pinned upstream model revision for PyTorch evaluation"
    )
    evaluation.add_argument("--trust-remote-code", action="store_true")
    evaluation.add_argument(
        "--native-base-url", help="Publisher System One server (not generic vLLM)"
    )
    evaluation.add_argument(
        "--native-family", help="Native family for a local publisher checkpoint"
    )
    evaluation.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    evaluation.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.75)
    evaluation.add_argument("--gpu", type=int)
    evaluation.add_argument("--max-rows", type=int)
    evaluation.add_argument(
        "--skip-rows", type=int, default=0, help="Skip rows before a diagnostic slice"
    )
    evaluation.add_argument("--metrics-output")
    evaluation.add_argument("--max-sequence-length", type=int)
    evaluation.add_argument("--progress-every", type=int, default=1000)
    evaluation.add_argument(
        "--predictions-output", help="Append per-row results for resumable evaluation"
    )
    evaluation.add_argument("--resume-evaluation", action="store_true")
    evaluation.add_argument("--max-batch-tokens", type=int, default=16384)
    evaluation.set_defaults(run=_evaluate)
    export = commands.add_parser("export", help="Export a full Hugging Face checkpoint")
    export.add_argument("--model", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--model-format",
        choices=("standard", "bosun", "decider", "jpt", "eikos", "native"),
        default="standard",
    )
    export.add_argument("--revision", help="Pinned upstream Bosun revision")
    export.add_argument("--gpu", type=int)
    export.set_defaults(run=_export)
    serve = commands.add_parser("serve", help="Start the FastAPI decision server")
    serve.add_argument("--model", required=True)
    serve.add_argument(
        "--model-format",
        choices=("standard", "decider", "jpt", "eikos", "native"),
        default="standard",
    )
    serve.add_argument("--native-base-url", help="Publisher System One server (not generic vLLM)")
    serve.add_argument("--native-family", help="Native family for a local publisher checkpoint")
    serve.add_argument("--revision", help="Immutable upstream model revision")
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    serve.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(run=_serve)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
