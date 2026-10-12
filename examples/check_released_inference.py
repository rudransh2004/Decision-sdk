"""Check ONE released decision checkpoint with live SDK load and typed inference.

Uses a fresh, self-cleaning Hub cache for each invocation. Large checkpoints
never overlap on disk or GPU; results appear on stdout before cache removal.
Use the Transformers-4 environment for Bosun/Laya/GLiNER and Transformers-5
with the corresponding publisher packages for the remaining releases.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# All remote-code revisions are immutable Hub commits, not moving branches.
EXTRAS = {
    "decider-4b": ("Mapika/decider-4b", "eb5fbdfc9448473ec25e399882912863afbdb70e"),
    "jpt-4b": ("kirp/jpt-4b", "78312f855b8bebf83ae7e9e8f05b5a0f73f519a9"),
    "jpt-9b": ("kirp/jpt-9b", "9fda2c7f123d36c4789c46c93737e80fc3a6c9fe"),
    "eikos-4b": ("caiovicentino1/Eikos-4B", "d06420bda550072bf1385905b1f0e1421fcd710b"),
    "eikos-27b": ("caiovicentino1/Eikos-27B", "f44903e3d485cc742334b224fe309c6680eb0c13"),
    "openjev": ("openjev/openjev", "b4a89f5884f3482a5e894b2e7ff01db8ea033726"),
    "openjev-flash": ("openjev/OpenJev-Flash-9B", "b77b0fff67c77a76c16d4da50a66d2fdbc124fae"),
    "nox-1": ("vllm-sr/Decision-1.0-Nox-4B", "e3cb3248a36056f1857012166691d6aaa0b293e8"),
    "nox-2": ("vllm-sr/Decision-2.0-Nox-4B", "ae269ecf34a0b3a89026bc1afd9c995279212fad"),
    "lux-1": ("vllm-sr/Decision-1.0-Lux-9B", "95bcd72d2ea1d30ac09eb433f71cb4a128ed8261"),
    "lux-2": ("vllm-sr/Decision-2.0-Lux-9B", "1dcd1e7f240eab37ac09a6d0b0f77654b59411f9"),
    "liquid-d1": ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432"),
}
EARLIER = (
    "bosun-0.6b",
    "bosun-1.7b",
    "kev",
    "laya",
    "laya-multilingual",
    "gliner",
    "clef",
    "clef-flash",
    "pplx-decider",
)
RAW_CONVERSIONS = {
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "qwen3.5-9b-base": ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
}
RELEASES = tuple(EXTRAS) + EARLIER + tuple(RAW_CONVERSIONS)


def _check_release(name: str, device: str, base_url: str | None, precision: str) -> None:
    import torch
    from test_released_decisions import MODELS, REVISIONS, show_inference

    from futurix_decision import DecisionModel, load_model

    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    if name in EXTRAS:
        checkpoint, revision = EXTRAS[name]
    elif name in RAW_CONVERSIONS:
        checkpoint, revision = RAW_CONVERSIONS[name]
    else:
        checkpoint = MODELS[name]
        revision = REVISIONS.get(checkpoint)
    print(
        f"SDK loading {checkpoint} @ {revision or 'SDK-pinned release'} "
        f"on {device} with {precision}",
        flush=True,
    )
    if name in RAW_CONVERSIONS:
        model = load_model(
            checkpoint,
            convert=True,
            head_type="pointer",
            revision=revision,
            dtype=dtype,
            device_map={"": device},
            max_sequence_length=512,
        )
    elif name.startswith("openjev"):
        # OpenJev's SDK readout requires the publisher's calibrated helper.
        model = load_model(checkpoint, base_url=base_url) if base_url else load_model(checkpoint)
    elif name in ("decider-4b", "jpt-4b", "jpt-9b", "eikos-4b", "eikos-27b"):
        method = (
            "from_decider"
            if name == "decider-4b"
            else "from_jpt"
            if name.startswith("jpt")
            else "from_eikos"
        )
        model = getattr(DecisionModel, method)(
            checkpoint,
            revision=revision,
            dtype=dtype,
            device_map={"": device},
            max_sequence_length=512,
        )
    elif name.startswith("bosun"):
        model = load_model(
            checkpoint,
            dtype=dtype,
            device_map={"": device},
            max_sequence_length=512,
        )
    else:
        model = load_model(
            checkpoint,
            device=device,
            revision=revision,
            trust_remote_code=name
            in (
                "clef",
                "clef-flash",
                "pplx-decider",
                "nox-1",
                "nox-2",
                "lux-1",
                "lux-2",
                "liquid-d1",
            ),
            **({"dtype": dtype, "merge": False} if name == "kev" else {}),
        )
    show_inference(model)
    if name in RAW_CONVERSIONS:
        print(
            f"PASS: {checkpoint} converted and answered SDK API probes with an UNTRAINED head; "
            "these predictions have no decision-model validity",
            flush=True,
        )
    else:
        print(f"PASS: {checkpoint} loaded and answered all four SDK inference probes", flush=True)
    if name == "liquid-d1":
        from PIL import Image

        image_result = model.system_one(
            None,
            {
                "color": {
                    "type": "choice",
                    "instructions": "What color is the image?",
                    "criteria": {"red": "Red image", "blue": "Blue image"},
                }
            },
            images=[Image.new("RGB", (64, 64), "red")],
        )
        assert set(image_result["answers"]) == {"color"}
        print("PASS: Liquid d1 image-bearing System One request", flush=True)
    if name in {"nox-1", "nox-2", "lux-1", "lux-2", "liquid-d1"}:
        import gc

        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(checkpoint, revision=revision, local_files_only=True))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        local_model = load_model(directory, device=device, trust_remote_code=True)
        show_inference(local_model)
        print(
            f"PASS: {checkpoint} reloaded from a local directory and answered all four probes",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=RELEASES)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--base-url", help="Publisher's running OpenJev System One helper")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    device = "cpu" if args.model == "gliner" and args.device == "cuda" else args.device
    if args.worker:
        _check_release(args.model, device, args.base_url, args.precision)
        return

    required_gib = 70 if args.model in ("clef", "pplx-decider", "eikos-27b") else 20
    available_gib = shutil.disk_usage(args.scratch).free / 1024**3
    if available_gib < required_gib:
        parser.error(
            f"{args.model} needs at least {required_gib} GiB scratch space; "
            f"{args.scratch} has {available_gib:.1f} GiB free"
        )
    with tempfile.TemporaryDirectory(
        prefix=f"futurix-check-{args.model}-", dir=args.scratch
    ) as scratch:
        environment = dict(os.environ, HF_HOME=str(Path(scratch) / "hf"))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            args.model,
            "--worker",
            "--device",
            device,
            "--precision",
            args.precision,
        ]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        print(f"Individual SDK check: {args.model}; temporary cache: {scratch}", flush=True)
        subprocess.run(command, env=environment, check=True)
    print(f"Completed {args.model}; only this run's temporary cache was removed", flush=True)


if __name__ == "__main__":
    main()
