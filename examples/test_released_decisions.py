"""One released checkpoint at a time: SDK dataset -> inference -> fine-tune -> reload.

Run with CUDA_VISIBLE_DEVICES set to an available GPU. Publisher-specific extras
must be installed in an environment compatible with that checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from futurix_decision import (
    DecisionTrainer, NativeSFTConfig, SFTConfig, load_decisions, load_model,
)

MODELS = {
    "bosun-0.6b": "Hanno-Labs/bosun-v3.1-0.6b",
    "bosun-1.7b": "Hanno-Labs/bosun-v3.1-1.7b",
    "kev": "jaredpalmer/kev-9b",
    "laya": "convaiinnovations/laya",
    "laya-multilingual": "convaiinnovations/laya-multilingual",
    "clef": "Cloudflare/clef",
    "clef-flash": "Cloudflare/clef-flash",
    "gliner": "fastino/GLiNER2.5-Decide",
    "pplx-decider": "perplexity-ai/pplx-decider-v1.1-27b",
}
REVISIONS = {
    "Cloudflare/clef": "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
    "Cloudflare/clef-flash": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
    "perplexity-ai/pplx-decider-v1.1-27b": "6195aa55a72ae21e503c8a60198f7b390d42adb5",
    "jaredpalmer/kev-9b": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
    "convaiinnovations/laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "convaiinnovations/laya-multilingual": "1720e3e3357cfe1e281542e223f8273b0890ca34",
    "fastino/GLiNER2.5-Decide": "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
}
REMOTE_CODE = {"clef", "clef-flash", "pplx-decider"}


def show_inference(model: object) -> None:
    state = {"ticket": "Customer was charged twice for one order."}
    questions = {
        "team": {
            "type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payment issue", "technical": "App issue"},
        },
        "urgent": {"type": "noul", "instructions": "Does this need urgent escalation?"},
    }
    results = {
        "choice": model.choice(
            state=state, question="Which team should handle this ticket?",
            choices=["billing", "technical"],
        ).to_dict(),
        "noul": model.noul(
            state=state, question="Does this need urgent escalation?"
        ).to_dict(),
        "score": model.score(
            state=state, question="Rate support priority from 0 to 2.",
            min_score=0, max_score=2,
        ).to_dict(),
        "system_one": model.system_one(state, questions),
    }
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).parent / "data" / "support_decisions.jsonl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.train and args.output is None:
        parser.error("--output is required with --train")
    if args.train and args.output.exists():
        parser.error(f"refusing to overwrite existing checkpoint: {args.output}")
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")

    # Both flat JSONL and grouped System One records use this one SDK loader.
    rows = load_decisions(args.dataset)
    print(f"SDK loaded {len(rows)} validated decisions from {args.dataset}", flush=True)
    checkpoint = MODELS[args.model]
    bosun = args.model.startswith("bosun-")
    if bosun:
        model = load_model(
            checkpoint, dtype=torch.bfloat16,
            device_map={"": args.device}, max_sequence_length=512,
        )
    else:
        native_options = {"dtype": torch.bfloat16, "merge": False} if args.model == "kev" else {}
        model = load_model(
            checkpoint, device=args.device, revision=REVISIONS[checkpoint],
            trust_remote_code=args.model in REMOTE_CODE,
            **native_options,
        )
    print(f"SDK loaded {checkpoint}; before fine-tuning:", flush=True)
    show_inference(model)
    if not args.train:
        return

    training_rows = rows[:args.max_examples]
    if bosun:
        trainer = DecisionTrainer(
            model=model, dataset=training_rows,
            sft_config=SFTConfig(
                method="lora", epochs=1, batch_size=1, precision="bf16",
                lora_rank=4, lora_alpha=8, lora_target_modules=["q_proj", "v_proj"],
            ),
        )
        history = trainer.train(stages=["decision_sft"])
        print(f"SDK SFT steps: {len(history)}; last: {history[-1]}", flush=True)
        trainer.export(args.output)  # Complete, reloadable Bosun decision checkpoint.
        reload_path = args.output
        reload_kwargs = {"device_map": {"": args.device}}
    else:
        options = {}
        if args.model in ("clef", "clef-flash", "pplx-decider"):
            options["config"] = NativeSFTConfig(
                epochs=1, batch_size=1, train_backbone=False,
            )
        elif args.model in ("laya", "laya-multilingual", "gliner"):
            options["config"] = NativeSFTConfig(
                epochs=1, batch_size=2, learning_rate=2e-5,
            )
        if args.model in ("laya", "laya-multilingual"):
            options["device"] = args.device
        if args.model == "kev":
            options["config"] = {
                "device": "cuda", "dtype": "bf16", "epochs": 1,
                "batch": 1, "accum": 1, "max_steps": args.max_examples,
            }
        trained = model.finetune(training_rows, args.output, **options)
        print(f"SDK native training report: {trained.training_report}", flush=True)
        reload_path = Path(trained.checkpoint)  # GLiNER publishes into output/final.
        reload_kwargs = {
            "device": args.device, "trust_remote_code": args.model in REMOTE_CODE,
            **({"dtype": torch.bfloat16, "merge": False} if args.model == "kev" else {}),
        }
        # A 27B checkpoint cannot coexist with its original GPU weights during
        # a second explicit reload. finetune() already returned a CPU-reloaded
        # native artifact, so release both copies before verifying GPU reload.
        del model, trained
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    reloaded = load_model(reload_path, **reload_kwargs)
    print(f"SDK reloaded fine-tuned checkpoint {reload_path}:", flush=True)
    show_inference(reloaded)


if __name__ == "__main__":
    main()
