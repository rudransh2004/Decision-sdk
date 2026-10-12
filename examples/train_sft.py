"""Reproducible Decision SFT entry point; run with python or accelerate launch."""

from __future__ import annotations

import argparse
import os

import torch
from transformers import BitsAndBytesConfig

from futurix_decision import DecisionModel, DecisionTrainer, SFTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decision checkpoint with SFT")
    parser.add_argument("--model", required=True, help="Hugging Face ID or local checkpoint")
    parser.add_argument("--dataset", required=True, help="JSON/JSONL path or Hugging Face ID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("full", "lora", "qlora"), default="lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--decision-tokens", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    load_kwargs = {}
    if args.method == "qlora":
        load_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": {"": int(os.environ.get("LOCAL_RANK", 0))},
        }
    model = DecisionModel.from_pretrained(
        args.model, decision_tokens=args.decision_tokens, **load_kwargs
    )
    trainer = DecisionTrainer(
        model=model,
        dataset=args.dataset,
        sft_config=SFTConfig(
            method=args.method,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            precision=args.precision,
            label_smoothing=args.label_smoothing,
            gradient_checkpointing=args.gradient_checkpointing,
        ),
    )
    history = trainer.train(stages=["decision_sft"])
    trainer.save_pretrained(args.output)
    if os.environ.get("RANK", "0") == "0":
        print(f"Finished {len(history)} optimizer steps; last step: {history[-1]}")


if __name__ == "__main__":
    main()
