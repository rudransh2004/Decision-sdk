# Convert an LLM into a decision model

[Back to the README](../README.md) ·
[Datasets and training](datasets-and-training.md) ·
[Supported released decision models](model-support.md)

> This public repository is a documentation preview. The SDK source and
> packaging files have not yet been published here.

`load_model(base, convert=True)` starts with an **ordinary LM**, attaches a
decision readout, and records a prompt/checkpoint contract. It does not import
Jev's weights or make the base decision-trained. Supply labeled decisions,
train, evaluate on held-out cases, and calibrate before interpreting the
output probabilities.

## What the base must provide

- A local directory or Hugging Face ID with loadable weights, tokenizer, and
  a compatible **text causal LM** architecture. Model size is constrained by
  the machine, not a fixed SDK parameter-count limit.
- Enough memory for the base and chosen training method. A raw Bosun-style
  conversion needs room to add and save fresh embedding rows before 4-bit
  quantization; a pointer head can be attached to an already-quantized base.
- For custom remote model code, explicit trust in reviewed code and an
  immutable 40-character Hub revision. Do not assume a model's chat or
  thinking mode matches the SDK's decision prompt without testing it.

Encoder-only, seq2seq, API-only, and arbitrary multimodal checkpoints are
**not** interchangeable with a text causal LM. A known trained decision
checkpoint should be loaded through its own native importer, not converted
again: doing so would discard its trained readout.

## Pick the readout before training

- **`strategy="pointer"` (default):** one trainable candidate-aware head,
  dynamically sized to the supplied options. No decision-token vocabulary
  resize. The backbone is frozen by default for quick head-only SFT; use
  `train_backbone=True` or LoRA when you want to adapt the LM itself.
- **`strategy="bosun"`:** 256 newly initialized decision tokens, stable
  candidate slots, and a Bosun-style prompt. Up to 255 supplied candidates;
  slot 255 is reserved. It is *inspired by* Bosun, not a copy of a published
  Bosun checkpoint. The backbone defaults to trainable.
- **`strategy="pplx"`:** a separate fixed-code decision readout using
  existing single-token answer-code rows, with no new decision tokens.
  Its usable width depends on tokenizer code availability (at most 255).
  This is PPLX-inspired, **not** Perplexity Decider's released weights.

```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

model = load_model("HuggingFaceTB/SmolLM2-135M", convert=True, strategy="pointer")
trainer = DecisionTrainer(
    model=model, dataset="decisions.jsonl",
    sft_config=SFTConfig(method="full"),  # head-only by default
)
trainer.train(stages=["decision_sft"])
trainer.export("./my-decision-model")
```

For full SFT, allow backbone training. LoRA is optional, not required;
QLoRA requires a supported 4-bit setup. Calibration and RLCD can follow
SFT using separate held-out calibration data. The same SDK loads an exported
checkpoint without `convert=True`.

## What has actually been exercised

- Tiny **GPT-2 and Llama** local fixtures cover pointer and Bosun-style
  conversion and reload; tiny GPT-2 also covers training/export. The released
  [`SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
  has an opt-in released-weight smoke path; a smoke test is not a
  quality benchmark.
- The fixed-code path has tiny local GPT-2 and Qwen3.5-family tests.
  Qwen3.5 has an explicit model-specific text-decision route; it is
  **not** evidence that arbitrary multimodal LMs or all thinking variants
  share that contract. Published large-model training and benchmark parity
  are not established by these fixtures.

We'll add a model-by-model list of recent convertible LLMs separately,
including which chat/thinking variants have been verified. Until then,
“compatible” means that the actual architecture, tokenizer, prompt
formatting, memory requirements, and training/export path all pass
validation for the chosen strategy.
