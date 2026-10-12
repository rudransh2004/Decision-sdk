# Convert an LLM into a decision model

[Back to the README](../README.md) ·
[Datasets and training](datasets-and-training.md) ·
[Supported released decision models](model-support.md)

`load_model(base, convert=True, strategy=...)` starts with an **ordinary,
untrained-for-decisions LM**. Choose its readout, train on labeled decisions,
evaluate on untouched cases, then calibrate. None of these strategies imports
Jev, Bosun, or Perplexity Decider's trained decision weights.

## Three training strategies

| Raw-LLM strategy | How it scores a choice | Choices per question | Vocabulary changes | Initial training policy |
|:--|:--|:--|:--|:--|
| `pointer` (default) | A trainable candidate-aware head scores candidate span endpoints and query state (`span-v1`). | 2–256; head width follows the supplied options | None | Backbone frozen; train the head, or opt into backbone updates |
| `bosun` | LM logits for 256 fresh decision tokens under a stable-slot ChatML/JSON prompt | 2–255 supplied; slot 255 reserved for null | Add 256 tokenizer/input/output embedding rows | Backbone trainable; new token rows must be learned |
| `letter_code` | A separate fixed-width head initialized from existing LM output rows for single-token `A`, `B`, …, `AA`, … codes | 2–255, **only as many as this tokenizer actually supports** | None | Backbone trainable; readout initialized, not decision-trained |

All three optimize **candidate logits** with hard- or soft-label decision SFT;
they can be followed by held-out temperature calibration and RLCD. `full`
SFT updates only unfrozen parameters. Set `train_backbone=True` with `pointer`
to train the LM as well. LoRA is an efficiency choice, not a requirement.
Tiny-fixture full SFT, LoRA, export and reload paths exist for all three;
QLoRA on 4-bit pointer bases and prepared Bosun bases has opt-in CUDA tests.
Do not assume untested 4-bit `letter_code` combinations work on every
architecture.

`pointer` uses segment-wise tokenized candidate spans, not a chat template
or new decision-token vocabulary. `bosun` uses the tokenizer's chat template
with `enable_thinking=False`; on a base without one it saves a generic
role-text fallback (not publisher prompt parity). `letter_code` also requests
a non-thinking assistant prefix if a chat template exists, then checks that
each answer code is one distinct token **after that prefix**. It fails with a
count if the tokenizer cannot provide `decision_tokens` codes; the default
255 is not guaranteed. Start with `decision_tokens=8` or your actual maximum
choice count and validate the tokenizer. Disabling thinking in a template
does not prove a reasoning-tuned model will obey the decision prompt.

Bosun adds `<|decision_000|>` through `<|decision_255|>`. Slot 255 is
structurally reserved, **not currently a learned abstention class in the
decision SFT loss**. No strategy should be advertised
as having supervised null training solely because a slot is reserved.
`confidence_threshold` offers inference-time abstention separately.

## Train and export through the SDK

The *same* dataset API applies to every strategy. Example `decisions.jsonl`:

```json
{"id":"ticket-1","kind":"choice","state":"Customer was charged twice","question":"Which team?","options":["billing","technical"],"target":"billing"}
```

```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

model = load_model(
    "HuggingFaceTB/SmolLM2-135M", convert=True,
    strategy="pointer", decision_tokens=16,
)
trainer = DecisionTrainer(
    model=model, dataset="decisions.jsonl",
    sft_config=SFTConfig(method="full", batch_size=2),
)
trainer.train(stages=["decision_sft"])
trainer.export("./my-decision-model")
restored = load_model("./my-decision-model")  # no convert=True on a trained export
```

To select another readout, change only the conversion call (and re-train):

```python
bosun_style = load_model(
    "your-org/text-causal-lm", convert=True, strategy="bosun",
)  # always adds 256 tokens; use an unquantized base

fixed_codes = load_model(
    "your-org/text-causal-lm", convert=True,
    strategy="letter_code", decision_tokens=8,
)  # tokenizer must have eight usable single-token codes
```

Pass either returned model to `DecisionTrainer` as above. For full pointer
backbone SFT, pass `train_backbone=True` to `load_model`. For LoRA, set
`SFTConfig(method="lora")`; for RLCD add `RLCDConfig` and stages
`["decision_sft", "calibration", "rlcd"]` with a distinct
`calibration_dataset`. Recalibrate after RLCD and keep final test data
untouched. [Training stages and dataset shapes](datasets-and-training.md).

For QLoRA, `pointer` can attach to a supported already-quantized 4-bit
base. `bosun` **cannot**: add its 256 rows unquantized, save the prepared
checkpoint, reload it with `BitsAndBytesConfig(load_in_4bit=True)`, then
train with `SFTConfig(method="qlora")` on a supported single GPU. Never
silently initialize fresh token rows inside a quantized embedding.

## Which open LLM checkpoints can be converted?

The table lists **raw base, instruction/chat, and reasoning checkpoints**,
not pretrained decision models. The listed architectures were checked against
the public Hugging Face `config.json` metadata on 12 October 2026. A config
is **not** a successful weight-load, tokenizer test, SFT run, or benchmark.

**C** = text-causal architecture passes the generic routing check; the
particular released checkpoint and strategy still need tokenizer, memory,
prompt, forward/backward, and export tests. **S** = explicit SDK Qwen3.5
route; synthetic family tests or mocked routing tests vary by row,
**not** published-weight verification.
**—** = rejected by the current generic conversion route. For every `C`
under `letter_code`, single-token answer codes are an **additional
mandatory runtime check**; a smaller `decision_tokens` may be needed.

| Hugging Face source checkpoint | Variant / declared architecture | `pointer` | `bosun` | `letter_code` |
|:--|:--|:--:|:--:|:--:|
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | base · `LlamaForCausalLM` | C | C | C |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | chat · `LlamaForCausalLM` | C | C | C |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) / [1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | base · `Qwen3ForCausalLM` | C | C | C |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | non-thinking chat · `Qwen3ForCausalLM` | C | C | C |
| [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | reasoning/chat · `Qwen3ForCausalLM` | C | C | C |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | chat · `Qwen2ForCausalLM` | C | C | C |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | reasoning · `Qwen2ForCausalLM` | C | C | C |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | chat · `MistralForCausalLM` | C | C | C |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | chat · `Phi3ForCausalLM` | C | C | C |
| [OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) | chat · `Olmo2ForCausalLM` | C | C | C |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | reasoning · `GptOssForCausalLM` | C | C | C |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) / [9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | hybrid text/vision `Qwen3_5ForConditionalGeneration` | S | S | S |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen3.5-family hybrid `Qwen3_5ForConditionalGeneration` | — | — | S |

The `Qwen3.5-9B` IDs have dedicated pointer/Bosun loading routes. For
Qwen3.5-family text decisions, `letter_code` uses an explicit multimodal
**loader** but rejects image inputs, applies noncausal masking only to full
attention (retaining recurrent layers), and requires Transformers 5.17+ and
SDPA. `Qwen3.8-27B` has that letter-code route by config; its pointer and
Bosun conversion paths are currently rejected because they are not in the
9B special-case loader. These are **routing statements**, not claims of
released-27B training or production readiness.

`meta-llama/Llama-3.1-8B-Instruct` and
`google/gemma-3-1b-it` require gated artifact access here; they are
**not** asserted compatible from an unavailable config. If you have access,
inspect the actual architecture and repeat the conversion/tokenizer checks
first. Larger Gemma 3 / Mistral 3 multimodal variants, encoder-only,
seq2seq, API-only, and arbitrary `*ForConditionalGeneration` checkpoints
are not generic conversion targets. A *trained* Bosun, Laya, Clef,
Perplexity Decider, etc. should be loaded via its native SDK importer,
**not** reconverted as a raw LM.

## Verification boundary

Tiny locally created GPT-2 and Llama weights exercise pointer and Bosun
conversion/export; tiny GPT-2 exercises letter-code SFT/RLCD/export, and a
tiny Qwen3.5 text-decision fixture exercises its letter-code route. The
released SmolLM2-135M pointer/Bosun smoke tests are **opt-in**, not
evidence of a benchmark result or a run on every release listed above.
The pointer/letter-code heads use SDK PyTorch inference; Bosun-style
trained-token exports are the vLLM-oriented conversion path. See
[inference backends](inference-backends.md) for the precise adapter limits.

Model size has no SDK API cap, but real conversion needs a loadable model,
tokenizer and output embeddings (plus accessible text hidden states for
heads), enough GPU RAM/disk, appropriate Transformers/PEFT versions, and
trainable data. For custom remote code, review it and pin an immutable
40-character Hub revision. A reasoning model is **not** a trained decision
model merely because it can produce a letter or decision token.
