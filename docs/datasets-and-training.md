# Datasets and training

Decision SDK uses a validated decision record rather than assuming a
universal dataset column convention. A record contains an `id`, `state`,
`question`, ordered `options`, and a `target`. `kind` is `choice`, `noul`
(yes/no), or `score` (ordered levels).

## The canonical JSONL row

```json
{"id":"case-42","kind":"choice","state":{"issue":"duplicate charge"},"question":"Who should handle this?","options":["billing","technical"],"target":"billing"}
```

`target` can be an option label, its index, or a probability distribution in
option order. For example, `[0.8, 0.2]` expresses a soft target for the two
teams. `noul` uses `["false", "true"]`; the label `false` is an ordinary
answer, not an abstention target.

<!-- test -->
```python
from futurix_decision import load_decisions

rows = load_decisions("examples/data/support_decisions.jsonl")
print(len(rows), rows[0].options, rows[0].target)  # 10 ['billing', 'technical'] billing
```

`examples/data/support_decisions.jsonl` has hard-labeled `choice`, `noul`, and
`score` rows and one grouped System One record. `load_decisions` parses both
forms into ten validated `DecisionExample` rows.

A target can also be a distribution in option order. Invalid targets are
rejected, never silently renormalized. `score` needs consecutive integer
levels (default 0 to 5).

<!-- test -->
```python
from futurix_decision.datasets import load_decisions

dataset = load_decisions(
    {
        "id": "procurement-001",
        "kind": "choice",
        "state": {"gst_certificate": "expired", "budget": "approved"},
        "question": "What should happen next?",
        "options": ["approve", "request_document", "reject", "escalate"],
        "target": [0.02, 0.78, 0.05, 0.15],
    }
)
assert dataset[0].target_distribution() == [0.02, 0.78, 0.05, 0.15]
```

`load_decisions` accepts a row, an iterable of rows, local `.json` and `.jsonl`
files, and Hugging Face dataset IDs (install the `datasets` extra). Set
`streaming=True` for a lazy iterator, and `split="calibration"` or another
split when the source has one. Common aliases (`choices`, `candidates`,
`label`, `answer`, `probabilities`, `query`, `context`) are normalized. Missing
or ambiguous labels raise an error rather than being guessed.

Rows such as `{"state": "...", "question": "...", "options": ["yes", "no"],
"label": "yes"}` are the default interchange. There is no single universal
decision-dataset column standard.

### System One rows

For multi-question decisions, use one System One record per line with `state`,
typed `questions`, and `targets` (also accepts `expected` or `gold`) keyed by
question ID:

```json
{"id":"ticket-7","state":"Card charged twice","questions":{"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Refunds","support":"Other"}},"urgent":{"type":"noul","instructions":"Is this urgent?"}},"targets":{"route":{"probabilities":{"billing":0.9,"support":0.1}},"urgent":false}}
```

`load_decisions` expands these into validated examples, preserving the shared
`group_id`, `question_id`, options, descriptions and soft targets. Training
and evaluation also accept these rows in memory. A label or distribution is
required for every question; arbitrary unlabeled text does not become a
decision dataset just because it is in JSONL. Conversation states can be a
JSON list; structured publisher questions remain available as
`example.native_question`. Optional `images` file paths or data URLs are
retained for Clef/PPLX native training and Clef/PPLX/d1 inference; text-only
trainers reject them. Video and generic media rows require an explicit
model-specific adapter. Keep held-out evaluation and calibration rows separate.

### The public corpus

The optional public corpus adapter is
[`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3):

<!-- test-skip: downloads a 655,806-row corpus -->
```python
from futurix_decision.datasets import load_open_decisions

dataset = load_open_decisions()  # streaming train split; 655,806 rows
example = next(dataset)
```

For an independent held-out evaluation, use `split="test_set_30k"` rather than
the train, validation, or calibration splits. Credit the dataset's creators if
you use it; the corpus is Apache-2.0 licensed and is not bundled in this wheel.

Its targets are distributions over the supplied options, not a dedicated
learned abstain class. Other datasets with unanswerable or out-of-scope
examples need a deliberate mapping to the readout you are training.
Keep validation, calibration, and final evaluation data separate.

## Train a new decision model

Convert a compatible causal LM, train it on labeled decisions, and export it.
The conversion starts untrained. This example uses a small base so it runs
quickly; [conversion](conversion.md) covers the three readouts and which
checkpoints they accept.

<!-- test -->
```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

model = load_model(
    "HuggingFaceTB/SmolLM2-135M",
    convert=True,
    strategy="pointer",
    max_sequence_length=512,
    decision_tokens=32,  # maximum options per question
)
trainer = DecisionTrainer(
    model=model,
    dataset="decisions.jsonl",
    sft_config=SFTConfig(method="full", batch_size=2),  # head-only by default
)
trainer.train(stages=["decision_sft"])
trainer.export("./my-decision-model")

restored = load_model("./my-decision-model")  # no convert=True on a trained export
print(restored.choice(state="I was charged twice", choices=["billing", "technical"]))
```

This quickstart updates the pointer head; to train the backbone too, set
`train_backbone=True` when converting or use LoRA. For reliable confidence,
calibrate on separate held-out data; one demo row cannot establish quality.

Already have a decision model? Continue its native readout instead:

```python
from futurix_decision import NativeSFTConfig, load_model

model = load_model("laya", device="cuda")
trained = model.finetune(
    "decisions.jsonl",
    "./my-laya",
    config=NativeSFTConfig(epochs=1, batch_size=2),
    device="cuda",
)
```

Only supported families expose that native training recipe. See
[model support](model-support.md) before selecting a checkpoint.

## Fine-tune a released model

These walkthroughs use `examples/data/support_decisions.jsonl` (see
[the canonical JSONL row](#the-canonical-jsonl-row)).

The Bosun checkpoints keep their trained token readout. This example continues
one with LoRA, exports a complete checkpoint, and loads it back:

```python
import torch
from futurix_decision import DecisionTrainer, SFTConfig, load_decisions, load_model

rows = load_decisions("examples/data/support_decisions.jsonl")
model = load_model("bosun-0.6b", dtype=torch.bfloat16, device_map={"": "cuda"})
before = model.choice(state={"ticket": "Charged twice"}, choices=["billing", "technical"])
trainer = DecisionTrainer(
    model=model,
    dataset=rows[:4],
    sft_config=SFTConfig(
        method="lora",
        epochs=1,
        batch_size=1,
        precision="bf16",
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=["q_proj", "v_proj"],
    ),
)
trainer.train(stages=["decision_sft"])
trainer.export("./checkpoints/bosun-support")
after = load_model("./checkpoints/bosun-support", device_map={"": "cuda"})
print(
    before.probabilities,
    after.choice(state={"ticket": "Charged twice"}, choices=["billing", "technical"]).probabilities,
)
```

Native models keep their publisher heads and use the SDK's `NativeSFTConfig`
for basic tuning controls. Their advanced publisher training configurations
are available through `config=`:

```python
from futurix_decision import NativeSFTConfig, load_decisions, load_model

rows = load_decisions("examples/data/support_decisions.jsonl")
model = load_model("laya", device="cuda")
print(model.noul(state="Invoice paid", question="Was it paid?").probabilities)
trained = model.finetune(
    rows,
    "./checkpoints/laya-support",
    config=NativeSFTConfig(epochs=1, batch_size=2),
    device="cuda",
)
reloaded = load_model(trained.checkpoint, device="cuda")
```

`examples/test_released_decisions.py` runs one checkpoint at a time with SDK
calls for dataset load, `choice`, `noul`, `score`, `system_one`, fine-tuning,
export, and checkpoint reload:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/test_released_decisions.py bosun-0.6b \
  --train --output /tmp/bosun-support --max-examples 4
CUDA_VISIBLE_DEVICES=1 python examples/test_released_decisions.py laya \
  --train --output /tmp/laya-support --max-examples 4
# GLiNER also works on CPU; its publisher trainer stores the release under output/final.
CUDA_VISIBLE_DEVICES='' python examples/test_released_decisions.py gliner \
  --device cpu --train --output /tmp/gliner-support --max-examples 2
```

The driver also accepts `bosun-1.7b`, `laya-multilingual`, `kev`,
`clef-flash`, `clef`, and `pplx-decider`. Install each publisher's extra
first; every family uses Transformers 5.17 or later, below 6. The driver pins
Clef and PPLX remote code and every released model revision. A 27B release needs enough disk for
both its download and its full native export (roughly 105–110 GB).
To run these large releases one after another without keeping duplicate
weights:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/run_released_ephemeral.py clef-flash --max-examples 8
CUDA_VISIBLE_DEVICES=1 python examples/run_released_ephemeral.py clef --max-examples 8
CUDA_VISIBLE_DEVICES=1 python examples/run_released_ephemeral.py pplx-decider --max-examples 8
```

Each command uses a separate temporary Hugging Face cache and native export;
the export is reloaded and exercised before both are deleted. Pre-existing
caches and checkpoints are untouched. Set `--scratch` to a writable disk with
at least 50 GiB free for Clef-Flash or 112 GiB for each 27B release. The
walkthrough frees the original GPU model before reloading a 27B export.

Four to ten demo labels do not calibrate probabilities or establish
generalization; use separate held-out data for that.

The results of those runs are in [benchmarks](benchmarks.md).

## Decision SFT

SFT gathers the assigned candidate logits for each example, masks all other
decision tokens, and computes hard CE (for labels/indices) or soft CE (for
distributions). `label_smoothing` mixes targets with a uniform distribution
over available candidates only. Slot assignments change deterministically
between training batches and epochs.

```python
from futurix_decision import DecisionModel, DecisionTrainer, SFTConfig

model = DecisionModel.from_pretrained("your-decoder-checkpoint")
trainer = DecisionTrainer(
    model=model,
    dataset="decisions.jsonl",  # or "SargeDev/jev-distill-corpus-v3"
    sft_config=SFTConfig(
        method="lora",
        epochs=1,
        batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        precision="bf16",
        gradient_checkpointing=True,
        lora_rank=16,
        lora_alpha=32,
    ),
)
history = trainer.train(stages=["decision_sft"])
trainer.save_pretrained("./decision-sft")
```

`method="full"` trains the entire decoder (including new decision tokens).
`method="lora"` trains adapters and decision-token rows; untied output
projections are also trainable. On save, LoRA merges into a standalone Hugging
Face checkpoint. `method="qlora"` requires a previously saved checkpoint with
decision tokens already added:

<!-- test -->
```python
from futurix_decision import DecisionModel

base = DecisionModel.from_pretrained("your-decoder-checkpoint")
base.save_pretrained("./base-with-decision-tokens")
del base
```

Then run the script below with `--method qlora`. QLoRA saves a PEFT adapter
referencing that base; keep the base accessible to load the adapter with
`DecisionModel.from_pretrained`. Do not save the adapter over its base.

The following is an example training script, not the planned
`futurix-decision` command:

```bash
python examples/train_sft.py \
  --model ./base-with-decision-tokens \
  --dataset decisions.jsonl \
  --method qlora --precision bf16 --gradient-checkpointing \
  --batch-size 16 --gradient-accumulation 2 \
  --output ./decision-sft-adapter
```

For multiple GPUs, run `accelerate launch --multi_gpu --num_processes 2
examples/train_sft.py ...` with an available GPU per process. The trainer
shards prepared dataloaders through Accelerate and saves only on the main
process. Iterable JSONL/HF sources reopen on each epoch; one-shot Python
generators are rejected when `epochs > 1`. BF16/4-bit training requires
supported hardware and current driver/CUDA packages. Tiny CUDA/BF16/QLoRA
and gradient-checkpointing integration tests pass; multi-GPU has not been
exercised here.

## Calibrate, improve, and evaluate

`DecisionTrainer` accepts stages named `decision_sft`, `calibration`, and
`rlcd`. Calibration fits a temperature on a distinct held-out dataset;
RLCD uses proper-scoring rewards over decision probabilities. If RLCD changes
the policy, a held-out temperature is fitted again. A final test split must
remain untouched by training or checkpoint selection.

<!-- test -->
```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

model = load_model("HuggingFaceTB/SmolLM2-135M", convert=True)
trainer = DecisionTrainer(
    model=model,
    dataset="decisions.jsonl",
    calibration_dataset="heldout.jsonl",
    sft_config=SFTConfig(method="lora", batch_size=4),
)
trainer.train(stages=["decision_sft", "calibration", "rlcd"])
trainer.export("./trained-decision-model")
```

### Calibration and proper scores

`CalibrationTrainer` optimizes cross-entropy (negative log score) plus a
configurable Brier loss. `fit_temperature` then minimizes soft-label NLL on a
different calibration split. `fit_temperature(backend, heldout)` works on any
backend, local or remote, because it fits on the returned probabilities:
`softmax(log p / T)` equals `softmax(logits / T)`. The result is relative to
the backend's current output.

Apply it with `Calibrated(backend, T)`, or store it in a `DecisionModel`
checkpoint with `model.config.temperature *= T` before `save_pretrained`.
`evaluate` reports accuracy (expected correctness for soft targets), NLL,
Brier, 15-bin ECE and MCE, mean confidence, and selective accuracy and
coverage at 0.50 to 0.99. It computes the metrics from batched masked decision
logits, not generated text.

<!-- test -->
```python
from futurix_decision import Calibrated, evaluate, fit_temperature, load_model

model = load_model("./my-decision-model")
metrics = evaluate(model, "heldout.jsonl")
calibrated = Calibrated(model, fit_temperature(model, "heldout.jsonl"))
print(sorted(metrics))
```

### RLCD

`RLCDTrainer` uses `Normal(mean=decision_logits, std=noise_sigma)` and
`Normal.sample` for `group_size` reports after one transformer forward
per batch. Its loss is `-mean(stop_gradient(R - baseline) *
log_pi(sample | mean))`. Default leave-one-out advantage excludes each
report from its own baseline; a full-group baseline is also available.
Defaults: G=8, σ=0.25, SFT anchor 0.2, KL anchor 0.01. When no reference is
provided, a frozen copy of the starting policy supplies KL; this adds one
reference forward, not G forwards. Log, Brier, spherical, and ordinal
ranked-probability rewards are independently weighted; RPS only applies
to `score` with the supplied type labels.

<!-- test -->
```python
from futurix_decision import RLCDTrainer, load_model
from futurix_decision.rewards import LogScore, SphericalScore

model = load_model("./my-decision-model")
trainer = RLCDTrainer(
    model,
    group_size=8,
    noise_sigma=0.25,
    rewards=[LogScore(weight=1.0), SphericalScore(weight=0.5)],
    sft_weight=0.2,
    kl_beta=0.01,
)
history = trainer.train("decisions.jsonl")
```

### Logging

The package logs through `logging.getLogger("futurix_decision")`, and
importing it never configures logging. Set `SFTConfig(log_stdout=True,
log_jsonl="logs/train.jsonl")` to log each step, or `wandb_project="..."` after
you install the `wandb` extra. The CLI's `--log-level` flag sends progress
messages to stderr. RLCD step logs hold total, RL, SFT, and KL losses, reward
moments, entropy, accuracy, NLL, Brier, ECE, and mean confidence.

## Full training pipeline

```python
from futurix_decision import DecisionModel, DecisionTrainer, SFTConfig

model = DecisionModel.from_pretrained("your-decoder-checkpoint")
trainer = DecisionTrainer(
    model=model,
    dataset="SargeDev/jev-distill-corpus-v3",
    sft_config=SFTConfig(
        method="lora",
        precision="bf16",
        batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        lora_rank=16,
        lora_alpha=32,
    ),
)
history = trainer.train(stages=["decision_sft", "calibration", "rlcd"])
trainer.export("./Futurix-Decision-1.7B", merge_lora=True, safetensors=True)
```

For a local dataset, pass a distinct `calibration_dataset=`; the public
SargeDev dataset has its own `calibration` split, selected automatically.
RLCD changes the policy, so temperature is refitted after RLCD using
held-out data. Without a held-out split, the stale temperature is reset to
1.0. Train uses the dataset's `train` split; evaluate on `test_set_30k`.
The returned history contains stage losses and RLCD diagnostics, not
benchmark results. Use a replayable path/collection for multiple stages.

For the Blackwell reference settings:

```bash
futurix-decision train --config configs/rtx_pro_6000.yaml
futurix-decision evaluate --model ./Futurix-Decision-1.7B
futurix-decision serve --model ./Futurix-Decision-1.7B
```

The reference config selects the public `Qwen/Qwen3-1.7B` base and SargeDev
corpus; it downloads weights/data and can take substantial time and disk.
This particular Blackwell config has not been run end-to-end; the pinned
1.7B comparison in [benchmarks](benchmarks.md) uses `configs/qwen_base_1_7b.yaml`. Override `--model`,
`--dataset`, `--output`, or choose one stage with `--method rlcd` (the latter
requires a decision-trained checkpoint). `futurix-decision export --model
./adapter --output ./merged` creates a standalone checkpoint from an adapter.
`futurix-decision evaluate --dataset decisions.jsonl --model ./merged` also
accepts local datasets. `scripts/prepare_dataset.py` streams conversions to
canonical JSONL.
