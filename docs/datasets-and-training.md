# Datasets and training

[Back to the project overview](../README.md) ·
[Getting started](getting-started.md) ·
[Model support](model-support.md)

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
answer, **not** an abstention target.

```python
from futurix_decision import load_decisions

rows = load_decisions("examples/data/support_decisions.jsonl")
assert rows
```

The loader also accepts an in-memory row or iterable, `.json` files, and
Hugging Face dataset IDs after installing `.[datasets]`. Use
`load_decisions(source, streaming=True)` for large datasets; choose a named
split with `split=...`. Common external aliases such as `choices`, `label`,
`context`, and `query` are normalized, but missing or ambiguous labels
raise an error rather than being guessed. Multi-question System One records
are expanded into individual examples.

The optional public corpus adapter is
[`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3):

```python
from futurix_decision.datasets import load_open_decisions

train = load_open_decisions()  # streams the train split by default
first = next(train)
```

Its targets are distributions over the supplied options, not a dedicated
learned abstain class. Other datasets with unanswerable or out-of-scope
examples need a deliberate mapping to the readout you are training.
Keep validation, calibration, and final evaluation data separate.

## Convert and train a compatible causal LM

```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

model = load_model(
    "HuggingFaceTB/SmolLM2-135M",
    convert=True,
    strategy="pointer",
    max_sequence_length=512,
)
trainer = DecisionTrainer(
    model=model,
    dataset="decisions.jsonl",
    sft_config=SFTConfig(method="full", batch_size=2),
)
trainer.train(stages=["decision_sft"])
trainer.export("./my-decision-model")
```

New pointer conversions freeze the backbone by default: this `full` method
trains the head's unfrozen parameters. Set `train_backbone=True` to train
the full LM, or choose `method="lora"` for adapters. `strategy="bosun"`
instead adds 256 fresh decision-token rows and defaults to a trainable
backbone. Both conversions **start untrained**. Bosun-style token rows must
be added before quantizing a base for QLoRA. Full SFT is possible; LoRA is
an efficiency choice, not a requirement. See the
[conversion guide](conversion.md) for architecture and tokenizer requirements.

For an already trained native model, use its publisher-specific fine-tuning
path through the SDK:

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

## Calibrate, improve, and evaluate

`DecisionTrainer` accepts stages named `decision_sft`, `calibration`, and
`rlcd`. Calibration fits a temperature on a distinct held-out dataset;
RLCD uses proper-scoring rewards over decision probabilities. If RLCD changes
the policy, a held-out temperature is fitted again. A final test split must
remain untouched by training or checkpoint selection.

```python
from futurix_decision import DecisionTrainer, SFTConfig

trainer = DecisionTrainer(
    model=model,
    dataset="train.jsonl",
    calibration_dataset="calibration.jsonl",
    sft_config=SFTConfig(method="lora", batch_size=4),
)
trainer.train(stages=["decision_sft", "calibration", "rlcd"])
trainer.export("./trained-decision-model")
```

The full [README reference](../README.md#full-training-pipeline) covers
stage configuration, LoRA/QLoRA, calibration, and reproducible evaluation.
Example smoke-test rows are not evidence of generalization.
