# Getting started

This tutorial installs the SDK, makes a decision with an existing model, calls
a decision server, and loads labeled data. Each step runs on its own.

The Python package is `futurix-decision`; import it as `futurix_decision`.
It needs Python 3.10 or later.

## Install

The base install has no torch. It covers the data schemas, dataset loading,
metrics, `DecisionClient`, and the HTTP runtimes:

```bash
python -m pip install futurix-decision
```

Local models need PyTorch. The `torch` extra adds it with `transformers`.
Every model-family, `train`, `serve`, and `vllm` extra includes `torch`:

```bash
python -m pip install "futurix-decision[torch]"
python -m pip install "futurix-decision[datasets,train,serve]"
```

Each model family has its own extra, such as `laya`, `bosun`, `decider`, or
`clef`. [Model support](model-support.md) lists them. Install more than one
only when you need them. Every extra that uses transformers needs
`transformers>=5.17,<6`, so they install together. Pick the PyTorch wheel that
matches your CUDA installation.

To work on the SDK from a clone, use `python -m pip install -e ".[dev,serve]"`.
See [CONTRIBUTING](https://github.com/rudransh2004/Decision-sdk/blob/main/CONTRIBUTING.md).
The SDK is not advertised as a package on PyPI yet, so install from a clone or
a release wheel until it is.

## Make a decision

<!-- test -->
```python
from futurix_decision import load_model

model = load_model("laya")  # python -m pip install "futurix-decision[laya]"
result = model.choice(
    state={"ticket": "Customer was charged twice"},
    question="Which team owns this request?",
    choices=["billing", "technical"],
)
print(result.model_dump())
```

`load_model` downloads a missing supported Hub checkpoint into the normal
Hugging Face cache. Pass a local checkpoint directory to load from disk. A
publisher model with executable remote code needs a reviewed, pinned
`revision`; see [model support](model-support.md).

The result has these fields:

| Field | Meaning |
|:--|:--|
| `kind` | `choice`, `noul`, or `score` |
| `decision` | The `choice` string, a bool for `noul`, an int level for `score`, or `None` if the model abstained |
| `confidence` | Probability of the decision |
| `probabilities` | Probability of each option you supplied |
| `abstained` | True when a `confidence_threshold` you set was not met |
| `latency_ms` | Wall time for this row |

Probabilities need calibration on held-out data before you treat them as
reliable confidence. See [datasets and training](datasets-and-training.md#calibration-and-proper-scores).
[Run decisions](inference.md) covers batches, abstention, and result types.

## Call a decision server

Serve a model, then call it from any process. The client needs only the base
install.

```bash
python -m pip install "futurix-decision[laya,serve]"
futurix-decision serve --model laya --model-format native
```

<!-- test -->
```python
from futurix_decision import DecisionClient

with DecisionClient("http://127.0.0.1:8000") as client:
    result = client.choice(state="Customer was charged twice", choices=["billing", "technical"])
    print(result.decision, result.probabilities)
```

[Serve decisions over HTTP](serving.md) covers authentication, batching, and
errors.

## Load your decision data

The smallest labeled example is a JSONL row:

```json
{"id":"ticket-1","kind":"choice","state":"Customer was charged twice","question":"Which team?","options":["billing","technical"],"target":"billing"}
```

<!-- test -->
```python
from futurix_decision import load_decisions

rows = load_decisions("examples/data/support_decisions.jsonl")
print(len(rows), rows[0].options, rows[0].target)
```

The same loader takes in-memory rows and Hugging Face dataset IDs (install the
`datasets` extra), and validates them before training. [Datasets and
training](datasets-and-training.md) covers soft labels, held-out splits, and
System One rows.

## Train your own model

Convert a plain causal LM, train it on your rows, and export it:

<!-- test -->
```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

# A plain LLM becomes a trainable decision model; it is not trained yet.
model = load_model("HuggingFaceTB/SmolLM2-135M", convert=True, strategy="pointer")
trainer = DecisionTrainer(
    model=model,
    dataset="decisions.jsonl",
    sft_config=SFTConfig(method="full"),  # trains the pointer head by default
)
trainer.train(stages=["decision_sft"])
trainer.export("./my-decision-model")

result = load_model("./my-decision-model").choice(
    state="A customer was charged twice",
    choices=["billing", "technical"],
)
print(result.decision, result.probabilities)
```

One demo row cannot establish quality. Calibrate on separate held-out data.
[Conversion](conversion.md) explains the three readouts, and
[datasets and training](datasets-and-training.md) covers the training stages.

## Check the installation

```bash
python -m pytest -q
```

The tests use tiny local checkpoints. They do not download large models.
