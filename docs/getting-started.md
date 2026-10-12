# Getting started

[Back to the project overview](../README.md) ·
[Datasets and training](datasets-and-training.md) ·
[Model support](model-support.md) ·
[LLM conversion](conversion.md) ·
[vLLM and inference](inference-backends.md)

> The public repository is a documentation preview; installation commands
> require the SDK source, which will be published separately once ready.

Decision SDK is a Python package named `futurix-decision`; import it as
`futurix_decision`. Python 3.10+ and PyTorch are required.

## Install for the model you want to use

From a clone of this repository:

```bash
python -m pip install -e .
python -m pip install -e ".[datasets,train,dev]"
```

Native checkpoints need their own dependencies. For the example below:

```bash
python -m pip install -e ".[laya]"
```

Install other extras only as needed, such as `.[bosun]`, `.[clef]`,
`.[gliner-train]`, or `.[serve]`. Bosun's Transformers 4 dependency and
several other publisher integrations' Transformers 5 dependencies are
incompatible in one environment; use separate virtual environments.
The SDK is not currently advertised here as a published PyPI package.

## Make a decision with an existing model

```python
from futurix_decision import load_model

model = load_model("laya")
result = model.choice(
    state={"ticket": "Customer was charged twice"},
    question="Which team owns this request?",
    choices=["billing", "technical"],
)
print(result.to_dict())
```

`load_model` downloads a missing supported Hub checkpoint into the normal
Hugging Face cache. Pass a local checkpoint directory to load from disk.
Publisher models with executable remote code require you to review that code
and supply a pinned revision; see [model support](model-support.md).

`result.decision` is the selected option, `result.probabilities` maps the
supplied options to probabilities, and `result.abstained` reports whether a
configured confidence threshold led the SDK to decline a choice. Probabilities
need held-out calibration before you treat them as reliable confidence.

## Bring your own decision data

The smallest labeled example is a JSONL row:

```json
{"id":"ticket-1","kind":"choice","state":"Customer was charged twice","question":"Which team?","options":["billing","technical"],"target":"billing"}
```

```python
from futurix_decision import load_decisions

rows = load_decisions("examples/data/support_decisions.jsonl")
print(len(rows), rows[0].options, rows[0].target)
```

The same loader accepts in-memory rows and Hugging Face dataset IDs (install
`.[datasets]`), with validation before training. Learn about soft labels,
held-out splits, and System One rows in
[datasets and training](datasets-and-training.md).

## Check the installation

```bash
python -m unittest discover -s tests -v
```

Tests use tiny local checkpoints; they do not download large models unless
you explicitly run one of the released-checkpoint example scripts.
