# Decision SDK

![Decision SDK: from a language model to a trained decision readout and runtime choices](docs/assets/decision-sdk-hero.svg)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![MIT license](https://img.shields.io/badge/License-MIT-2ea44f)

Decision SDK is a Python library for typed decisions. You give a model a
state, a question, and the options you want it to choose between. It returns
probabilities over those options instead of free-form text. The same
`choice`, `noul` (yes/no), and `score` calls work on local models, vLLM, and a
remote server. You can also convert a compatible open LLM into a decision
model, or continue training a released decision checkpoint on your own labeled
data, then calibrate, export, and serve it. "Jev-style" describes the
typed-decision workflow. It does not promise parity with Jev or any
publisher's model. The model weights are downloaded separately.

## Install

```bash
pip install "futurix-decision[torch]"   # local models: torch and transformers
pip install futurix-decision            # client and data only, no torch
```

Each model family has its own extra, for example `[laya]`, `[bosun]`, or
`[decider]`. `[serve]` adds the server, `[train]` the trainers, and `[vllm]`
the vLLM backend. [Model support](docs/model-support.md) lists every extra.

## Quickstart: local

<!-- test -->
```python
from futurix_decision import load_model

model = load_model("laya")  # pip install "futurix-decision[laya]"
result = model.choice(
    state={"ticket": "Customer was charged twice"},
    question="Which team owns this request?",
    choices=["billing", "technical"],
)
print(result.decision, result.probabilities)
print(model.noul(state="Invoice paid", question="Was it paid?").decision)  # True or False
print(model.score(state="Great support", question="Rate satisfaction.", max_score=5).decision)
```

`result.decision` is the chosen option, `True` or `False` for `noul`, an int
level for `score`, and `None` if the model abstained below your
`confidence_threshold`.

## Quickstart: HTTP

```bash
pip install "futurix-decision[laya,serve]"
futurix-decision serve --model laya --model-format native --api-key "$FUTURIX_API_KEY"
```

<!-- test -->
```python
from futurix_decision import DecisionClient  # needs only the base install

with DecisionClient("http://127.0.0.1:8000", api_key="secret") as client:
    result = client.choice(state="Customer was charged twice", choices=["billing", "technical"])
    print(result.decision, result.probabilities)
```

The server micro-batches concurrent requests, supports bearer auth, and maps SDK
errors to HTTP statuses. [Serving](docs/serving.md) has the details.

## Supported decision models

| Released decision model | SDK inference | Train on your dataset | Fine-tune release | vLLM decisions |
|:--|:--:|:--:|:--:|:--:|
| [Bosun v3.1 0.6B](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b) / [1.7B](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b) | ✅ | ✅ SFT / RLCD | ✅ | ✅ export |
| [Laya](https://huggingface.co/convaiinnovations/laya) / [Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | ✅ | ✅ native | ✅ | — |
| [Kev 9B](https://huggingface.co/jaredpalmer/kev-9b) | ✅ | ✅ native | ✅ | — |
| [Clef](https://huggingface.co/Cloudflare/clef) / [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) | ✅ | ✅ native | ✅ | — |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | ✅ | ✅ hard labels | ✅ | — |
| [Perplexity Decider 27B](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | ✅ | ✅ native | ✅ | — |
| [Decider 4B](https://huggingface.co/Mapika/decider-4b) | ✅ | ✅ SFT / RLCD | ✅ | 🧪 export |
| [JPT 4B](https://huggingface.co/kirp/jpt-4b) / [9B](https://huggingface.co/kirp/jpt-9b) | ✅ text | ✅ SFT / RLCD | ✅ | 🧪 export |
| [Eikos 4B](https://huggingface.co/caiovicentino1/Eikos-4B) / [27B](https://huggingface.co/caiovicentino1/Eikos-27B) | ✅ text | ✅ SFT / RLCD | ✅ | 🧪 export |
| Nox [1.0](https://huggingface.co/vllm-sr/Decision-1.0-Nox-4B) / [2.0](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B) | ✅ | — | — | — |
| Lux [1.0](https://huggingface.co/vllm-sr/Decision-1.0-Lux-9B) / [2.0](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B) | ✅ | — | — | — |
| [Liquid d1 3B](https://huggingface.co/LiquidAI/d1-3B) | ✅ | — | — | — |
| [OpenJev](https://huggingface.co/openjev/openjev) / [Flash 9B](https://huggingface.co/openjev/OpenJev-Flash-9B) | ✅ server | — | — | ↗ helper |

A check in "Train on your dataset" means an SDK path for training on labeled
decision data, not pretraining a publisher architecture from scratch. Each
check is model-specific. Some paths need publisher packages, pinned remote
code, a lot of GPU memory, or hard labels. OpenJev needs its running publisher
server. Nox, Lux, and d1 are inference-only here.
[See requirements and limits](docs/model-support.md).

vLLM legend: ✅ export means decision-aware SDK inference from an exported
checkpoint. 🧪 export means the adapter and fixture path exists but is not
verified for numerical parity on the published weights. ↗ helper is the
publisher's separate vLLM-backed server. — means no SDK vLLM decision adapter.
`vllm-sr` in a repo name does not mean `VLLMEngine` can serve its decision
head. [Backend requirements and examples](docs/inference-backends.md).

Have a plain causal LM instead? `load_model("your-org/lm", convert=True)`
attaches an untrained readout that you train on your data.
See [conversion](docs/conversion.md).

## Documentation

- Tutorial: [Getting started](docs/getting-started.md)
- How to:
  [run decisions](docs/inference.md),
  [serve over HTTP](docs/serving.md),
  [train and calibrate](docs/datasets-and-training.md),
  [convert an LLM](docs/conversion.md),
  [pick a vLLM backend](docs/inference-backends.md)
- Reference:
  [model support](docs/model-support.md),
  [API reference](docs/api.md),
  [benchmarks and run logs](docs/benchmarks.md),
  [CHANGELOG](CHANGELOG.md)
- Contributing: [CONTRIBUTING](CONTRIBUTING.md), including how to add a model family

Build the docs locally with `pip install -e ".[docs]"` and `mkdocs serve`.

The SDK's MIT license covers this repository's code, not upstream model weights
or training data. Check each checkpoint's license before you use it.
