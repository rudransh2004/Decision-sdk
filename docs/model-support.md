# Choose a model

[Back to the project overview](../README.md) ·
[Getting started](getting-started.md) ·
[Datasets and training](datasets-and-training.md)

The model name, its readout, and its license all matter. Loading a language
model backbone alone does **not** reproduce a publisher's decision model.
`load_model(...)` preserves supported native readouts; use `convert=True`
only for a plain, compatible causal LM that you intend to train.

## Native decision releases

- **Bosun v3.1:** `Hanno-Labs/bosun-v3.1-0.6b` and `-1.7b` have inference
  and SDK decision SFT/LoRA/RLCD continuation. They use their original
  trained decision-token rows and stable-slot prompt. Install `.[bosun]`.
- **Laya:** `convaiinnovations/laya` and `laya-multilingual` have inference
  and native fine-tuning through the SDK. Install `.[laya]`.
- **Kev:** the verified `jaredpalmer/kev-9b` path retains its native
  pointer head and supports inference and publisher-backed LoRA/head
  fine-tuning; install the publisher's compatible `kev` package.
- **Clef and Clef-Flash:** `Cloudflare/clef` and `Cloudflare/clef-flash`
  support native inference and head-only or full fine-tuning. The loader
  requires reviewed remote code and an immutable 40-character revision.
  Install `.[clef]`.
- **GLiNER2.5-Decide:** `fastino/GLiNER2.5-Decide` supports its native
  encoder decision path and hard-label training via `.[gliner-train]`.
  Soft labels require an explicit `target_policy="argmax"` conversion.
- **Perplexity Decider:** `perplexity-ai/pplx-decider-v1.1-27b` supports
  native inference and readout-only or full fine-tuning using the
  publisher's installed runtime and a pinned revision. It is large;
  check disk and accelerator memory before loading.
- **Decider, JPT, Eikos:** `Mapika/decider-4b`, `kirp/jpt-4b`/`-9b`, and
  `caiovicentino1/Eikos-4B`/`-27B` have model-specific text readouts and
  SDK training/export paths. They require compatible Transformers 5
  environments; released image paths are not covered by these adapters.
- **Nox and Lux:** `vllm-sr/Decision-{1.0,2.0}-Nox-4B` and
  `vllm-sr/Decision-{1.0,2.0}-Lux-9B` support native SDK inference.
  The released wrappers do not permit the SDK's training path.
- **Liquid d1:** `LiquidAI/d1-3B` supports native text/image inference;
  training/export has not been verified. Check Liquid's model license.
- **OpenJev:** supported inference goes through the publisher's System
  One helper in front of vLLM, not a local weights-directory importer.
  Native fine-tuning is not implemented.

For exact checkpoint IDs, model-specific limits, environment extras,
verified paths, and caveats, see
[selected native decision models](../README.md#selected-native-decision-models).
This list does not imply that every checkpoint supports vLLM. Some vLLM
paths require an SDK-exported checkpoint and a compatible vLLM version;
pointer-head conversion uses PyTorch inference.

## Converting a plain LM

```python
from futurix_decision import load_model

pointer = load_model("your-org/text-causal-lm", convert=True, strategy="pointer")
bosun_style = load_model("your-org/text-causal-lm", convert=True, strategy="bosun")
```

These are two **new, untrained** readouts, not copies of a publisher's
decision model. The pointer readout scores candidates directly and does
not add decision vocabulary rows; Bosun-style conversion creates 256
token rows with a reserved null slot. Parameter count does not impose an
SDK limit, but storage, memory, model architecture, and tokenizer
compatibility still do. Encoder-only, seq2seq, multimodal, API-only, and
unsupported custom-code checkpoints are not general conversion targets.

## Licenses, downloads, and trust

The SDK's MIT license covers this repository's code, not upstream weights,
training corpora, or publisher dependencies. Some checkpoints restrict
commercial use. The SDK downloads missing model files through the normal
Hugging Face cache and accepts local directories as well. When a model
requires `trust_remote_code=True`, review the upstream implementation,
pin an immutable revision, and use an isolated environment.

The source project's tests exercise tiny local fixtures; published-checkpoint
smoke tests and reported comparisons are described in the
[README](../README.md#verification-and-limitations). A successful load or
toy fine-tuning run is not a claim of benchmark parity.
