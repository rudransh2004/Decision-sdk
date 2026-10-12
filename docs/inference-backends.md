# Decision inference and vLLM

[Back to the README](../README.md) ·
[Supported models](model-support.md) ·
[LLM conversion](conversion.md)

> This public repository is a documentation preview; the SDK source and
> installation files have not yet been published here.

**“Loads in vLLM” is not the same as “makes the trained decision.”** A
publisher's decision model may require a separate head, a specialized
prompt, a temperature, or an external server. Serving only its LM weights
through `vllm serve` does not preserve that behavior.

## Which readouts does this SDK serve through vLLM?

- **Bosun v3.1 0.6B and 1.7B:** export the released checkpoint through
  `DecisionModel.from_bosun(...).save_pretrained(...)` or the SDK CLI, then
  use the exported directory with `VLLMEngine` / `--backend vllm`. This
  retains Bosun's candidate-token prompt; it is not direct serving of the
  upstream Hub repository. Bosun was also used through this export path
  for the published-weight DecisionBench comparisons in the
  [technical reference](../README.md#completed-17b-comparison-10-october-2026).
  That vLLM comparison used 1.7B; it does not establish 0.6B benchmark parity.
- **Decider 4B, JPT 4B/9B, Eikos 4B/27B:** SDK importers can export a
  self-contained checkpoint whose text decision format has a vLLM adapter.
  The adapters have fixture tests (including an opt-in tiny-GPU Decider
  test); **published-weight vLLM numerical parity is not established**.
  Decider and JPT need compatible vLLM 0.29+ / Transformers 5.17+;
  Eikos needs vLLM 0.30+ / Transformers 5.12+. Their released image
  paths are outside these text-decision adapters.
- **SDK-trained token-readout models:** a completed standalone export,
  including a trained Bosun-style conversion, can use the token-logprob
  vLLM path. A newly converted but untrained LM does not make meaningful
  decisions merely because the backend starts.
- **OpenJev / OpenJev-Flash:** the SDK calls the **publisher's System One
  HTTP helper**, which uses vLLM behind that server. This is not an
  `VLLMEngine` checkpoint importer; set the publisher helper's `base_url`
  when loading the native model.

**No decision-aware SDK vLLM adapter** is provided for Kev-9B, Laya
(either variant), Clef/Clef-Flash, GLiNER2.5-Decide, Perplexity Decider,
the Nox/Lux 1.0 and 2.0 releases, or Liquid d1-3B. In particular,
`vllm-sr` in Nox/Lux repository names refers to the publisher's
ecosystem, **not** an SDK `VLLMEngine` capability. Use their supported
native/PyTorch inference paths. Converted **pointer** and **letter-code**
custom heads likewise need PyTorch; serving the underlying LM via vLLM
would discard those trained readouts.

## Minimal export-to-API flow

Once the SDK source is published, export in a Bosun-compatible environment:

```bash
python -m pip install -e ".[bosun]"
futurix-decision export \
  --model Hanno-Labs/bosun-v3.1-0.6b --model-format bosun \
  --revision 5e78a37c8d86860ebac9ef2d5327d6299498498e \
  --output ./bosun-vllm
```

Serve from the exported directory in a **separate compatible vLLM
environment** (Bosun's Transformers 4 extra conflicts with several
publisher integrations' Transformers 5 requirements):

```bash
python -m pip install -e ".[vllm,serve]"
futurix-decision serve --backend vllm --model ./bosun-vllm \
  --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/v1/decision/choice \
  -H 'content-type: application/json' \
  -d '{"state":"Customer charged twice","choices":["billing","technical"]}'
```

`VLLMEngine` requires an SDK-exported `decision_config.json` and the
underlying standalone model/tokenizer. It performs a restricted one-token
step, reads only the candidate logprobs, applies stored calibration, and
returns typed decisions; it does not generate a free-form answer. A
Decider `score` may require one prefill per level. The bare `vllm serve`
command exposes vLLM's own API, **not** `/v1/decision`.

The SDK adapter can abstain using a `confidence_threshold`, but it
currently scores **candidate tokens only**, not a separately trained
null/abstain slot; use PyTorch for any model relying on a learned null
readout. Match the export's tokenizer, compatible vLLM version, GPU
capacity, and maximum context length to the model you choose. See the
[README model matrix](../README.md#supported-decision-models) for the
per-release status.
