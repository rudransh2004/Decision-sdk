# vLLM and inference backends

"Loads in vLLM" is not the same as "makes the trained decision." A publisher's
decision model may need a separate head, a specialized prompt, a temperature,
or an external server. Serving only its LM weights through `vllm serve` does
not keep that behavior.

## Which readouts does this SDK serve through vLLM?

- Bosun v3.1 0.6B and 1.7B: export the released checkpoint through
  `DecisionModel.from_bosun(...).save_pretrained(...)` or the SDK CLI, then
  use the exported directory with `VLLMEngine` / `--backend vllm`. This
  retains Bosun's candidate-token prompt; it is not direct serving of the
  upstream Hub repository. Bosun was also used through this export path
  for the published-weight DecisionBench comparisons in the
  [benchmarks](benchmarks.md#bosun-17b-against-a-raw-qwen-base-2026-10-10).
  That vLLM comparison used 1.7B; it does not establish 0.6B benchmark parity.
- Decider 4B, JPT 4B/9B, Eikos 4B/27B: SDK importers can export a
  self-contained checkpoint whose text decision format has a vLLM adapter.
  The adapters have fixture tests (including an opt-in tiny-GPU Decider
  test); published-weight vLLM numerical parity is not established.
  Decider, JPT, and Eikos need vLLM 0.30+ / Transformers 5.17+. Their released image
  paths are outside these text-decision adapters.
- SDK-trained token-readout models: a completed standalone export,
  including a trained Bosun-style conversion, can use the token-logprob
  vLLM path. A newly converted but untrained LM does not make meaningful
  decisions merely because the backend starts.
- OpenJev / OpenJev-Flash: the SDK calls the publisher's System One
  HTTP helper, which uses vLLM behind that server. This is not an
  `VLLMEngine` checkpoint importer; set the publisher helper's `base_url`
  when loading the native model.

No decision-aware SDK vLLM adapter is provided for Kev-9B, Laya
(either variant), Clef/Clef-Flash, GLiNER2.5-Decide, Perplexity Decider,
the Nox/Lux 1.0 and 2.0 releases, or Liquid d1-3B. In particular,
`vllm-sr` in Nox/Lux repository names refers to the publisher's
ecosystem, not an SDK `VLLMEngine` capability. Use their supported
native/PyTorch inference paths. Converted pointer and letter-code
custom heads likewise need PyTorch; serving the underlying LM via vLLM
would discard those trained readouts.

## Export and serve

`trainer.export(...)` produces Hugging Face config, generation config,
tokenizer files, safetensors weights, `decision_config.json`, and
`training_metadata.json`. A QLoRA export reloads its base at full precision
before merging; the base must remain accessible. A standalone checkpoint
can be loaded by `vllm serve ./Futurix-Decision-1.7B`, but that command
alone exposes only vLLM's OpenAI API, not `/v1/decision`.

## Minimal export-to-API flow

Export the released Bosun checkpoint:

```bash
python -m pip install "futurix-decision[bosun]"
futurix-decision export \
  --model Hanno-Labs/bosun-v3.1-0.6b --model-format bosun \
  --revision 5e78a37c8d86860ebac9ef2d5327d6299498498e \
  --output ./bosun-vllm
```

Serve from the exported directory with the vLLM extra. The extras require
vLLM 0.30 or later:

```bash
python -m pip install "futurix-decision[vllm,serve]"
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
Decider `score` may require one prefill per level.

`--backend vllm` refuses a format that `VLLMEngine` cannot load as given.
Bosun, Decider, JPT, and Eikos ask you to run `futurix-decision export` first,
and formats with no vLLM path say so. The server micro-batches concurrent
requests for the vLLM backend too, with one `generate` call per batch; see
[serving](serving.md).

The SDK adapter can abstain using a `confidence_threshold`, but it
currently scores candidate tokens only, not a separately trained
null/abstain slot; use PyTorch for any model relying on a learned null
readout. Match the export's tokenizer, compatible vLLM version, GPU
capacity, and maximum context length to the model you choose. See the
[support matrix in the README](https://github.com/rudransh2004/Decision-sdk#supported-decision-models) for the
per-release status.

## How the vLLM adapter works

The vLLM adapter masks to `allowed_token_ids` and requests `logprobs` for every allowed candidate
after one restricted one-token step. Current GPU workers do not honor
`logprob_token_ids`, so the SDK uses processed top-k logprobs instead.
It ignores the sampled token and performs no freeform text generation;
vLLM's public API does not expose a zero-token logits-only pass. Temperature
is applied to the restricted decision scores. Batch requests run together.
Decider `score` uses a separate restricted prefill per score level.

The `vllm` extras require vLLM 0.30 or later. The adapter was first tested
against vLLM 0.20 on a tiny checkpoint. The GPU smoke test is opt-in
(`FUTURIX_TEST_VLLM=1`), and CI only checks that the extras' pins resolve,
because vLLM does not build on CPU runners.
