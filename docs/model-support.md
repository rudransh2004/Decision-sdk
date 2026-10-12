# Model support

The README lists every released decision model with its inference, training,
and vLLM status. This page has the details: how to load each family, what it
needs installed, and what it does not support. Loading a language model
backbone alone does not reproduce a publisher's decision model. `load_model`
keeps each release's own readout. Use `convert=True` only for a plain,
compatible causal LM that you plan to train (see [conversion](conversion.md)).

A "train" check in the matrix means an SDK path for training on labeled
decision data. It does not mean pretraining a publisher architecture from
scratch. Some paths need publisher packages, pinned remote code, a lot of GPU
memory, or hard labels.

## Install the extra for your family

Every extra includes `torch`. All extras that use transformers need
`transformers>=5.17,<6`, so the `bosun`, `decider`, `jpt`, and `eikos` extras
install in one environment.

| Family | Extra |
|:--|:--|
| Bosun | `bosun` |
| Decider | `decider` |
| JPT | `jpt` |
| Eikos | `eikos` |
| Laya | `laya` |
| Kev | `kev-native` (plus the publisher's `kev` package) |
| Clef | `clef` |
| GLiNER2.5-Decide | `gliner` (inference), `gliner-train` (training) |
| Perplexity Decider | `pplx-decider` (plus the publisher's `autojev` source) |
| Nox, Lux | `nox-lux` |
| Liquid d1 | `liquid-d1` |
| OpenJev | none; the HTTP runtime needs only the base install |

The vLLM extras are `vllm`, `decider-vllm`, `jpt-vllm`, and `eikos-vllm`.

## Native decision releases

- Bosun v3.1: `Hanno-Labs/bosun-v3.1-0.6b` and `-1.7b` have inference and SDK
  decision SFT, LoRA, and RLCD continuation. They use their original trained
  decision-token rows and stable-slot prompt.
- Laya: `convaiinnovations/laya` and `laya-multilingual` have inference and
  native fine-tuning through the SDK.
- Kev: the verified `jaredpalmer/kev-9b` path keeps its native pointer head
  and supports inference and publisher-backed LoRA and head fine-tuning.
- Clef and Clef-Flash: `Cloudflare/clef` and `Cloudflare/clef-flash` support
  native inference and head-only or full fine-tuning. The loader requires
  reviewed remote code and an immutable 40-character revision.
- GLiNER2.5-Decide: `fastino/GLiNER2.5-Decide` supports its native encoder
  decision path and hard-label training. Soft labels need an explicit
  `target_policy="argmax"` conversion.
- Perplexity Decider: `perplexity-ai/pplx-decider-v1.1-27b` supports native
  inference and readout-only or full fine-tuning using the publisher's
  installed runtime and a pinned revision. It is large; check disk and
  accelerator memory before loading.
- Decider, JPT, Eikos: `Mapika/decider-4b`, `kirp/jpt-4b` and `-9b`, and
  `caiovicentino1/Eikos-4B` and `-27B` have model-specific text readouts and
  SDK training and export paths.
- Nox and Lux: `vllm-sr/Decision-{1.0,2.0}-Nox-4B` and
  `vllm-sr/Decision-{1.0,2.0}-Lux-9B` support native SDK inference. The
  released wrappers do not permit the SDK's training path.
- Liquid d1: `LiquidAI/d1-3B` supports native text and image inference.
  Training and export have not been verified. Check Liquid's model license.
- OpenJev: supported inference goes through the publisher's System One helper
  in front of vLLM, not a local weights-directory importer. Native
  fine-tuning is not implemented.

This list does not imply that every checkpoint runs under vLLM. Bosun,
Decider, JPT, and Eikos need an SDK export first. OpenJev uses its publisher's
vLLM-backed helper. The other native decision heads are not served by
`VLLMEngine`. See [inference backends](inference-backends.md) for the
per-model backend guide.

## The noul option order

A `noul` question has the options `["false", "true"]` in the SDK's canonical
data, and a result's `probabilities` always follow that order. Families that
present the options as lettered choices list them in the order their
checkpoint saw during training:

| Family | Order in the prompt | Why |
|:--|:--|:--|
| Decider | false first | Matches the v2.1 training prompt (`NOUL_ORDER` in `models/decider.py`). |
| JPT | true first | llm2jev trained JPT on "A. Yes / B. No". |
| Eikos | true first | The SemIf checkpoint was trained with `true` as option A. |

Decider, JPT, and Eikos reject `noul` inputs whose answer IDs are not the
literal `true` and `false`. They do not guess the meaning of opaque benchmark
IDs. Training permutes the presented order of choice and noul options per row
under the trainer's `assignment_seed`. The result type does not depend on
the order: `decision` is `True` or `False`.

## Loading from Hugging Face or disk

Pass a supported Hub ID or alias to `load_model(...)`: the publisher loader
downloads missing files into the normal Hugging Face cache and reuses cached
files on later calls. Pass an existing directory (a `Path`, absolute path,
or `./relative/path`) to load its checkpoint from disk. A publisher copy
can be recognized by its native config/head even if renamed; for an
ambiguous local variant, specify `family=...` explicitly.

```python
from pathlib import Path
from futurix_decision import load_model

# Downloads on first use; remote publisher code needs a reviewed pinned commit.
release = load_model(
    "clef-flash", revision="<40-character-commit>", trust_remote_code=True, device="cuda"
)
# No Hub download for the local checkpoint itself (referenced base models may
# still need to be cached or downloaded, e.g. Bosun and Kev).
on_disk = load_model(
    Path("/models/my-clef-variant"), family="clef", trust_remote_code=True, device="cuda"
)
```

Before it loads a local checkpoint or a Clef, Perplexity, Nox, Lux, or d1 Hub
snapshot, the SDK checks the required configs, trained heads and readouts, and
every weight shard listed in a safetensors index. Each family's
`FamilyAdapter.requires` does this check once per `load_model` call. Missing
components raise `CheckpointError` that names the family, source, and files. A
missing local directory raises `FileNotFoundError`, and the SDK never reads a
missing local path as a Hub ID. Kev's Hub pointer head is checked before its
base is loaded. An incomplete directory cannot silently become an untrained
generic LM. SDK-exported Futurix checkpoints load from disk or from a Hub repo
that contains `decision_config.json`. OpenJev is the exception: it needs its
published HTTP and vLLM helper, not a weights directory.

`load_model` also accepts Bosun's typed-decision releases: `bosun-0.6b`
(`Hanno-Labs/bosun-v3.1-0.6b`) and `bosun-1.7b` (`Hanno-Labs/bosun-v3.1-1.7b`).
It pins each repository to its verified commit and keeps Bosun's 256 trained
decision-token rows and prompt contract. Bosun-XS and Bosun-4B are binary
rerankers, not these typed-decision releases.

A native model exposes `choice`, `noul`, `score`, `decide`, `batch_decide`, and
`system_one`. `model.runtime` is the publisher's runtime for
publisher-specific hooks, and `model.capabilities` reports whether a native
training and export recipe exists.

## Notes per family

- Kev: `jaredpalmer/kev-9b` local pointer-head load, inference and native
  LoRA/pointer-head fine-tuning via the publisher's `kev` package, saving
  `head.pt` and adapter together. Other Kev sizes are not verified.
- Laya and Laya Multilingual: `convaiinnovations/laya` and
  `convaiinnovations/laya-multilingual` load with `laya`; its native fine-tuner
  accepts our grouped soft targets, fits held-out temperatures and saves a
  reloadable encoder/head checkpoint.
- Clef: `Cloudflare/clef` and `Cloudflare/clef-flash` load their
  `joint_head.safetensors` with the publisher's pinned `joint_schema_model.py`.
  Native differentiable head-only (default) or full fine-tuning exports the
  backbone, processor, joint head, config and code together. Requires
  `trust_remote_code=True` and a 40-character pinned Hub revision.
  `clef-omni` is not verified.
- GLiNER: `fastino/GLiNER2.5-Decide` loads a schema-aware encoder and
  trains/exports through `gliner2[train]`. Its native loss is hard-label
  classification: soft labels fail unless you explicitly set
  `target_policy="argmax"` (which discards their distribution).
- Perplexity decider: `perplexity-ai/pplx-decider-v1.1-27b` loads through
  the publisher's installed `autojev` source. Full-weight or readout-only
  fine-tuning keeps the checkpoint's noncausal attention and 255-code readout
  and saves via its native `save()`. Requires a pinned revision and adequate
  GPU memory; there is no generic vLLM decision exporter.
- OpenJev: `openjev/openjev` and `openjev/OpenJev-Flash-9B` infer through
  their publisher's System One helper in front of vLLM, not a bare vLLM
  chat endpoint. The SDK checks `/v1/version` for the requested model and
  release-specific calibration. Native fine-tuning is not implemented.
- Nox and Lux: aliases select `vllm-sr/Decision-2.0-Nox-4B` and
  `vllm-sr/Decision-2.0-Lux-9B`; explicit `Decision-1.0-*` IDs also load.
  These use the publisher's candidate head through pinned Transformers
  remote code. Both 1.0 and 2.0 Nox/Lux passed released-weight `choice`,
  `noul`, `score`, and grouped `system_one` checks. The released wrappers
  prohibit `.train(True)`, so this SDK exposes inference only until a
  safe native train/export path exists. Its local SDK path does not start
  a vLLM Semantic Router server. Transformers currently warns about the
  tokenizer regex in the 1.0 releases; these smoke checks do not establish
  benchmark accuracy or tokenizer parity with the publisher's runtime.
- Plain text causal LMs: `load_model("<Hub ID or local directory>",
  convert=True, strategy="pointer")` creates a candidate-aware pointer head;
  `strategy="bosun"` instead initializes 256 decision-token rows with a
  Bosun-style stable-slot prompt. Both work over *compatible* Hugging Face
  text causal LMs independent of parameter count in the SDK API, but need
  decision-data training before their outputs mean anything. Head-only SFT
  and direct 4-bit QLoRA are available for pointer heads; full SFT and LoRA
  train either strategy. Hardware requirements scale with the base.
- Qwen3.5-9B: `Qwen/Qwen3.5-9B` is not already a decision model.
  Its existing multimodal-specific `load_model("qwen3.5-9b", convert=True)`
  path remains available. `strategy="bosun"` selects its token-row contract,
  although live training of this 9B variant has not been verified.
- Liquid d1: `LiquidAI/d1-3B` preserves its native multimodal verbalizer
  via pinned remote code; `d1-omni-600M` and a safe native training/export
  path have not been verified. Use `system_one(..., images=[...])` for the
  3B publisher's image path. Released-weight text, image and local reload
  checks passed; training and publisher benchmark parity were not tested.

```python
from pathlib import Path
from futurix_decision import load_model

nox = load_model(
    "vllm-sr/Decision-2.0-Nox-4B",
    revision="ae269ecf34a0b3a89026bc1afd9c995279212fad",
    trust_remote_code=True,
    device="cuda",
)
print(
    nox.choice(
        state="Customer requests a refund", question="Which team?", choices=["billing", "technical"]
    ).probabilities
)
del nox

# Run separately if GPU memory is limited; any release also loads from disk.
lux = load_model(Path("/models/Decision-2.0-Lux-9B"), trust_remote_code=True, device="cuda")
```

To check a published checkpoint without keeping its weights, run
`python examples/check_released_inference.py nox-1` (or `nox-2`, `lux-1`,
`lux-2`, `liquid-d1`) in its matching Transformers 5 environment. Each run uses
a separate temporary Hub cache and removes it after remote and local
inference. Liquid d1 also checks an image request.

Fine-tune a native release with `finetune`. OpenJev needs its helper running
with the model-card vLLM flags and calibration. `load_model` returns a
`DecisionBackend`; narrow it with `isinstance(model, NativeDecisionModel)`
before you call `finetune` if you type-check.

```python
from futurix_decision import NativeSFTConfig, load_model

clef = load_model(
    "clef-flash",
    revision="<40-character-Hub-commit>",
    trust_remote_code=True,
    device="cuda",
)
clef.finetune(
    "decisions.jsonl", "./my-clef", config=NativeSFTConfig(epochs=1, train_backbone=False)
)

openjev = load_model("openjev-flash", base_url="http://127.0.0.1:3000")
print(openjev.noul(state="Invoice paid", question="Was it paid?").probabilities)
```

The CLI equivalents are `futurix-decision evaluate --model laya --model-format
native --dataset decisions.jsonl` and `futurix-decision train --model laya
--model-format native --dataset decisions.jsonl --output ./my-laya`.

Publisher runtimes have separate dependency and version requirements: install
the matching `laya`, `gliner-train`, `clef`, `nox-lux`, or `liquid-d1` extra.
The HTTP runtime needs only the base install. Install Kev's `kev` package and
the Perplexity `source/src/autojev` package from their publishers in a
compatible environment. The SDK never downloads and executes that package
code for you. OpenJev weights are CC BY-NC 4.0, and d1 has Liquid's LFM Open
License, which has commercial restrictions. Check each checkpoint's license
before you use it. The smoke runs in [benchmarks](benchmarks.md) verify
specific paths. They are not benchmark parity.

## Decider, JPT, and Eikos

Use an explicit native loader. `DecisionModel.from_pretrained` is for plain
LLMs and Futurix exports; it now rejects recognized third-party decision
checkpoints so their trained prompts and readouts cannot be silently replaced.

Bosun v3.1 uses `DecisionModel.from_bosun(...)` (see below). The first new
native importer is `Mapika/decider-4b` v2.1 for text/JSON inputs:

```python
from futurix_decision import DecisionTrainer, SFTConfig, load_model

model = load_model("Mapika/decider-4b")  # DecisionModel.from_decider(...) does the same
result = model.choice(state="Review this ticket", choices=["approve", "reject"])
trainer = DecisionTrainer(
    model=model,
    dataset="decisions.jsonl",
    sft_config=SFTConfig(method="lora"),
)
trainer.train(stages=["decision_sft"])
trainer.export("./decider-futurix")
restored = load_model("./decider-futurix")
```

The CLI trains and evaluates the same way:

```bash
python -m pip install "futurix-decision[decider]"
futurix-decision evaluate --model Mapika/decider-4b --model-format decider \
  --dataset decisions.jsonl
futurix-decision train --model Mapika/decider-4b --model-format decider \
  --dataset decisions.jsonl --calibration-dataset heldout.jsonl \
  --stages decision_sft,calibration,rlcd --finetune-method lora \
  --output ./decider-futurix
```

This importer keeps the upstream existing A–Z/AA–... vocabulary rows,
plain prompt, v2.1 type-specific temperatures and independent yes/no rows
for each Score level. It does not create new decision tokens or a new head.
There are at most 255 Choice options and ten Score levels. SDK `score()`
returns the most likely level (and its full probability map), whereas the
publisher's Jev-shaped API returns the distribution's expected level.
The native checkpoint needs Transformers 5.17 or later, as does every other
extra that includes transformers. A 4B training run needs suitable GPU memory.
Unit tests use tiny local models, not the published 4B weights.

After you export a complete Futurix checkpoint, `VLLMEngine("./decider-futurix")`
or `futurix-decision evaluate --backend vllm --model ./decider-futurix` runs
one restricted-logprob prefill per row (or per isolated Score level). Install
`decider-vllm` and use a vLLM build that supports Qwen3.5. A bare
`vllm serve Mapika/decider-4b` does not expose the SDK's typed decisions.
Numerical parity between the native model and vLLM has not been verified on the
published 4B checkpoint.

Text-only importers also exist for `kirp/jpt-4b`, `kirp/jpt-9b`, and
`caiovicentino1/Eikos-4B` and `Eikos-27B`:

```bash
python -m pip install "futurix-decision[jpt]"   # or [eikos]
futurix-decision train --model kirp/jpt-4b --model-format jpt \
  --dataset decisions.jsonl --output ./jpt-futurix --finetune-method lora
futurix-decision evaluate --model caiovicentino1/Eikos-4B \
  --model-format eikos --dataset decisions.jsonl
```

`DecisionModel.from_jpt(...)` uses llm2jev's contextual label-token check,
chat template with thinking disabled and JPT's held-out temperature.
`DecisionModel.from_eikos(...)` uses SemIf's JSON prompt, its published
one-token label table and checks the release's T=1 calibration. Both
preserve their trained LM rows, use the SDK's differentiable decision
losses for *new* SFT/LoRA/RLCD fine-tuning, and export reloadable models.
JPT supports up to 255 options; the Eikos one-pass limits are 100 (4B) and
160 (27B). Above those Eikos limits the publisher uses a tournament, which
this SDK does not implement. Both original JPT weights are
CC BY-NC 4.0. JPT requires Transformers 5.17+; Eikos requires 5.12+.
Their released vision/image paths are not supported by these canonical
text/JSON `DecisionInput` adapters. The JPT/Eikos vLLM paths operate on
Futurix-exported checkpoints using restricted first-token logprobs; use
`jpt-vllm` or `eikos-vllm` and compatible vLLM releases. No full
publisher-weight GPU parity or published-benchmark tests have been run.
All three native letter importers reject `noul` inputs whose answer IDs
cannot be unambiguously identified as the literal `true` and `false`;
they must not guess the meaning of opaque benchmark IDs. If exporting an
unmerged PEFT adapter for continued training, reload it with
`DecisionModel.from_pretrained(path, trainable_adapter=True)`.

The generic `DecisionModel.from_pretrained` loader rejects these publisher
checkpoints. Use `load_model`, which keeps each publisher's own decision
readout. Loading an LM backbone in vLLM is not the same as serving its trained
decision readout. JPT-35B-A3B is catalogued so that loading fails closed, but
it has no verified importer.

## Bosun

Bosun 3.1 is an existing decision checkpoint, not a plain Qwen base.
Its custom loader reconstructs the pinned Qwen base, the original LoRA,
tokenizer snapshot, and the separately trained input/output decision rows.
Importing it requires a trusted, pinned commit and the `bosun` extra
(`pip install "futurix-decision[datasets,train,bosun]"`):

```python
from futurix_decision import DecisionModel

bosun = DecisionModel.from_bosun(
    "Hanno-Labs/bosun-v3.1-1.7b",
    revision="1d8dc82a20e4a32ed60927a47272d6efff48eed2",
)
```

The SDK preserves Bosun's ChatML JSON prompt, row-ID-seeded presentation
permutation, 255 candidate slots plus reserved null slot, and trained
decision-token rows. On the pinned checkpoint its readout has been checked
against Bosun's native `predict`. Bosun continuation merges its original
inference-only adapter into the restored base before applying a trainable
LoRA to both the model and decision-token input/output surfaces. Exported
checkpoints carry the same prompt contract for Transformers and vLLM.

For the Bosun training runs and their results, see [benchmarks](benchmarks.md).

## Converting a plain LM

<!-- test -->
```python
from futurix_decision import load_model

pointer = load_model("your-org/text-causal-lm", convert=True, strategy="pointer")
bosun_style = load_model("your-org/text-causal-lm", convert=True, strategy="bosun")
```

These are two new, untrained readouts, not copies of a publisher's decision
model. The pointer readout scores candidates directly and does not add
decision vocabulary rows. Bosun-style conversion creates 256 token rows with a
reserved null slot. The SDK sets no limit on parameter count, but storage,
memory, model architecture, and tokenizer compatibility still limit you.
Encoder-only, seq2seq, multimodal, API-only, and unsupported custom-code
checkpoints are not general conversion targets. See [conversion](conversion.md)
for strategy requirements and what has been tested.

## Add a family

A third-party family is a `FamilyAdapter` registered with `register_family`.
[Contributing](https://github.com/rudransh2004/Decision-sdk/blob/main/CONTRIBUTING.md) describes the
interface.

## Licenses, downloads, and trust

The SDK's MIT license covers this repository's code, not upstream weights,
training corpora, or publisher dependencies. Some checkpoints restrict
commercial use. The SDK downloads missing model files through the normal
Hugging Face cache and accepts local directories as well. When a model
requires `trust_remote_code=True`, review the upstream implementation, pin an
immutable revision, and use an isolated environment.

The project's tests use tiny local fixtures. Published-checkpoint smoke tests
and reported comparisons are in [benchmarks](benchmarks.md). A successful load
or toy fine-tuning run is not a claim of benchmark parity.
