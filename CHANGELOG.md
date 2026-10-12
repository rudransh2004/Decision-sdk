# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `DecisionModelConfig` (exported from the top-level package): a dataclass that holds a
  `DecisionModel`'s settings, validates them, and is what `decision_config.json` stores.
  Checkpoints saved by earlier versions load unchanged.
- `ReadoutStrategy` (`futurix_decision.models`): a `DecisionModel`'s readout lives in
  `model.readout`, one of `TokenReadout` (Futurix and Bosun decision tokens),
  `PointerReadout` (candidate heads), `LetterCodeReadout`, or, for Decider, JPT, and
  Eikos, `LabelReadout`.
- `FamilyAdapter` and `register_family`, exported from the top-level package (no torch
  needed). An adapter names a family, lists the Hub IDs it loads, names its pip extra,
  and implements `load`; `requires`, `finetune`, and `training_kwargs` are optional.
  `load_model(checkpoint, family=...)` dispatches through the registry, so a registered
  third-party family loads like a built-in one.
- Each `FamilyAdapter` declares its own releases and capabilities: `checkpoints`,
  `unsupported_checkpoints`, `aliases`, `markers` (with an overridable `detect(directory)`),
  `trainable`, and `vllm` (`"direct"`, `"export"`, `"server"`, or `None`). Hub ID lookup,
  short names, and local-copy detection all read these, so a registered third-party
  family is found by alias and recognized in a local directory without `family=`.
  `register_family` refuses a Hub ID or alias another family already claims.
- `DecisionBackend`, a runtime-checkable protocol exported from the top-level package
  (no torch needed): `decide`, `batch_decide`, `choice`, `noul`, `score`, and `close`.
  `DecisionModel`, `NativeDecisionModel`, `VLLMEngine`, `DecisionClient`, and
  `Calibrated` implement it. `DecisionModel.close()` and `VLLMEngine.close()` do nothing.
- `Calibrated(backend, temperature)` wraps any backend and rescales its probabilities to
  `softmax(log p / T)`, then recomputes the decision, confidence, and abstention.
- `DecisionResult.kind` and `DecisionResult.from_probabilities(example, probabilities,
  latency_ms=..., confidence_threshold=...)`.
- `DecisionClient` and `AsyncDecisionClient` (`futurix_decision.client`, no torch needed)
  for the `futurix-decision serve` API: `decide`, `batch_decide`, `choice`, `noul`,
  `score`, `health`, and `close`, usable as (async) context managers. Settings:
  `base_url`, `api_key`, `timeout` (30 s), and `max_retries` (3). Connect errors and
  HTTP 429/502/503/504 are retried with exponential backoff and jitter, honoring
  `Retry-After` up to 60 s; other failures raise `BackendError`. The server has no
  batch endpoint, so `batch_decide` sends one request per example: sequentially in
  the sync client, at most `max_concurrency` (8) at a time in the async one.
- A `torch` extra (`torch>=2.4`, `transformers>=5.17,<6`). Every model-family extra,
  `train`, `qlora`, `serve`, the vLLM extras, and `dev` include it.
- `futurix_decision.errors`, exported from the top-level package: `FuturixError` is the
  base class of `ConfigError` (also a `ValueError`), `CheckpointError`,
  `MissingDependencyError` (also an `ImportError`), and `BackendError` (with
  `status_code` and `body`), whose subclass is `InvalidOutputError`.
- The package ships a `py.typed` marker, so type checkers read its inline annotations.
- The `dev` extra installs `fastapi`, `httpx`, `ruff`, `mypy`, and `types-PyYAML`.
  The serving tests now run under `pip install -e ".[dev]"` instead of skipping.

### Changed

- **Breaking:** `DecisionResult.decision` has one type per kind: `bool` for noul, `int`
  for score, `str` for choice, and `None` when abstained. `preserve_order` rows no longer
  return the option ID string: a true-first noul row returns `True`/`False`. A
  `preserve_order` row whose candidate IDs are not `true`/`false` or integer levels (such
  as DecisionBench binary rows with IDs `c000`/`c001`) returns `decision=None` with
  `abstained=False`. Migration: compare noul decisions with `True`/`False`, not
  `"true"`/`"false"`; read opaque-ID winners from `probabilities`.
- **Breaking:** `DecisionResult` is a frozen pydantic model and is also the HTTP response
  model; `DecisionResponse` is gone. It gains a required `kind` field, which the server
  now sends and `DecisionClient` checks against the request. Migration: replace
  `result.to_dict()` with `result.model_dump()`, construct results with keyword arguments
  including `kind`, and upgrade the client and server together.
- **Breaking:** `latency_ms` is always wall time per row. `DecisionModel` and
  `VLLMEngine` batches used to stamp the whole batch time on every row. Migration:
  multiply by the batch size for the batch total.
- **Breaking:** `fit_temperature(backend, heldout)` fits on the probabilities any backend
  returns and returns the temperature, relative to the backend's current output. It no
  longer sets `DecisionModel.temperature`. Migration: wrap with
  `Calibrated(backend, fit_temperature(backend, heldout))`, or persist into a checkpoint
  with `model.config.temperature *= fit_temperature(model, heldout)`. `CalibrationTrainer` and
  `DecisionTrainer` still store the fitted temperature on the model.
- **Breaking:** `load_model` has an explicit signature: `family`, `convert`,
  `strategy: Literal["pointer", "bosun", "letter_code"] | None`, `head_type`,
  `train_backbone`, `revision`, `device`, and `**family_kwargs`. It is typed to return
  `DecisionBackend` (`DecisionModel` when `convert=True`). `head_type` and
  `train_backbone` raise `ConfigError` for publisher, Bosun, Decider, JPT, and Eikos
  releases instead of reaching their loaders. `device` now also places a `DecisionModel`
  (publisher runtimes still default to `"cpu"`); passing it with `device_map` raises
  `ConfigError`. Migration: drop `head_type` and `train_backbone` where a release
  rejects them, and narrow the result with `isinstance` before calling
  family-specific methods such as `NativeDecisionModel.finetune`.
- **Breaking:** `DecisionModel(backbone, tokenizer, config=None, *, decision_head=None,
  **settings)` takes a `DecisionModelConfig`; keyword `settings` replace its fields, so
  `DecisionModel(lm, tokenizer, decision_tokens=16)` still works. The settings moved off
  the model onto `model.config`: `model.temperature`, `model.head_type`,
  `model.prompt_format`, `model.max_sequence_length`, `model.source_checkpoint`, and the
  rest are gone. The private `_token_ids`, `_marker_id`, `_prompt_and_slots`, and
  `_input_device` are gone too. `DecisionBatch` moved to
  `futurix_decision.models.readouts` and is still exported from
  `futurix_decision.models`. Migration: read and write `model.config.temperature` and
  friends; use `model.readout.token_ids` and `model.readout.prompt(example, seed)`.
- `DecisionModel.from_pretrained` finds the class for a saved `prompt_format` in the
  family registry, so a registered `LabelReadoutModel` family loads through it like
  Decider, JPT, and Eikos. A setting that conflicts with the checkpoint raises
  `cannot override the saved checkpoint's <field>`, naming the field.
- `evaluate(..., max_batch_tokens=...)` on Decider, JPT, and Eikos measures each row's
  own tokenized prompt. It used to measure the generic Futurix prompt instead.
- **Breaking:** each publisher runtime (Kev, Laya, GLiNER, Clef, PPLX Decider, OpenJev,
  Nox, Lux, Liquid d1) is its own `PublisherAdapter` subclass in
  `futurix_decision.models.families`, which loads, checks, runs, and fine-tunes it;
  `NativeDecisionModel` no longer branches on the family and gains an `adapter`
  attribute. `models.sources.require_checkpoint_files` is gone: each adapter's
  `requires(path)` checks its own files. Private helpers moved and became public:
  `native_question` and `option_keys` in `futurix_decision.datasets.systemone`, and
  `require_transformers` in `futurix_decision._lazy`. `load_model` lives in
  `futurix_decision.models.loader`. Migration: call `get_family(name).requires(path)`
  instead of `require_checkpoint_files(path, name)`; import `load_model` from
  `futurix_decision` or `futurix_decision.models`.
- A local publisher or Bosun checkpoint has its files checked once per `load_model`
  call instead of two or three times.
- Bosun loading lives in `BosunAdapter.load` (`futurix_decision.models.families.bosun`),
  which also holds `BOSUN_RELEASES` (moved from `futurix_decision.models.decision_model`).
  `DecisionModel.from_bosun` checks a local copy's files and calls it.
- `DecisionModel.to(device)` moves the backbone and any decision head and returns the
  model; `device=None` leaves them in place. It replaces
  `futurix_decision.models.families.futurix.on_device`.
- `load_model(..., convert=True)` on any known decision-model family raises one
  `ConfigError` that names the family. Bosun, Decider, JPT, Eikos, and Futurix
  checkpoints used to raise per-family messages.
- The CLI loads every `--model-format` through the family registry. `--model-format`
  accepts any registered family, and `serve` now accepts `bosun`. Other changes:
  - `train` raises `ConfigError` for an unknown YAML `model.format`; it used to train it
    as `standard`.
  - `train --decision-tokens` with `decider`, `jpt`, or `eikos` is checked like the YAML
    value (255 for Decider and JPT, not allowed for Eikos) instead of being ignored.
  - `train` with Bosun and QLoRA raises `ConfigError` before loading the model, instead
    of in the trainer.
  - `export --revision` applies to every format; `standard` used to ignore it.
  - `serve --device` also moves a custom decision head, and `--device cuda` without
    CUDA raises `ConfigError` for `native` too.
- `evaluate` and `fit_temperature` take any `DecisionBackend`. `max_batch_tokens` in
  `evaluate` needs a `DecisionModel` or `VLLMEngine` (optionally inside `Calibrated`) and
  raises `ConfigError` for other backends.
- OpenJev's published server temperatures live in
  `futurix_decision.models.families.openjev.CALIBRATIONS`. The SDK still only verifies
  that the helper reports them; it never applies them a second time.
- Loading a decision release through the generic LM loader (`DecisionModel.from_pretrained`,
  `convert=True`) now says to use `load_model(..., family=...)` instead of naming a
  `DecisionModel.from_*` constructor.
- `--backend vllm` in `evaluate` and `serve` refuses every format whose checkpoints
  `VLLMEngine` cannot load as given: Bosun, Decider, JPT, and Eikos ask for
  `futurix-decision export` first, and formats with no vLLM path say so. Only `evaluate`
  checked, and only for Bosun; the others failed inside `VLLMEngine`.
- The native HTTP runtime (`load_model(..., base_url=...)`) uses the client's transport:
  new `timeout` (default 60 s, as before) and `max_retries` (default 3) arguments, the
  same retry policy, and `BackendError` instead of `httpx.HTTPStatusError` for failed
  requests. `NativeDecisionModel` gains `close()` and context-manager support, which
  close only an HTTP client it created.
- **Breaking:** the SDK raises its own exception types. Bad arguments and configuration
  raise `ConfigError`, which still subclasses `ValueError`. These changed type:
  - Incomplete, unrecognizable, or wrong-family checkpoints and invalid checkpoint
    metadata raise `CheckpointError` instead of `ValueError` or `FileNotFoundError`.
    A missing local checkpoint directory still raises `FileNotFoundError`.
  - Publisher runtimes and vLLM returning malformed probabilities or answers raise
    `InvalidOutputError` instead of `ValueError`. The OpenJev server check raises
    `BackendError`.
  - Missing optional packages raise `MissingDependencyError`, still an `ImportError`.
    `load_model` on a transformers checkpoint without torch raises it instead of
    `ModuleNotFoundError`.
- **Breaking:** the base install no longer depends on torch or transformers. It needs
  only `pydantic`, `httpx`, `pyyaml`, and `huggingface-hub`. Install
  `futurix-decision[torch]` (or a model-family extra) to run local models.
  `import futurix_decision` works without torch. `DecisionModel`, trainers, `evaluate`,
  `fit_temperature`, and the rewards load on first access and raise
  `MissingDependencyError` when torch is missing.
- **Breaking:** `compute_metrics` takes sequences of floats and computes with `math`.
  One-dimensional tensors still work; call `.tolist()` on anything else.
- **Breaking:** `prepare_decider_rows` returns `LabelRow`s
  (`futurix_decision.models.label_readout`), and `DeciderRow.letter_ids` is now
  `LabelRow.label_ids`. Decider, JPT, and Eikos models subclass `LabelReadoutModel`.
- Eikos fine-tuning permutes choice and noul options per row under the trainer's
  `assignment_seed`, as Decider does. It used to ignore the seed and always train on the
  presented order. Score levels stay in order, because Eikos names them by position.
  Inference is unchanged.
- The package version comes from `futurix_decision.__version__` alone.
  The `futurix-decision serve` API reports that version instead of a hard-coded `0.1.0`.
- Every install that includes transformers requires `transformers>=5.17,<6` (the base
  previously accepted 4.57).
  The `bosun` extra no longer pins `transformers<5`, so Bosun installs alongside
  `decider`, `jpt`, and `eikos`. The `eikos`, `clef`, and `liquid-d1` extras raise
  their minimum from 5.12, 5.10.2, and 5.14.
- Every vLLM extra (`vllm`, `decider-vllm`, `jpt-vllm`, `eikos-vllm`) requires `vllm>=0.30,<1`.
  The `vllm` extra previously accepted 0.20 and the Decider and JPT extras 0.29.
- `gliner` and `gliner-train` require `gliner2>=2.1,<3`. `clef` and `liquid-d1` require
  `torchvision>=0.28,<1`. Both were unpinned.

### Removed

- **Breaking:** the separate family tables. `futurix_decision.models.families.FAMILIES`,
  `ModelFamily`, `identify_family`, `server_calibration` (moved to
  `models.families.openjev`, now taking only the checkpoint), and
  `models.families.base.supported_checkpoints`; `models.families.publisher.PUBLISHER_CHECKPOINTS`
  and `PublisherAdapter.vllm_server`; `models.sources.ALIASES` and
  `models.sources.infer_local_family`. Migration: read the adapters through
  `futurix_decision.models.registry` (`registered_families`, `get_family`,
  `family_for_checkpoint`, `resolve_alias`, `infer_local_family`,
  `reject_native_checkpoint`). The Laya and GLiNER `encoder` row is gone; they are the
  `laya` and `gliner` families.
- **Breaking:** `LetterAdapter(name, model, markers=...)` and `NoxLuxAdapter(name)` no
  longer take constructor arguments. Use `DeciderAdapter`, `JPTAdapter`, `EikosAdapter`,
  `NoxAdapter`, and `LuxAdapter`.
- **Breaking:** `futurix_decision.models.registry.register_backend` and `create_backend`.
  Nothing in the SDK called them. Migration: implement a `FamilyAdapter` and call
  `register_family`.
- **Breaking:** the `pplx` strategy and head-type alias and the `pplx-v1` prompt-version
  alias, in `load_model`, `DecisionModel`, and `futurix-decision train
  --convert-strategy`. Migration: use `letter_code` and `letter-code-v1`. A checkpoint
  whose `decision_config.json` still names `pplx` raises `CheckpointError`; set
  `head_type` to `letter_code` and `head_prompt_version` to `letter-code-v1` in that file.
- The `native-http` extra. `httpx` is now a base dependency.

### Fixed

- A pointer head with `learn_abstain=True` no longer fails every forward with "decision
  head must return one logit per supplied option". `forward_decisions` returns the
  abstain logit after the widest row's options, sets `DecisionBatch.null_slot`, and
  `assigned_logits(row)` ends with it. Inference results still report probabilities
  over the options only.
- `PointerDecisionHead.forward` applies its `available` mask: unavailable option
  positions get `-inf`.
- Every `zip` in the SDK is strict (ruff B905 is on). A length mismatch between rows
  and results, or between probabilities and targets, raises `ValueError` instead of
  silently dropping the extra items.
