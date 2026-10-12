# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
  with `model.temperature *= fit_temperature(model, heldout)`. `CalibrationTrainer` and
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
- `evaluate` and `fit_temperature` take any `DecisionBackend`. `max_batch_tokens` in
  `evaluate` needs a `DecisionModel` or `VLLMEngine` (optionally inside `Calibrated`) and
  raises `ConfigError` for other backends.
- OpenJev's published server temperatures live in the family table
  (`ModelFamily.server_calibration`). The SDK still only verifies that the helper reports
  them; it never applies them a second time.
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

- **Breaking:** the `pplx` strategy and head-type alias and the `pplx-v1` prompt-version
  alias, in `load_model`, `DecisionModel`, and `futurix-decision train
  --convert-strategy`. Migration: use `letter_code` and `letter-code-v1`. A checkpoint
  whose `decision_config.json` still names `pplx` raises `CheckpointError`; set
  `head_type` to `letter_code` and `head_prompt_version` to `letter-code-v1` in that file.
- The `native-http` extra. `httpx` is now a base dependency.
