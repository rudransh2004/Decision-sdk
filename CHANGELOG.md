# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- The `native-http` extra. `httpx` is now a base dependency.
