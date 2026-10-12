# Futurix Decision SDK 0.2 plan

Goal: turn the research toolkit into an SDK an outside developer can install, type-check against, and call over HTTP without torch.

## Decisions

| Question | Decision |
|---|---|
| API breaks | Allowed. Ship as 0.2.0 with migration notes in `CHANGELOG.md`. No deprecation shims. |
| transformers | One range, `>=5.17,<6`, for every extra. The Bosun `<5` pin goes. |
| Torch-free use | Base package needs only pydantic, httpx, pyyaml, and huggingface-hub. Local models need `pip install futurix-decision[torch]` or a family extra. |
| CI | GitHub Actions on CPU. No GPU job, no PyPI publishing in this plan. |
| Noul true/false order | Stays per family. Decider uses false-first; Eikos and JPT use true-first. Each matches its checkpoint's training prompt. Document it, don't unify it. |
| `decision` type | Fixed per kind: noul → `bool`, score → `int`, choice → `str`, or `None` when abstained. `preserve_order` stops changing the type. |

## Phases

Each phase is one PR, or a short stack where noted. Every PR ends with CI green.

### Phase 0: land current work

- Commit the `src/` layout move and the cleanup pass that are now in the working tree.
- Fix the 6 tolerance failures: use `places=5` or `torch.testing.assert_close` for float32 checkpoint round-trips in `test_custom_head.py`, `test_decider.py`, `test_jpt.py`, and `test_letter_code_conversion.py`.
- Done when `pytest` passes with 0 failures on macOS and Linux.

### Phase 1: tooling and CI

- Add `ruff` (lint and format) and `mypy` config to `pyproject.toml`. Start mypy non-strict, with `disallow_untyped_defs` on `inference/`, `datasets/schema.py`, and the new `client/` and `errors.py`. Tighten module by module later.
- Add `src/futurix_decision/py.typed`.
- Single-source the version: `[tool.hatch.version] path = "src/futurix_decision/__init__.py"`. Make `vllm/api.py` read `futurix_decision.__version__` instead of hard-coding `0.1.0`.
- Add `fastapi` and `httpx` to the `dev` extra, so the serving tests stop skipping silently.
- `.github/workflows/ci.yml` runs these jobs:
  - `lint`: `ruff check`, `ruff format --check`, `mypy`.
  - `test`: Python 3.10, 3.11, and 3.12 on ubuntu-latest, running `pytest`.
  - `extras`: one job per extra, each running `uv pip install .[extra]` and then importing the family module. It catches pin conflicts like the old Bosun one. vLLM extras only check resolution with `uv pip compile`, because vLLM won't build on CPU runners.
- Add `CHANGELOG.md` (Keep a Changelog format) and `CONTRIBUTING.md` covering setup, tests, the ruff/mypy commands, and how to add a model family (filled in after Phase 5).
- Done when CI is green on the PR and the `extras` job fails if `bosun` is put back on `transformers<5`.

### Phase 2: dependency cleanup

- Every extra moves to `transformers>=5.17,<6`. This affects `bosun`, `eikos`, `clef`, and `liquid-d1`.
- vLLM extras move to one lower bound: `vllm>=0.30,<1`.
- Pin the open-ended deps: `gliner2[local]>=X,<Y` and `torchvision>=X,<Y`, using the versions in the current lockfile.
- Rerun the Bosun tests on the unified range. Run `examples/check_released_inference.py` against `Hanno-Labs/bosun-v3.1-0.6b` once by hand and paste the probabilities into the PR.
- Done when `pip install .[bosun,decider,jpt,eikos]` resolves.

### Phase 3: torch-free core

- Base `dependencies`: `pydantic`, `httpx`, `pyyaml`, `huggingface-hub`. Add a new `torch` extra with `torch>=2.4` and `transformers>=5.17,<6`. Every family extra and `train` include the `torch` extra.
- Torch-free modules: `datasets/schema.py`, `datasets/loader.py` (the `datasets` lib stays optional), `inference/`, `errors.py`, `client/`, `utils/config.py`, and `evaluation/metrics.py`. That last one is rewritten on plain floats; the 69-line file only needs `math`.
- `futurix_decision/__init__.py` uses a module-level `__getattr__` to import `DecisionModel`, trainers, `fit_temperature`, and the rest on first access. Accessing them without torch raises `MissingDependencyError("install futurix-decision[torch]")`.
- CI adds a `core` job: install with no extras, `import futurix_decision`, use `DecisionClient` against a stub server, and assert `"torch" not in sys.modules`.
- Done when the `core` job passes and the base wheel's installed footprint is under 50 MB.

### Phase 4: errors and HTTP clients

- Add `src/futurix_decision/errors.py`:
  - `FuturixError(Exception)` is the base class.
  - `ConfigError(FuturixError, ValueError)`: bad arguments or config. It subclasses `ValueError`, so existing `except ValueError` blocks still catch it.
  - `CheckpointError(FuturixError)`: missing files, wrong family, or a bad manifest.
  - `MissingDependencyError(FuturixError, ImportError)`: replaces `_missing()`.
  - `BackendError(FuturixError)`: a transport or server failure. It carries `status_code` and `body`.
  - `InvalidOutputError(BackendError)`: a publisher returned probabilities that fail validation (`native.py:221-242`).
- Replace the existing `raise ValueError` and `raise ImportError` sites with these. Most become `ConfigError`.
- Add `src/futurix_decision/client/`, a client for the SDK's own `futurix-decision serve` server, which currently has no client at all:
  - `DecisionClient` and `AsyncDecisionClient` provide `decide`, `batch_decide`, `choice`, `noul`, `score`, `health`, and `close`, and work as context managers.
  - Config: `base_url`, `api_key`, `timeout` (default 30s), and `max_retries` (default 3). Retries use exponential backoff with jitter on connect errors, 429, 502, 503, and 504, and honor `Retry-After`.
  - Responses parse into `DecisionResult`.
- Native HTTP runtime (`native.py:331-346`): build it through the same transport helper, so it also gets timeout and retry settings. `NativeDecisionModel` gains `close()`, `__enter__`, and `__exit__`, and closes only clients it created.
- Done when the client tests run against a FastAPI `TestClient` and an `httpx.MockTransport` that injects 503s, and the retry, timeout, and error-mapping tests pass.

### Phase 5: one typed backend contract

- Add `models/protocol.py`, defining `DecisionBackend(Protocol)`:
  - `decide(example, *, confidence_threshold=None) -> DecisionResult`
  - `batch_decide(examples, *, confidence_threshold=None, max_batch_tokens=None) -> list[DecisionResult]`
  - `choice(...)`, `noul(...)`, and `score(...)`
  - `close() -> None`
- `DecisionModel`, `NativeDecisionModel`, `VLLMEngine`, and `DecisionClient` all satisfy it. A test checks this with `isinstance` on a `runtime_checkable` protocol.
- `load_model` gets an explicit signature and returns `DecisionBackend`:
  - `strategy: Literal["pointer", "bosun", "letter_code"] | None`
  - `head_type`
  - `train_backbone`
  - `revision`
  - `device`
  - `**family_kwargs` only for family-specific knobs
- Remove the `pplx` aliases in `native.py:729` and `:735`, and `pplx-v1` everywhere else.
- `DecisionResult`:
  - Uses the per-kind `decision` type from the decisions table. Change `_result` (`native.py:247-251`) and the `DecisionModel` equivalent.
  - `latency_ms` always means amortized per-row time. `DecisionModel` stops stamping batch time on each row (`decision_model.py:920-940`).
  - Merge `DecisionResponse` into `DecisionResult` (`inference/schemas.py`).
- `fit_temperature` operates on returned probabilities instead of `.backbone` logits. That is exact, because `softmax(log p / T) == softmax(logits / T)`. It then works for every backend. Apply the fitted temperature through a `Calibrated(backend, temperature)` wrapper, and replace the hard-coded OpenJEV temperatures in `_verify_openjev` (`native.py:145-167`) with that wrapper and per-checkpoint values in the family table.
- `evaluate(model: DecisionBackend, ...)` replaces `model: Any`.
- Done when mypy passes on the public surface, and a test calls `load_model`, `fit_temperature`, and `evaluate` on both a `DecisionModel` and a fake native backend.

### Phase 6: family adapters (stack of PRs)

The largest phase. Each PR moves one family and keeps all tests green.

- **6a. Adapter interface.** Add `models/families/base.py` with `FamilyAdapter`:
  - `name`
  - `checkpoints`
  - `extra` (the pip extra to install)
  - `load(...) -> DecisionBackend`
  - `finetune(...)`, which is optional
  - `requires(path)`, which replaces the branches in `require_checkpoint_files`

  Replace `models/registry.py` (28 lines, unused for this) with a registry of adapters. `load_model`, `cli._train`, `_evaluate`, `_export`, and `_serve` all dispatch through it, which deletes the four repeated `from_decider` / `from_jpt` / `from_eikos` / `from_pretrained` chains in `cli.py`.
- **6b. Shared label-readout helpers.** Decider, Eikos, and JPT copy these pieces: the label table, `_LABEL_CACHE`, pad/EOS registration, left-pad batching, the `sha256` row seed, and `from_native` kwarg filtering. Move them into one `LabelReadoutModel(DecisionModel)` base. Per-family subclasses keep only the prompt and the noul order. Eikos either honors `assignment_seed` or drops the parameter.
- **6c. Native families.** Laya, GLiNER, KEV, CLEF, OpenJEV, and HTTP each become an adapter. This removes the `self.family ==` branching in `native.py` `_predict`, `system_one`, and `batch_decide`. `finetune_native` (`trainers/native.py`) dispatches to `adapter.finetune` and stops importing private `_images_for_native` and `_question`.
- **6d. Split `DecisionModel`.**
  - Move token, pointer, and letter-code readouts into `ReadoutStrategy` classes. That removes the 125-line `forward_decisions` and the `type(self) is DecisionModel` checks.
  - Merge `_register_candidate_marker` and `_register_decision_tokens` into one helper.
  - Turn the 20-parameter constructor into a `DecisionModelConfig` dataclass.
- **6e. Single family table.** Collapse `ModelFamily.checkpoints` and `supported_checkpoints` into one field. `infer_local_family`, `identify_family`, and the README support matrix all read from it. A test fails if the README matrix and the table disagree.
- Done when `native.py` and `decision_model.py` are each under 400 lines, `cli.py` has no per-family `if` chains, and adding a family touches one new file plus one registry line.

### Phase 7: runtime correctness and serving

- `DecisionModel.batch_decide` chunks by `max_batch_tokens`, reusing the token-budget batching in `evaluator.py:49`. Add a test that 10,000 rows stay within a set peak memory on CPU.
- Server (`vllm/api.py`):
  - Drop the global `threading.Lock` (`:78`).
  - The vLLM backend gets async endpoints over vLLM's async engine.
  - The HF backend gets a micro-batching queue: collect requests for up to `--batch-window-ms` (default 5) or `--max-batch` rows, run one `batch_decide`, and fan the results out.
  - Optional bearer auth through `--api-key` or the `FUTURIX_API_KEY` env var. Without a key, the server binds `127.0.0.1` by default.
  - Either export `create_vllm_app` from `vllm/__init__.py` and document it, or delete it.
- `PointerDecisionHead.forward` applies `available` or drops the parameter.
- Make the unstrict `zip` calls in `decision_model._infer` strict (ruff B905). They are only sound if lengths match.
- Done when a load test with 64 concurrent requests on the tiny GPT-2 server shows batching (fewer forward calls than requests), and the auth tests pass.

### Phase 8: logging

- `logging.getLogger("futurix_decision")` throughout, with `NullHandler` on the package logger.
- `utils/logging.TrainingLogger` keeps writing JSONL metrics, but sends human-readable output through the logger instead of `print`.
- The CLI configures logging with `--log-level`.
- Done when `grep -rn "print(" src/` hits only CLI user output.

### Phase 9: docs

- Cut the README to about 150 lines: what it is, install (`pip install futurix-decision[torch]` and the client-only install), a 10-line quickstart for local and HTTP use, a family table, and links.
- Move the rest into `docs/`:
  - run logs and benchmark results (`README.md:922`, `:989`, and the rest) go to `docs/benchmarks.md`;
  - training walkthroughs go to `docs/datasets-and-training.md`;
  - the per-family notes, including the noul order, go to `docs/model-support.md`.
- Add `mkdocs-material` and `mkdocstrings` for an API reference generated from docstrings. CI runs `mkdocs build --strict`, and publishing is out of scope.
- Fill in `CONTRIBUTING.md`'s "adding a model family" section against the Phase 6 adapter interface.
- Done when every public symbol in `__all__` has a docstring and the docs build clean.

### Phase 10: cut 0.2.0

- Write the `CHANGELOG.md` migration section with:
  - the `decision` type change;
  - the `latency_ms` meaning;
  - removed `pplx` aliases;
  - the `load_model` signature;
  - the new exception classes;
  - the `[torch]` extra;
  - the `fit_temperature` behavior.
- Bump `__version__` to `0.2.0`, tag `v0.2.0`, and create a GitHub release with built wheel and sdist attached.

## Order and dependencies

```
0 ─ 1 ─ 2 ─ 3 ─ 4 ─ 5 ─ 6a ─ 6b ─ 6c ─ 6d ─ 6e ─ 7 ─ 8 ─ 9 ─ 10
```

- 1 comes first so every later PR runs under CI.
- 3 comes before 4 because the client must live in the torch-free core.
- 5 comes before 6 because the protocol is what the adapters return, and its tests guard the refactor.
- 7 comes after 6 so batching and serving are written once, against the adapter interface.
- 8 and 9 can run in parallel with 6 and 7 if someone else picks them up.

## Out of scope

- GPU nightly job against released checkpoints.
- PyPI publishing.
- Hosted docs.
