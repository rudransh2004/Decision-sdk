# Contributing

## Set up

The SDK needs Python 3.10 or later. These steps use [uv](https://docs.astral.sh/uv/);
plain `pip` works too.

```bash
uv venv
uv pip install -e ".[dev,serve]"
```

On a machine without a GPU, set `UV_TORCH_BACKEND=cpu` before the install to get the
smaller CPU-only torch wheels.

## Run the tests

```bash
python -m pytest -q
```

The tests run on CPU. Some tests skip unless an extra is installed or an opt-in
variable is set, for example `FUTURIX_TEST_VLLM=1` for the GPU vLLM smoke test.

## Lint, format, and type-check

```bash
ruff check .
ruff format .
mypy
```

CI runs `ruff check`, `ruff format --check`, and `mypy`, and fails on any finding.

Mypy runs non-strict. `inference/`, `datasets/schema.py`, `client/`, and `errors.py`
also require annotated function definitions. Some older modules disable specific error
codes in `pyproject.toml`. When you fix a module, remove its codes from that list.

To hide the bulk formatting commit from `git blame`, run this once:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Check an extra

CI installs each extra on its own and imports what it provides. To check one locally:

```bash
uv venv /tmp/extra-check
uv pip install --python /tmp/extra-check/bin/python ".[bosun]"
/tmp/extra-check/bin/python -c "import futurix_decision.models.bosun"
```

vLLM does not build on CPU, so for the vLLM extras CI only checks that the pins resolve:

```bash
uv pip compile pyproject.toml --extra decider-vllm --python-platform x86_64-manylinux_2_28
```

## Add a model family

Each family is a `FamilyAdapter` (`src/futurix_decision/models/families/base.py`).
`load_model` and the CLI find adapters by name in the registry
(`src/futurix_decision/models/registry.py`). To add a family:

1. Create `src/futurix_decision/models/families/<family>.py` with a `FamilyAdapter`
   subclass. Import torch and the publisher's packages inside methods, not at module
   level, so `import futurix_decision` stays torch-free.
2. Add one instance of it to the list at the end of `registry.py`.
3. Add tests, and a row for the family in the family table
   (`src/futurix_decision/models/families/__init__.py`) and the README matrix.

The adapter sets three attributes and implements `load`:

```python
class FamilyAdapter(ABC):
    name: str  # family= in load_model, --model-format in the CLI
    checkpoints: tuple[str, ...]  # Hub IDs load_model recognizes by name
    extra: str | None  # pip extra that installs the family

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs) -> DecisionBackend: ...
    def requires(self, path: Path) -> None: ...  # optional
    def finetune(self, model, dataset, output_dir, **kwargs) -> DecisionBackend: ...  # optional
    def training_kwargs(self, section, *, method: str) -> dict: ...  # optional
```

- `load` returns any `DecisionBackend`. It receives the `load_model` keyword arguments,
  or the options of the CLI command. Raise `ConfigError` for options the family does
  not support.
- `requires` raises `CheckpointError` when a local directory lacks a file `load` needs.
  `load_model` calls it before `load`. Today the built-in adapters delegate to
  `require_checkpoint_files` in `models/sources.py`.
- `finetune` trains in the family's own format and returns the reloaded backend.
  Implement it only when the family has its own trainer; the others train through
  `DecisionTrainer`, which needs `load` to return a `DecisionModel`.
- `training_kwargs` turns the `model:` section of a `futurix-decision train` config into
  `load` options. The default accepts a section that sets no family options.

`load_model` still identifies local checkpoints through `infer_local_family` in
`models/sources.py`. A family without a recognizable file layout loads with
`family=` passed explicitly.

Code outside the SDK can register a family at runtime:

```python
from futurix_decision import FamilyAdapter, load_model, register_family


class AcmeAdapter(FamilyAdapter):
    name = "acme"
    checkpoints = ("acme/decider-1",)

    def load(self, checkpoint, *, device=None, **kwargs):
        return AcmeBackend(checkpoint, device=device, **kwargs)


register_family(AcmeAdapter())
model = load_model("acme/decider-1")
```

After `register_family`, `futurix-decision evaluate --model-format acme` also loads it,
when the CLI runs in the same process (for example through `futurix_decision.cli.main`).

## Changelog

Add an entry under `## [Unreleased]` in `CHANGELOG.md` for every user-visible change.
