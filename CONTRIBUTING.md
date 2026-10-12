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

## Write docs

The docs are Markdown in `docs/`, built with MkDocs Material and mkdocstrings. The API
reference reads docstrings, so every public symbol needs one (a test checks this).

```bash
uv pip install -e ".[docs]"
mkdocs serve          # preview
mkdocs build --strict # what CI runs; any warning fails
```

`tests/test_docs_snippets.py` compiles every ```` ```python ```` block in `README.md` and
`docs/*.md`. A block preceded by a `<!-- test -->` line also runs, against a tiny local
model and an in-process server, so keep those blocks self-contained. Put
`<!-- test-skip: reason -->` before a block that looks runnable but needs a GPU or a
download. The test also checks that each `futurix-decision` flag in a `bash` block exists,
and that `docs/api.md` lists every name in `futurix_decision.__all__`.

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

Each family is a `FamilyAdapter` (`src/futurix_decision/models/families/base.py`), and
the adapter is the only place the SDK records the family. `load_model` and the CLI find
adapters through the registry (`src/futurix_decision/models/registry.py`), which derives
Hub ID lookup, short names, and local-copy detection from the adapters' attributes. To
add a family:

1. Create `src/futurix_decision/models/families/<family>.py` with a `FamilyAdapter`
   subclass. Import torch and the publisher's packages inside methods, not at module
   level, so `import futurix_decision` stays torch-free.
2. Import it in `registry.py` and add one instance to the list at the end.
3. Add tests, and a README "Supported decision models" row if the family lists
   `checkpoints`. `tests/test_family_registry.py` fails when a row's Hub IDs, training
   columns, or vLLM column disagree with the adapter.

The adapter sets `name`, implements `load`, and declares what it supports:

```python
class FamilyAdapter(ABC):
    name: str  # family= in load_model, --model-format in the CLI
    checkpoints: tuple[str, ...] = ()  # Hub IDs load_model loads by name
    unsupported_checkpoints: tuple[str, ...] = ()  # published, recognized only to refuse
    aliases: Mapping[str, str] = {}  # short name -> Hub ID in checkpoints
    markers: tuple[str, ...] = ()  # files that identify a local copy
    extra: str | None = None  # pip extra that installs the family
    trainable: bool = False  # trains on labeled data (DecisionTrainer or finetune)
    vllm: Literal["direct", "export", "server"] | None = None

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs) -> DecisionBackend: ...
    def detect(self, directory: Path) -> bool: ...  # optional; default checks markers
    def requires(self, path: Path) -> None: ...  # optional
    def finetune(self, model, dataset, output_dir, **kwargs) -> DecisionBackend: ...  # optional
    def training_kwargs(self, section, *, method: str) -> dict: ...  # optional
    def model_class(self) -> type[DecisionModel] | None: ...  # optional
```

- `load` returns any `DecisionBackend`. It receives the `load_model` keyword arguments,
  or the options of the CLI command. Raise `ConfigError` for options the family does
  not support.
- `detect` tells `load_model` that a local directory without `family=` belongs to this
  family. The registry asks families in registration order and takes the first `True`,
  so check something only this family's releases contain. Override it to read a config
  file; `optional_json` in `models/sources.py` reads one that may be absent. When no
  family claims a directory, a family whose Hub ID ends in the directory's name does.
- `requires` raises `CheckpointError` when a local directory lacks a file `load` needs.
  `load_model` calls it before `load`, so `load` need not check again. Build it with
  `CheckpointFiles` from `models/sources.py`: `require`, `one_of`, `tokenizer`, and
  `weights` collect the missing files, and `check` raises one error naming them all.
- `finetune` trains in the family's own format and returns the reloaded backend.
  Implement it only when the family has its own trainer; the others train through
  `DecisionTrainer`, which needs `load` to return a `DecisionModel`.
- `training_kwargs` turns the `model:` section of a `futurix-decision train` config into
  `load` options. The default accepts a section that sets no family options.
- `model_class` names the `DecisionModel` subclass that reads the family's saved
  checkpoints, so `DecisionModel.from_pretrained` can pick it from a saved
  `prompt_format`. Decider, JPT, and Eikos subclass `LetterAdapter`
  (`models/families/letter.py`), which provides it.
- `vllm` decides what `--backend vllm` does: `"direct"` checkpoints load in
  `VLLMEngine`, `"export"` ones need `futurix-decision export` first, and `"server"`
  means the publisher's own vLLM-backed System One server.

`register_family` refuses a Hub ID or alias that another family already claims.

A family that runs a publisher's own code subclasses `PublisherAdapter`
(`models/families/publisher.py`) instead; `kev.py` and `laya.py` are short examples.
It also sets `readout` and the runtime capabilities (`images`, `max_options`,
`remote_code`, `transformers`, `http_error`), implements `open` to load the publisher's
model object and `files` to list its checkpoint files, and pairs with a
`PublisherRuntime` subclass that answers System One payloads. `NativeDecisionModel`
enforces the capabilities and validates the answers. A trainable family implements
`train`; `finetune_native` writes the `futurix_native.json` manifest, which names the
family, and reloads the export.

Code outside the SDK registers a family at runtime, with the same attributes:

```python
from futurix_decision import FamilyAdapter, load_model, register_family


class AcmeAdapter(FamilyAdapter):
    name = "acme"
    checkpoints = ("acme/decider-1",)
    aliases = {"acme-small": "acme/decider-1"}
    markers = ("acme.json",)

    def load(self, checkpoint, *, device=None, **kwargs):
        return AcmeBackend(checkpoint, device=device, **kwargs)


register_family(AcmeAdapter())
model = load_model("acme-small")
local = load_model("./downloads/acme")  # found by its acme.json
```

After `register_family`, `futurix-decision evaluate --model-format acme` also loads it,
when the CLI runs in the same process (for example through `futurix_decision.cli.main`).

## Changelog

Add an entry under `## [Unreleased]` in `CHANGELOG.md` for every user-visible change.
