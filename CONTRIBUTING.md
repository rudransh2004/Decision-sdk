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

This section will be written after the family adapter refactor (Phase 6 of
`docs/plans/sdk-0.2.md`). That refactor gives each family one adapter file and one
registry line.

## Changelog

Add an entry under `## [Unreleased]` in `CHANGELOG.md` for every user-visible change.
