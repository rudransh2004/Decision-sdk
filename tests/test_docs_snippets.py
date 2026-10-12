"""The code in README.md and docs/*.md matches the current API.

Every ```python block must compile. A block directly preceded by ``<!-- test -->``
is also executed, against a tiny local model and an in-process server in place of
downloads and a network. Blocks that need a GPU or a large download stay
compile-only. ``<!-- test-skip: reason -->`` marks a block that looks runnable but
is not; the reason is for the reader.

Hub IDs in the snippets map to a tiny causal LM built here. ``load_model("laya")``
returns a stub backend, and the HTTP clients talk to a FastAPI app around that stub.
"""

import contextlib
import importlib.util
import io
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

import futurix_decision
from futurix_decision import DecisionResult
from futurix_decision.models.protocol import DecisionShortcuts

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
MARKER = re.compile(r"<!-- (test|test-skip[^>]*) -->")

# Names the snippets use for a base model they would download.
TINY_LM_NAMES = {
    "HuggingFaceTB/SmolLM2-135M",
    "your-decoder-checkpoint",
    "your-org/text-causal-lm",
    "your-plain-causal-LM",
}

DECISIONS = (
    '{"id":"ticket-1","kind":"choice","state":"Customer was charged twice",'
    '"question":"Which team?","options":["billing","technical"],"target":"billing"}\n'
    '{"id":"ticket-2","kind":"choice","state":"The app crashes on start",'
    '"question":"Which team?","options":["billing","technical"],"target":"technical"}\n'
)


@dataclass(frozen=True)
class Block:
    path: Path
    line: int
    language: str
    code: str
    marker: str | None

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}"


def extract_blocks() -> list[Block]:
    blocks = []
    for path in DOCS:
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            marker = MARKER.fullmatch(lines[index].strip())
            fence = index + 1 if marker else index
            opening = lines[fence] if fence < len(lines) else ""
            if opening.startswith("```") and opening.strip("`").strip():
                end = fence + 1
                while not lines[end].startswith("```"):
                    end += 1
                language = opening.strip("`").strip()
                code = "\n".join(lines[fence + 1 : end]) + "\n"
                blocks.append(
                    Block(path, fence + 2, language, code, marker.group(1) if marker else None)
                )
                index = end
            elif marker:
                pytest.fail(f"{path.name}:{index + 1}: marker is not followed by a code block")
            index += 1
    return blocks


BLOCKS = extract_blocks()
PYTHON = [block for block in BLOCKS if block.language == "python"]
RUNNABLE = [block for block in PYTHON if block.marker == "test"]
BASH = [block for block in BLOCKS if block.language == "bash"]


class StubBackend(DecisionShortcuts):
    """Puts 0.7 on the first option and splits the rest."""

    def decide(self, example, *, confidence_threshold=None):
        return self.batch_decide([example], confidence_threshold=confidence_threshold)[0]

    def batch_decide(self, examples, *, confidence_threshold=None):
        results = []
        for example in examples:
            rest = 0.3 / (len(example.options) - 1)
            probabilities = [0.7] + [rest] * (len(example.options) - 1)
            results.append(
                DecisionResult.from_probabilities(
                    example,
                    probabilities,
                    latency_ms=1.0,
                    confidence_threshold=confidence_threshold,
                )
            )
        return results

    def close(self):
        pass


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    """A saved GPT-2 with a two-word tokenizer, standing in for any Hub causal LM."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory = tmp_path_factory.mktemp("docs-models") / "tiny-lm"
    torch.manual_seed(0)
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
    config = GPT2Config(vocab_size=2, n_positions=512, n_embd=32, n_layer=1, n_head=2)
    GPT2LMHeadModel(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def trained_model(tiny_lm, tmp_path_factory):
    """``./my-decision-model``: a converted tiny pointer model, exported once."""
    pytest.importorskip("accelerate")
    from futurix_decision import DecisionTrainer, SFTConfig, load_model

    workdir = tmp_path_factory.mktemp("docs-trained")
    (workdir / "decisions.jsonl").write_text(DECISIONS)
    model = load_model(str(tiny_lm), convert=True, strategy="pointer", decision_tokens=32)
    trainer = DecisionTrainer(
        model=model,
        dataset=str(workdir / "decisions.jsonl"),
        sft_config=SFTConfig(method="full", batch_size=2),
    )
    trainer.train(stages=["decision_sft"])
    trainer.export(workdir / "my-decision-model")
    return workdir / "my-decision-model"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    for name in ("decisions", "heldout", "train", "calibration"):
        (tmp_path / f"{name}.jsonl").write_text(DECISIONS)
    (tmp_path / "examples").symlink_to(ROOT / "examples", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def servers(monkeypatch):
    """Point both HTTP clients at an in-process app around the stub backend."""
    if not importlib.util.find_spec("fastapi"):
        pytest.skip("serve extra not installed")
    from fastapi.testclient import TestClient

    from futurix_decision.vllm.api import create_app

    app = create_app(StubBackend())

    class LocalClient(futurix_decision.DecisionClient):
        def __init__(self, base_url=None, *, api_key=None, **settings):
            super().__init__(http_client=TestClient(app, base_url=base_url or "http://testserver"))

    class LocalAsyncClient(futurix_decision.AsyncDecisionClient):
        def __init__(self, base_url=None, *, api_key=None, **settings):
            transport = httpx.ASGITransport(app=app)
            http = httpx.AsyncClient(transport=transport, base_url=base_url or "http://testserver")
            super().__init__(http_client=http)

    monkeypatch.setattr(futurix_decision, "DecisionClient", LocalClient)
    monkeypatch.setattr(futurix_decision, "AsyncDecisionClient", LocalAsyncClient)
    yield
    app.state.batcher.close()


@pytest.fixture
def patched_loaders(monkeypatch, tiny_lm):
    from futurix_decision.models import DecisionModel

    real_load_model = futurix_decision.load_model
    real_from_pretrained = DecisionModel.from_pretrained.__func__

    def load_model(checkpoint, **kwargs):
        if str(checkpoint) in TINY_LM_NAMES:
            checkpoint = str(tiny_lm)
        if str(checkpoint) == "laya":
            return StubBackend()
        return real_load_model(checkpoint, **kwargs)

    def from_pretrained(cls, checkpoint, **kwargs):
        if str(checkpoint) in TINY_LM_NAMES:
            checkpoint = str(tiny_lm)
        return real_from_pretrained(cls, checkpoint, **kwargs)

    monkeypatch.setattr(futurix_decision, "load_model", load_model)
    monkeypatch.setattr(DecisionModel, "from_pretrained", classmethod(from_pretrained))


@pytest.mark.parametrize("block", PYTHON, ids=lambda block: block.id)
def test_python_block_compiles(block):
    compile(block.code, block.id, "exec")


@pytest.mark.parametrize("block", RUNNABLE, ids=lambda block: block.id)
def test_marked_python_block_runs(block, workdir, trained_model, servers, patched_loaders):
    shutil.copytree(trained_model, workdir / "my-decision-model")
    exec(compile(block.code, block.id, "exec"), {"__name__": "docs_snippet"})


def test_the_readme_quickstarts_run():
    readme = [block for block in RUNNABLE if block.path.name == "README.md"]
    assert len(readme) >= 2, "README.md needs a runnable local and HTTP quickstart"


def cli_help(command):
    from futurix_decision.cli import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        main([command, "--help"])
    return out.getvalue()


@pytest.mark.parametrize("block", BASH, ids=lambda block: block.id)
def test_cli_flags_in_bash_blocks_exist(block):
    text = block.code.replace("\\\n", " ")
    for line in text.splitlines():
        words = line.split()
        if "futurix-decision" not in words:
            continue
        rest = words[words.index("futurix-decision") + 1 :]
        if "--log-level" in rest:
            rest.remove("--log-level")
            rest.pop(0)
        if not rest or rest[0].startswith("-"):
            continue
        command = rest[0]
        help_text = cli_help(command)
        for word in rest[1:]:
            if word.startswith("--"):
                flag = word.split("=")[0]
                assert re.search(rf"(?<![\w-]){flag}(?![\w-])", help_text), (
                    f"{block.id}: `futurix-decision {command}` has no {flag}"
                )


def test_api_reference_lists_every_public_symbol():
    page = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^::: futurix_decision\.(?:client\.|errors\.)?(\w+)$", page, re.M))
    missing = sorted(set(futurix_decision.__all__) - listed)
    assert not missing, f"docs/api.md does not list {missing}"
