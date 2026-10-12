"""A row's decision must not depend on the rows batched with it.

The server micro-batches unrelated requests, so a short prompt padded to a
long neighbour has to score exactly as it does alone, on absolute-position
(GPT-2) and rotary (Llama, Qwen3.5) backbones, for every readout.
"""

import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from futurix_decision import DecisionInput, DecisionModel, load_model
from test_bosun import bosun_tiny_model
from test_custom_head import make_model
from test_decider import make_native
from test_decision_model import tiny_checkpoint
from test_letter_code_conversion import qwen35_checkpoint, raw_checkpoint
from test_letter_readout import build as build_letter_family
from test_readout_pins import BUILDERS

SHORT = DecisionInput(state="review", question="Next?", options=["approve", "reject"])
LONG = DecisionInput(
    # Words and capitals, so each test tokenizer encodes this far longer than SHORT.
    state={"ticket": "refund", "notes": " ".join(["approve reject escalate A B C D"] * 6)},
    question="Which way should this ticket go next?",
    options=["approve", "reject", "escalate"],
    candidate_descriptions=["approve one", "reject it", "send up the chain"],
)


def llama(vocab_size: int) -> LlamaForCausalLM:
    torch.manual_seed(31)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
        )
    )


def on_llama(base: DecisionModel, **kwargs) -> DecisionModel:
    """``base``'s tokenizer and readout settings over a tiny Llama backbone."""
    config = base.config
    return DecisionModel(
        llama(len(base.tokenizer)),
        base.tokenizer,
        decision_tokens=config.decision_tokens,
        max_sequence_length=config.max_sequence_length,
        prompt_format=config.prompt_format,
        head_type=config.head_type,
        head_prompt_version=config.head_prompt_version,
        learn_abstain=config.learn_abstain,
        **kwargs,
    )


def llama_letter_code(root: Path) -> DecisionModel:
    base = root / "base"
    base.mkdir()
    raw_checkpoint(base)
    llama(len(AutoTokenizer.from_pretrained(base))).save_pretrained(base)
    return load_model(base, convert=True, strategy="letter_code", decision_tokens=8)


def llama_decider(root: Path) -> DecisionModel:
    source = root / "native"
    source.mkdir()
    make_native(source)
    llama(len(AutoTokenizer.from_pretrained(source))).save_pretrained(source)
    return DecisionModel.from_decider(source, max_sequence_length=1024)


def qwen35_letter_code(root: Path) -> DecisionModel:
    base = root / "base"
    base.mkdir()
    qwen35_checkpoint(base)
    model = load_model(
        base, convert=True, strategy="letter_code", decision_tokens=8, max_sequence_length=512
    )
    assert model.config.attention_mode == "noncausal_full_attention"
    return model


GPT2 = {
    **BUILDERS,
    "decider": lambda root: build_letter_family("decider", root),
    "jpt": lambda root: build_letter_family("jpt", root),
    "eikos": lambda root: build_letter_family("eikos", root),
}
ROTARY = {
    "llama-tokens": lambda root: on_llama(tiny_checkpoint()),
    "llama-bosun": lambda root: on_llama(bosun_tiny_model()),
    "llama-pointer-marker": lambda root: on_llama(make_model()),
    "llama-pointer-span": lambda root: on_llama(BUILDERS["pointer-span"](root)),
    "llama-letter-code": llama_letter_code,
    "llama-decider": llama_decider,
}


def assigned(model: DecisionModel, rows: list[DecisionInput]) -> list[torch.Tensor]:
    with torch.no_grad():
        batch = model.forward_decisions(rows)
    return [batch.assigned_logits(index).float() for index in range(len(rows))]


class BatchInvarianceTests(unittest.TestCase):
    def check(self, name: str, model: DecisionModel) -> None:
        model.backbone.eval()
        if model.decision_head is not None:
            model.decision_head.eval()
        self.assertGreater(
            model.readout.prompt_tokens(LONG), model.readout.prompt_tokens(SHORT) + 8
        )
        together = assigned(model, [SHORT, LONG])
        for index, row in enumerate((SHORT, LONG)):
            torch.testing.assert_close(
                together[index],
                assigned(model, [row])[0],
                rtol=0,
                atol=1e-5,
                msg=lambda text, index=index: f"{name} row {index} logits: {text}",
            )
        batched = model.batch_decide([SHORT, LONG])[0].probabilities
        alone = model.decide(SHORT).probabilities
        for option in SHORT.options:
            self.assertAlmostEqual(batched[option], alone[option], delta=1e-5, msg=name)

    def check_all(self, builders) -> None:
        for name, builder in builders.items():
            with self.subTest(readout=name), tempfile.TemporaryDirectory() as temp:
                self.check(name, builder(Path(temp)))

    def test_absolute_position_backbone(self):
        self.check_all(GPT2)

    def test_rotary_backbone(self):
        self.check_all(ROTARY)

    @unittest.skipUnless(
        tuple(int(part) for part in version("transformers").split(".")[:2]) >= (5, 17),
        "Qwen3.5 architecture requires Transformers 5.17+",
    )
    def test_qwen35_noncausal_letter_code(self):
        with tempfile.TemporaryDirectory() as temp:
            self.check("qwen3.5-letter-code", qwen35_letter_code(Path(temp)))
