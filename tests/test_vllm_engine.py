"""VLLMEngine against the PyTorch readout of the same checkpoint, through a fake vLLM.

``ModelLLM`` answers ``generate`` the way vLLM does with restricted processed
logprobs: one forward per prompt, log-softmax over the allowed token IDs at the
last position. Every prompt format must then reproduce ``DecisionModel``'s
probabilities, so a change to prompt building, label tables, temperatures,
option reordering, or row grouping fails here.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from futurix_decision import DecisionInput
from futurix_decision.vllm.engine import VLLMEngine
from test_bosun import bosun_tiny_model
from test_letter_readout import INPUTS as LETTER_INPUTS
from test_letter_readout import build as build_letter_family
from test_readout_pins import INPUTS as TOKEN_INPUTS
from test_sft import tiny_model


class ModelLLM:
    def __init__(self, backbone):
        self.backbone = backbone.eval()
        self.calls = 0
        self.prompts = 0

    def generate(self, prompts, params, use_tqdm=False):
        self.calls += 1
        self.prompts += len(prompts)
        outputs = []
        for prompt, setting in zip(prompts, params, strict=True):
            assert setting.max_tokens == 1
            assert setting.logprobs == len(setting.allowed_token_ids)
            ids = torch.tensor([prompt["prompt_token_ids"]])
            with torch.no_grad():
                logits = self.backbone(input_ids=ids).logits[0, -1].float()
            allowed = torch.log_softmax(logits[setting.allowed_token_ids], dim=-1).tolist()
            logprobs = {
                token: SimpleNamespace(logprob=value)
                for token, value in zip(setting.allowed_token_ids, allowed, strict=True)
            }
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[logprobs])]))
        return outputs


def sampling_params(**kwargs):
    return SimpleNamespace(**kwargs)


def fake_vllm():
    return patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=sampling_params)})


def models(root: Path):
    yield "futurix", tiny_model(), TOKEN_INPUTS
    yield "bosun-v3", bosun_tiny_model(), TOKEN_INPUTS
    for family in ("decider", "jpt", "eikos"):
        directory = root / family
        directory.mkdir()
        yield family, build_letter_family(family, directory), LETTER_INPUTS


class VLLMEngineTests(unittest.TestCase):
    def test_every_prompt_format_matches_the_pytorch_readout(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, model, inputs in models(Path(temp)):
                with self.subTest(prompt_format=name):
                    export = Path(temp) / f"{name}-export"
                    model.save_pretrained(export)
                    # One row per forward: padding shifts positions in some tiny backbones.
                    expected = [model.decide(row) for row in inputs]
                    llm = ModelLLM(model.backbone)
                    with fake_vllm():
                        engine = VLLMEngine(export, llm=llm, tokenizer=model.tokenizer)
                        actual = engine.batch_decide(inputs)
                    self.assertEqual(llm.calls, 1)
                    for row, before, after in zip(inputs, expected, actual, strict=True):
                        self.assertEqual(after.kind, row.kind)
                        self.assertEqual(after.decision, before.decision)
                        for option in row.options:
                            self.assertAlmostEqual(
                                after.probabilities[option],
                                before.probabilities[option],
                                places=4,
                                msg=f"{name} {row.kind} {option}",
                            )
                    for row in inputs:
                        self.assertEqual(
                            engine.prompt_tokens(row), model.readout.prompt_tokens(row), name
                        )

    def test_short_output_is_invalid(self):
        from futurix_decision.errors import InvalidOutputError

        model = tiny_model()
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)

            class Short(ModelLLM):
                def generate(self, prompts, params, use_tqdm=False):
                    return super().generate(prompts, params)[:-1]

            row = DecisionInput(state="s", question="q", options=["a", "b"])
            with fake_vllm():
                engine = VLLMEngine(temp, llm=Short(model.backbone), tokenizer=model.tokenizer)
                with self.assertRaisesRegex(InvalidOutputError, "different number"):
                    engine.batch_decide([row, row])


if __name__ == "__main__":
    unittest.main()
