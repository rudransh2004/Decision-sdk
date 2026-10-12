"""JPT text prompt/readout checks against a tiny local test backbone."""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from futurix_decision import DecisionInput, DecisionModel, DecisionTrainer, SFTConfig
from futurix_decision.datasets.schema import DecisionExample
from futurix_decision.models.jpt import JPTDecisionModel, _prompt, prepare_jpt_row
from futurix_decision.vllm.engine import VLLMEngine
from test_decider import make_native


def make_jpt(path: Path):
    make_native(path)
    (path / "decider_config.json").unlink()
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(path)
    backbone = AutoModelForCausalLM.from_pretrained(path)
    return JPTDecisionModel(
        backbone,
        tokenizer,
        decision_tokens=255,
        temperature=1.036,
        max_sequence_length=512,
    )


class JPTTests(unittest.TestCase):
    def test_chat_template_no_new_tokens_true_first_and_contextual_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            model = make_jpt(Path(temp))
            self.assertNotIn("<|decision_000|>", model.tokenizer.get_vocab())
            noul = DecisionInput(
                kind="noul",
                state={"policy": "review"},
                question="Valid?",
                options=["false", "true"],
            )
            ids, labels, order = prepare_jpt_row(noul, model.tokenizer)
            text = _prompt(model.tokenizer, noul, order)
            self.assertIn("A. Yes: yes", text)
            self.assertIn("B. No: no", text)
            self.assertIn("Answer:", text)
            self.assertEqual(order, [1, 0])
            self.assertEqual(labels, model.readout.token_ids[:2])
            batch = model.forward_decisions([noul])
            self.assertEqual(batch.slots, [[0, 1]])
            self.assertAlmostEqual(sum(model.batch_decide([noul])[0].probabilities.values()), 1)
            shuffled = model.forward_decisions([noul], assignment_seed=19)
            self.assertTrue(torch.isfinite(shuffled.logits).all())
            labelled = DecisionInput(
                state="review",
                question="Next?",
                options=["c000", "c001"],
                candidate_ids=["c000", "c001"],
                candidate_labels=["Reject", "Approve"],
                candidate_descriptions=["No action", "Finish work"],
            )
            prompt = _prompt(model.tokenizer, labelled, [0, 1])
            self.assertIn("c001: Approve — Finish work", prompt)

    def test_jpt_finetuning_export_and_vllm_reorders_noul(self):
        with tempfile.TemporaryDirectory() as temp:
            model = make_jpt(Path(temp) / "source")
            rows = [
                DecisionExample(
                    id="x",
                    state="review",
                    question="Next?",
                    options=["approve", "reject"],
                    target="approve",
                )
            ]
            if importlib.util.find_spec("accelerate"):
                trainer = DecisionTrainer(
                    model=model,
                    dataset=rows,
                    sft_config=SFTConfig(batch_size=1),
                )
                trainer.train(stages=["decision_sft"])
            exported = Path(temp) / "export"
            model.save_pretrained(exported)
            # Transformers 4.x cannot load Qwen3.5 multimodal releases. Patch
            # the synthetic GPT2 test fixture to model its 5.x auto-loader.
            with patch("transformers.AutoModelForMultimodalLM", AutoModelForCausalLM, create=True):
                restored = DecisionModel.from_pretrained(exported)
            self.assertIsInstance(restored, JPTDecisionModel)
            choice = DecisionInput(state="review", question="Next?", options=["approve", "reject"])
            before, after = model.decide(choice), restored.decide(choice)
            for key in before.probabilities:
                self.assertAlmostEqual(before.probabilities[key], after.probabilities[key])

            class SamplingParams:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            class FakeLLM:
                def generate(self, prompts, params, use_tqdm=False):
                    assert len(prompts) == 1
                    assert params[0].allowed_token_ids == model.readout.token_ids[:2]
                    return [
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    logprobs=[
                                        {
                                            token: SimpleNamespace(logprob=math.log(prob))
                                            for token, prob in zip(
                                                params[0].allowed_token_ids,
                                                [0.8, 0.2],
                                                strict=True,
                                            )
                                        }
                                    ]
                                )
                            ]
                        )
                    ]

            with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SamplingParams)}):
                vllm = VLLMEngine(exported, llm=FakeLLM(), tokenizer=model.tokenizer)
                response = vllm.noul(state="review")
            self.assertTrue(response.decision)
            self.assertGreater(response.probabilities["true"], response.probabilities["false"])


if __name__ == "__main__":
    unittest.main()
