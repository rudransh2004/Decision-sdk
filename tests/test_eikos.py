"""Eikos SemIf protocol and train/export/vLLM checks on a tiny local model."""

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from futurix_decision import DecisionInput, DecisionModel, DecisionTrainer, SFTConfig
from futurix_decision.datasets.schema import DecisionExample
from futurix_decision.errors import CheckpointError
from futurix_decision.models.eikos import EikosDecisionModel, eikos_label_table, prepare_eikos_row
from futurix_decision.vllm.engine import VLLMEngine
from test_decider import make_native


def make_eikos(path: Path):
    make_native(path)
    (path / "decider_config.json").unlink()
    (path / "decision_config.json").write_text(
        json.dumps(
            {
                "prompt_version": "letter-v1-semif",
                "readout": "letter-logit",
                "max_one_pass": 100,
                "calib": "calib.json",
            }
        )
    )
    (path / "calib.json").write_text(
        json.dumps(
            {
                "mode": "T1",
                "t_global": 1.0,
                "b": 0,
                "w": [0, 0, 0, 0],
            }
        )
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(path)


class EikosTests(unittest.TestCase):
    def test_native_import_chat_json_order_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_eikos(source)
            with self.assertRaisesRegex(CheckpointError, "non-Futurix"):
                DecisionModel.from_pretrained(source)
            model = DecisionModel.from_eikos(source, max_sequence_length=512)
            self.assertIsInstance(model, EikosDecisionModel)
            names, ids = eikos_label_table(model.tokenizer, 100)
            self.assertEqual(names[:2], ["A", "B"])
            self.assertNotIn("BQ", names)
            self.assertEqual(len(ids), 100)
            row = DecisionInput(
                kind="noul",
                state={"policy": "valid"},
                question="Valid?",
                options=["false", "true"],
            )
            tokens, selected, order = prepare_eikos_row(row, model.tokenizer, max_options=100)
            self.assertEqual(selected, ids[:2])
            self.assertEqual(order, [1, 0])
            self.assertGreater(len(tokens), 0)
            labelled = DecisionInput(
                state="review",
                question="Next?",
                options=["c000", "c001"],
                candidate_ids=["c000", "c001"],
                candidate_labels=["Reject", "Approve"],
                candidate_descriptions=["No action", "Finish work"],
            )
            with patch.object(
                model.tokenizer, "apply_chat_template", wraps=model.tokenizer.apply_chat_template
            ) as render:
                prepare_eikos_row(labelled, model.tokenizer, max_options=100)
            payload = json.loads(render.call_args.args[0][1]["content"])
            self.assertEqual(payload["options"][1]["description"], "c001: Approve — Finish work")
            inputs = [
                row,
                DecisionInput(state="review", question="Next?", options=["approve", "reject"]),
                DecisionInput(
                    kind="score", state="review", question="Rate", options=["0", "1", "2"]
                ),
            ]
            expected = model.batch_decide(inputs)
            self.assertIsInstance(expected[0].decision, bool)
            self.assertIsInstance(expected[2].decision, int)
            output = Path(temp) / "export"
            model.save_pretrained(output)
            restored = DecisionModel.from_pretrained(output)
            self.assertIsInstance(restored, EikosDecisionModel)
            for before, after in zip(expected, restored.batch_decide(inputs)):
                for option in before.probabilities:
                    self.assertAlmostEqual(
                        before.probabilities[option], after.probabilities[option]
                    )
            with self.assertRaisesRegex(ValueError, "at most 100 options"):
                prepare_eikos_row(
                    DecisionInput(
                        state="review", question="Pick", options=[f"option {i}" for i in range(101)]
                    ),
                    model.tokenizer,
                    max_options=100,
                )

    def test_vllm_reads_one_prefill_and_remaps_true_first(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_eikos(source)
            model = DecisionModel.from_eikos(source, max_sequence_length=512)
            output = Path(temp) / "export"
            model.save_pretrained(output)

            class SamplingParams:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            class FakeLLM:
                def generate(self, prompts, params, use_tqdm=False):
                    assert len(prompts) == 1
                    return [
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    logprobs=[
                                        {
                                            token: SimpleNamespace(logprob=math.log(p))
                                            for token, p in zip(
                                                params[0].allowed_token_ids, [0.8, 0.2]
                                            )
                                        }
                                    ]
                                )
                            ]
                        )
                    ]

            with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SamplingParams)}):
                engine = VLLMEngine(output, llm=FakeLLM(), tokenizer=model.tokenizer)
                actual = engine.noul(state="ready")
            self.assertTrue(actual.decision)
            self.assertAlmostEqual(actual.probabilities["true"], 0.8)


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class EikosTrainingTests(unittest.TestCase):
    def test_cli_eikos_import_and_export(self):
        from futurix_decision.cli import main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "native"
            source.mkdir()
            make_eikos(source)
            dataset = root / "rows.jsonl"
            dataset.write_text(
                json.dumps(
                    DecisionExample(
                        id="x",
                        state="review",
                        question="Next?",
                        options=["approve", "reject"],
                        target="approve",
                    ).model_dump()
                )
                + "\n"
            )
            output = root / "export"
            main(
                [
                    "train",
                    "--model",
                    str(source),
                    "--model-format",
                    "eikos",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--method",
                    "sft",
                ]
            )
            self.assertIsInstance(DecisionModel.from_pretrained(output), EikosDecisionModel)

    def test_sft_preserves_the_trained_letter_readout(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_eikos(source)
            model = DecisionModel.from_eikos(source, max_sequence_length=512)
            trainer = DecisionTrainer(
                model=model,
                dataset=[
                    DecisionExample(
                        id="x",
                        state="review",
                        question="Next?",
                        options=["approve", "reject"],
                        target="approve",
                    )
                ],
                sft_config=SFTConfig(method="full", batch_size=1),
            )
            trainer.train(stages=["decision_sft"])
            output = Path(temp) / "trained"
            trainer.export(output)
            self.assertIsInstance(DecisionModel.from_pretrained(output), EikosDecisionModel)


if __name__ == "__main__":
    unittest.main()
