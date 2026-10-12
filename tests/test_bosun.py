import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from futurix_decision import DecisionInput, DecisionModel, load_model
from futurix_decision.datasets import DecisionExample
from futurix_decision.models.bosun import BOSUN_PROMPT_SCHEMA, format_bosun_prompt
from futurix_decision.models.families.bosun import BOSUN_RELEASES
from futurix_decision.models.registry import family_for_checkpoint
from futurix_decision.trainers import SFTConfig, SFTTrainer
from test_decision_model import tiny_checkpoint


def bosun_tiny_model():
    base = tiny_checkpoint()
    base.tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return DecisionModel(
        base.backbone,
        base.tokenizer,
        decision_tokens=256,
        max_sequence_length=256,
        prompt_format="bosun-v3",
    )


class BosunContractTests(unittest.TestCase):
    def test_unified_loader_preserves_both_released_decision_models(self):
        for checkpoint, revision in BOSUN_RELEASES.items():
            with self.subTest(checkpoint=checkpoint):
                tiny = bosun_tiny_model()
                package = SimpleNamespace(
                    model=tiny.backbone,
                    tokenizer=tiny.tokenizer,
                    config=SimpleNamespace(
                        prompt_schema=BOSUN_PROMPT_SCHEMA,
                        decision_token_count=256,
                        null_token_index=255,
                        base_model_name_or_path="Qwen/Qwen3-0.6B",
                        base_model_revision="a" * 40,
                    ),
                )
                with patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=package,
                ) as loader:
                    model = load_model(checkpoint)
                loader.assert_called_once_with(
                    checkpoint, trust_remote_code=True, revision=revision
                )
                self.assertEqual(model.config.prompt_format, "bosun-v3")
                self.assertEqual(model.config.source_checkpoint, checkpoint)
                self.assertEqual(model.config.source_revision, revision)
                adapter = family_for_checkpoint(checkpoint)
                self.assertEqual((adapter.name, adapter.trainable), ("bosun", True))
                with self.assertRaisesRegex(ValueError, "already contains a trained decision head"):
                    load_model(checkpoint, convert=True)

    @unittest.skipUnless(os.environ.get("FUTURIX_LIVE_BOSUN") == "1", "opt-in model integration")
    def test_real_bosun_native_predict_parity(self):
        checkpoint = "Hanno-Labs/bosun-v3.1-1.7b"
        native = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            revision="1d8dc82a20e4a32ed60927a47272d6efff48eed2",
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        choices = [
            {"id": "billing", "label": "Billing", "description": "Payment issue"},
            {"id": "access", "label": "Account access", "description": "Login issue"},
        ]
        expected = native.predict(
            state={"request": "Customer cannot access their account."},
            instructions="Choose the best support route.",
            row_id="example-1",
            candidates=choices,
        )
        model = DecisionModel(native.model, native.tokenizer, prompt_format="bosun-v3")
        row = DecisionExample(
            id="example-1",
            state={"request": "Customer cannot access their account."},
            question="Choose the best support route.",
            options=["billing", "access"],
            target="access",
            candidate_labels=["Billing", "Account access"],
            candidate_descriptions=["Payment issue", "Login issue"],
        )
        prompt, slots = model.readout.prompt(row, None)
        self.assertEqual(prompt, expected["prompt"])
        actual = model.forward_decisions([row]).logits[0, slots]
        probabilities = torch.softmax(actual.float(), dim=-1).tolist()
        for ours, theirs in zip(probabilities, expected["probabilities"]):
            self.assertAlmostEqual(ours, theirs, places=6)

    def test_stable_slots_chat_prompt_and_candidate_metadata(self):
        model = bosun_tiny_model()
        example = DecisionExample(
            id="row-17",
            kind="choice",
            state={"ticket": "charge"},
            question="Route it",
            options=["billing", "access", "other"],
            target=[0.8, 0.1, 0.1],
            candidate_labels=["Billing", "Account access", "Other"],
            candidate_descriptions=["Payment issue", "Login issue", "Escalation"],
        )
        prompt, slots = model.readout.prompt(example, None)
        self.assertEqual(prompt, model.readout.prompt(example, None)[0])
        self.assertEqual(sorted(slots), [0, 1, 2])
        self.assertTrue(prompt.startswith("system:Choose exactly one supplied decision token."))
        content = json.loads(prompt.split("user:", 1)[1].split("\nassistant:", 1)[0])
        self.assertEqual(content["schema"], BOSUN_PROMPT_SCHEMA)
        self.assertEqual(content["question"]["instructions"], "Route it")
        fields = content["question"]["criteria_fields"]
        self.assertEqual(fields, ["t", "o", "n", "d"])
        for row in content["question"]["criteria"]:
            mapped = dict(zip(fields, row))
            self.assertEqual(mapped["t"], f"<|decision_{slots[mapped['o']]:03d}|>")
            self.assertEqual(mapped["n"], example.candidate_labels[mapped["o"]])
            self.assertEqual(mapped["d"], example.candidate_descriptions[mapped["o"]])
        self.assertNotEqual(
            model.readout.prompt(example, 20)[1], model.readout.prompt(example, 21)[1]
        )
        self.assertEqual(model.tokenizer.padding_side, "left")

    def test_structured_descriptions_compact_as_bosun_does(self):
        model = bosun_tiny_model()
        example = DecisionInput(
            state="record",
            question="Select",
            options=["a", "b"],
            candidate_descriptions=['{"z":2,"a":"one"}', '{"z":4,"a":"two"}'],
        )
        prompt, _ = format_bosun_prompt(example, model.tokenizer)
        content = json.loads(prompt.split("user:", 1)[1].split("\nassistant:", 1)[0])
        self.assertEqual(content["question"]["criteria_fields"], ["t", "o", "n", "a", "z"])

    def test_masked_inference_and_export_preserve_contract(self):
        model = bosun_tiny_model()
        result = model.choice(state="ready", choices=["a", "b"], question="Choose")
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
        self.assertEqual(len(result.probabilities), 2)
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored = DecisionModel.from_pretrained(path)
            self.assertEqual(restored.config.prompt_format, "bosun-v3")
            self.assertEqual(restored.tokenizer.padding_side, "left")
            other = restored.choice(state="ready", choices=["a", "b"], question="Choose")
            for name in result.probabilities:
                self.assertAlmostEqual(
                    result.probabilities[name], other.probabilities[name], places=5
                )

    def test_typed_system_one_preserves_bosun_probabilities(self):
        model = bosun_tiny_model()
        questions = {
            "route": {
                "type": "choice",
                "instructions": "Route this ticket",
                "criteria": {"billing": "Payments", "technical": "App errors"},
            },
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "priority": {
                "type": "score",
                "instructions": "Priority",
                "criteria": ["Low", "Medium", "High"],
            },
        }
        answers = model.system_one({"ticket": "Duplicate charge"}, questions)["answers"]
        self.assertEqual(list(answers), list(questions))
        for answer in answers.values():
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1, places=5)
        self.assertEqual(answers["urgent"]["noul"], answers["urgent"]["probabilities"]["true"])
        self.assertGreaterEqual(answers["priority"]["score"], 0)
        self.assertLessEqual(answers["priority"]["score"], 2)
        with self.assertRaisesRegex(ValueError, "noul criteria"):
            model.system_one("record", {"bad": {"type": "noul", "criteria": {"yes": "y"}}})

    def test_null_slot_reserved_and_metadata_must_align(self):
        model = bosun_tiny_model()
        with self.assertRaisesRegex(ValueError, "reserves slot 255"):
            model.readout.prompt(
                DecisionInput(state="s", question="q", options=[str(i) for i in range(256)]),
                None,
            )
        with self.assertRaisesRegex(ValueError, "candidate_labels"):
            DecisionInput(state="s", question="q", options=["a", "b"], candidate_labels=["A"])

    @unittest.skipUnless(importlib.util.find_spec("peft"), "PEFT not installed")
    def test_continuation_preserves_rows_and_trains_untied_output(self):
        from peft import LoraConfig, TaskType, get_peft_model

        from test_sft import tiny_untied_model

        base = tiny_untied_model()
        base.tokenizer.chat_template = bosun_tiny_model().tokenizer.chat_template
        model = DecisionModel(
            base.backbone,
            base.tokenizer,
            decision_tokens=256,
            prompt_format="bosun-v3",
            max_sequence_length=256,
        )
        rows_before = (
            model.backbone.get_input_embeddings().weight[model.readout.token_ids].detach().clone()
        )
        head_before = (
            model.backbone.get_output_embeddings().weight[model.readout.token_ids].detach().clone()
        )
        model.backbone = get_peft_model(
            model.backbone,
            LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules=["q_proj"]),
        )
        example = DecisionExample(
            id="continuation",
            state="ready",
            question="Choose",
            options=["a", "b"],
            target=[0.1, 0.9],
        )
        trainer = SFTTrainer(
            model,
            config=SFTConfig(
                method="lora",
                batch_size=1,
                learning_rate=0.01,
                lora_rank=2,
                lora_alpha=4,
            ),
        )
        trainer.train([example, example])
        self.assertTrue(model._bosun_continuation_prepared)
        input_after = (
            model.backbone.get_input_embeddings().weight[model.readout.token_ids].detach().cpu()
        )
        output_after = (
            model.backbone.get_output_embeddings().weight[model.readout.token_ids].detach().cpu()
        )
        self.assertGreater((input_after - rows_before.cpu()).abs().sum().item(), 0)
        self.assertGreater((output_after - head_before.cpu()).abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored = DecisionModel.from_pretrained(path)
            self.assertEqual(restored.config.prompt_format, "bosun-v3")
            self.assertAlmostEqual(
                sum(restored.decide(example).probabilities.values()), 1.0, places=6
            )


if __name__ == "__main__":
    unittest.main()
