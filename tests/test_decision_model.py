import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from futurix_decision import DecisionInput, DecisionModel
from futurix_decision.models.decision_head import (
    assign_slots,
    decision_token,
    mask_unavailable_slots,
)


def tiny_checkpoint() -> DecisionModel:
    torch.manual_seed(7)
    raw_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    )
    backbone = GPT2LMHeadModel(
        GPT2Config(vocab_size=2, n_positions=256, n_embd=32, n_layer=1, n_head=2)
    )
    return DecisionModel(backbone, tokenizer, decision_tokens=16, max_sequence_length=256)


class DecisionHeadTests(unittest.TestCase):
    def test_slot_mapping_randomization_is_reproducible(self):
        options = ["approve", "reject", "escalate"]
        first = assign_slots(options, total_slots=16, seed=11)
        self.assertEqual(first, assign_slots(options, total_slots=16, seed=11))
        self.assertEqual(len(set(first)), 3)
        self.assertNotEqual(first, assign_slots(options, total_slots=16, seed=12))
        self.assertEqual(decision_token(4), "<|decision_004|>")

    def test_masking_and_gradients(self):
        logits = torch.arange(12.0).view(2, 6).requires_grad_()
        masked = mask_unavailable_slots(logits, [[1, 3], [2, 4]])
        self.assertTrue(torch.isneginf(masked[0, 0]))
        self.assertEqual(masked[0, 1].item(), 1.0)
        self.assertEqual(masked[1, 4].item(), 10.0)
        masked.log_softmax(dim=-1)[0, 1].backward()
        self.assertEqual(logits.grad[0, 0].item(), 0.0)
        self.assertNotEqual(logits.grad[0, 1].item(), 0.0)


class DecisionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_checkpoint()

    def test_tokens_are_single_vocabulary_ids(self):
        self.assertEqual(
            len({self.model.tokenizer.convert_tokens_to_ids(decision_token(i)) for i in range(16)}),
            16,
        )
        self.assertEqual(self.model.backbone.get_input_embeddings().num_embeddings, len(self.model.tokenizer))

    def test_new_tokens_initialize_padded_base_rows_independently(self):
        raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]"
        )
        backbone = GPT2LMHeadModel(
            GPT2Config(vocab_size=32, n_positions=256, n_embd=32, n_layer=1, n_head=2)
        )
        with torch.no_grad():
            backbone.get_input_embeddings().weight[2:18].fill_(0.25)
        model = DecisionModel(backbone, tokenizer, decision_tokens=16, max_sequence_length=256)
        weight = model.backbone.get_output_embeddings().weight.detach()
        self.assertGreater((weight[model._token_ids[0]] - weight[model._token_ids[1]]).abs().sum().item(), 0)

    def test_export_preserves_original_model_provenance(self):
        model = tiny_checkpoint()
        model.source_checkpoint = "Example/Source"
        model.source_revision = "a" * 40
        model.base_model_revision = "b" * 40
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored = DecisionModel.from_pretrained(path)
            self.assertEqual(restored.source_checkpoint, "Example/Source")
            self.assertEqual(restored.source_revision, "a" * 40)
            self.assertEqual(restored.base_model_revision, "b" * 40)

    def test_choice_and_abstention(self):
        result = self.model.choice(state={"document": "expired"}, choices=["approve", "reject"])
        self.assertIn(result.decision, ("approve", "reject"))
        self.assertEqual(set(result.probabilities), {"approve", "reject"})
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
        self.assertGreaterEqual(result.latency_ms, 0)
        abstained = self.model.choice(
            state="expired", choices=["approve", "reject"], confidence_threshold=1.0
        )
        self.assertTrue(abstained.abstained)
        self.assertIsNone(abstained.decision)
        self.assertLess(abstained.confidence, 1.0)
        self.assertTrue(abstained.to_dict()["abstained"])

    def test_noul_score_and_threshold_validation(self):
        self.assertIsInstance(self.model.noul(state="ready").decision, bool)
        score = self.model.score(state="ready")
        self.assertIsInstance(score.decision, int)
        self.assertEqual(set(score.probabilities), {str(i) for i in range(6)})
        with self.assertRaises(ValueError):
            self.model.score(state="ready", min_score=0, max_score=20)
        with self.assertRaises(ValueError):
            self.model.noul(state="ready", confidence_threshold=1.1)

    def test_one_forward_per_batch_and_seed_changes_mapping(self):
        rows = [
            {"state": {"a": 1}, "choices": ["approve", "reject"]},
            {"state": "second", "choices": ["approve", "reject", "escalate"]},
            {"state": "third", "choices": ["continue", "stop"]},
        ]
        with patch.object(self.model.backbone, "forward", wraps=self.model.backbone.forward) as forward:
            results = self.model.batch_choice(rows)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
        example = DecisionInput(state="ready", question="Next?", options=["yes", "no"])
        first = self.model.forward_decisions([example], assignment_seed=10)
        again = self.model.forward_decisions([example], assignment_seed=10)
        second = self.model.forward_decisions([example], assignment_seed=11)
        self.assertEqual(first.slots, again.slots)
        self.assertNotEqual(first.slots, second.slots)
        self.assertEqual(torch.isfinite(first.logits).sum().item(), 2)

    def test_temperature_changes_probability_sharpness(self):
        example = dict(state="ready", choices=["a", "b", "c"])
        old_temperature = self.model.temperature
        try:
            self.model.temperature = 1
            regular = self.model.choice(**example)
            self.model.temperature = 100
            flattened = self.model.choice(**example)
        finally:
            self.model.temperature = old_temperature
        uniform = 1 / 3
        self.assertLess(
            max(abs(value - uniform) for value in flattened.probabilities.values()),
            max(abs(value - uniform) for value in regular.probabilities.values()),
        )

    def test_save_load_preserves_mapping_logits_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            original = self.model.choice(state="ready", choices=["yes", "no"])
            self.model.save_pretrained(temp)
            config = json.loads((Path(temp) / "decision_config.json").read_text())
            self.assertEqual(config["decision_tokens"], 16)
            self.assertTrue(list(Path(temp).glob("*.safetensors")))
            reloaded = DecisionModel.from_pretrained(temp)
            reloaded_result = reloaded.choice(state="ready", choices=["yes", "no"])
            self.assertEqual(reloaded_result.decision, original.decision)
            for name in ("yes", "no"):
                self.assertTrue(
                    math.isclose(
                        reloaded_result.probabilities[name],
                        original.probabilities[name],
                        abs_tol=1e-6,
                    )
                )

    def test_no_silent_truncation(self):
        old_maximum = self.model.max_sequence_length
        try:
            self.model.max_sequence_length = 5
            with self.assertRaisesRegex(ValueError, "max_sequence_length"):
                self.model.choice(state="too long", choices=["a", "b"])
        finally:
            self.model.max_sequence_length = old_maximum


if __name__ == "__main__":
    unittest.main()
