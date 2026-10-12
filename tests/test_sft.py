import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    BitsAndBytesConfig,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from futurix_decision import DecisionModel, DecisionTrainer, SFTConfig, SFTTrainer
from futurix_decision.datasets import DecisionExample
from futurix_decision.models import DecisionBatch
from futurix_decision.trainers import decision_sft_loss, hard_cross_entropy, soft_cross_entropy
from futurix_decision.trainers.sft import _representative_rows


def tiny_model() -> DecisionModel:
    torch.manual_seed(11)
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
    backbone = GPT2LMHeadModel(
        GPT2Config(vocab_size=2, n_positions=256, n_embd=32, n_layer=1, n_head=2)
    )
    return DecisionModel(backbone, tokenizer, decision_tokens=16, max_sequence_length=256)


def tiny_untied_model() -> DecisionModel:
    model = tiny_model()
    backbone = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(model.tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
    )
    return DecisionModel(backbone, model.tokenizer, decision_tokens=16, max_sequence_length=256)


def example(target="approve", state="valid") -> DecisionExample:
    return DecisionExample(
        id=state,
        kind="choice",
        state=state,
        question="Next?",
        options=["approve", "reject"],
        target=target,
    )


class SFTLossTests(unittest.TestCase):
    def test_hard_ce_matches_torch_and_has_gradients(self):
        logits = torch.tensor([[1.0, -0.5], [0.1, 2.0]], requires_grad=True)
        target = torch.tensor([0, 1])
        loss = hard_cross_entropy(logits, target)
        self.assertAlmostEqual(loss.item(), F.cross_entropy(logits, target).item())
        loss.backward()
        self.assertTrue(torch.any(logits.grad != 0))
        with self.assertRaisesRegex(ValueError, "integer"):
            hard_cross_entropy(logits, torch.tensor([0.3, 0.7]))

    def test_soft_ce_and_smoothing(self):
        logits = torch.tensor([[1.0, -0.5]], requires_grad=True)
        target = torch.tensor([[0.7, 0.3]])
        loss = soft_cross_entropy(logits, target, label_smoothing=0.2)
        smoothed = torch.tensor([[0.66, 0.34]])
        expected = -(smoothed * F.log_softmax(logits, dim=-1)).sum()
        self.assertAlmostEqual(loss.item(), expected.item())
        loss.backward()
        self.assertTrue(torch.any(logits.grad != 0))

    def test_masked_decisions_variable_candidates_and_hard_soft_mix(self):
        raw = torch.tensor(
            [[10.0, 1.0, 30.0, -0.3, 20.0], [0.5, 30.0, 1.0, -5.0, -1.0]],
            requires_grad=True,
        )
        masked = raw.masked_fill(
            ~torch.tensor([[False, True, False, True, False], [True, False, True, True, False]]),
            float("-inf"),
        )
        batch = DecisionBatch(masked, [[3, 1], [2, 0, 3]])
        soft = DecisionExample(
            id="2",
            state="other",
            question="Next?",
            options=["a", "b", "c"],
            target=[0.2, 0.7, 0.1],
        )
        hard = example("reject")
        loss = decision_sft_loss(batch, [hard, soft])
        expected_hard = F.cross_entropy(raw[0, [3, 1]].unsqueeze(0), torch.tensor([1]))
        expected_soft = -(
            torch.tensor([0.2, 0.7, 0.1]) * F.log_softmax(raw[1, [2, 0, 3]], dim=-1)
        ).sum()
        self.assertAlmostEqual(loss.item(), ((expected_hard + expected_soft) / 2).item())
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        self.assertEqual(raw.grad[0, 0].item(), 0.0)
        self.assertEqual(raw.grad[1, 1].item(), 0.0)
        self.assertNotEqual(raw.grad[0, 1].item(), 0.0)
        with self.assertRaisesRegex(ValueError, "hard loss"):
            decision_sft_loss(batch, [hard, soft], mode="hard")

    def test_one_hot_hard_and_soft_are_equivalent(self):
        batch = DecisionBatch(torch.tensor([[1.0, -2.0]], requires_grad=True), [[1, 0]])
        hard = example("approve")
        soft = example([1.0, 0.0])
        for smoothing in (0.0, 0.2):
            self.assertAlmostEqual(
                decision_sft_loss(batch, [hard], label_smoothing=smoothing).item(),
                decision_sft_loss(batch, [soft], label_smoothing=smoothing).item(),
            )


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class SFTTrainingTests(unittest.TestCase):
    def test_bounded_training_samples_entire_split_deterministically(self):
        rows = [example("approve", str(i)) for i in range(200)]
        chosen = _representative_rows(rows, count=12, seed=42)
        reversed_chosen = _representative_rows(list(reversed(rows)), count=12, seed=42)
        self.assertEqual({row.id for row in chosen}, {row.id for row in reversed_chosen})
        self.assertEqual([row.id for row in chosen], [row.id for row in reversed_chosen])
        self.assertGreater(max(int(row.id) for row in chosen), 100)

    def test_full_sft_trains_and_roundtrips(self):
        model = tiny_model()
        before = model.backbone.get_output_embeddings().weight.detach().clone()
        config = SFTConfig(
            epochs=2, batch_size=2, gradient_accumulation_steps=2, learning_rate=0.003, seed=7
        )
        trainer = DecisionTrainer(
            model=model,
            dataset=[example("approve", "a"), example("reject", "b")] * 3,
            sft_config=config,
        )
        with self.assertRaisesRegex(ValueError, "calibration_dataset"):
            trainer.train(stages=["decision_sft", "calibration"])
        history = trainer.train(stages=["decision_sft"])
        self.assertTrue(history)
        self.assertTrue(all(math.isfinite(row["loss"]) for row in history))
        self.assertFalse(
            torch.equal(before, model.backbone.get_output_embeddings().weight.detach().cpu())
        )
        with tempfile.TemporaryDirectory() as temp:
            result = model.choice(state="test", choices=["approve", "reject"])
            trainer.save_pretrained(temp)
            loaded = DecisionModel.from_pretrained(temp)
            other = loaded.choice(state="test", choices=["approve", "reject"])
            self.assertEqual(result.decision, other.decision)
            self.assertAlmostEqual(result.confidence, other.confidence, places=5)

    def test_reiterable_stream_and_one_shot_rejection(self):
        def source():
            return (example("approve", str(i)) for i in range(3))

        class Repeatable:
            def __iter__(self):
                return source()

        trainer = SFTTrainer(
            tiny_model(), config=SFTConfig(epochs=2, batch_size=2, learning_rate=0.002)
        )
        self.assertTrue(trainer.train(Repeatable()))
        with self.assertRaisesRegex(ValueError, "one-shot"):
            SFTTrainer(tiny_model(), config=SFTConfig(epochs=2)).train(source())

    def test_jsonl_reopens_for_multiple_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decisions.jsonl"
            path.write_text(json.dumps(example().model_dump()) + "\n", encoding="utf-8")
            trainer = SFTTrainer(
                tiny_model(), config=SFTConfig(epochs=2, batch_size=1, learning_rate=0.002)
            )
            self.assertEqual(len(trainer.train(path)), 2)

    def test_empty_data_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            SFTTrainer(tiny_model()).train([])

    def test_gradient_checkpointing_training(self):
        model = tiny_model()
        trainer = SFTTrainer(
            model,
            config=SFTConfig(
                gradient_checkpointing=True,
                epochs=1,
                batch_size=2,
                learning_rate=0.002,
            ),
        )
        self.assertTrue(trainer.train([example("approve", "a"), example("reject", "b")]))
        self.assertTrue(model.backbone.config.use_cache)

    @unittest.skipUnless(importlib.util.find_spec("peft"), "PEFT not installed")
    def test_lora_merges_to_standalone_checkpoint(self):
        model = tiny_model()
        original_decision_rows = (
            model.backbone.get_input_embeddings().weight[model.readout.token_ids].detach().clone()
        )
        trainer = SFTTrainer(
            model,
            config=SFTConfig(
                method="lora",
                epochs=1,
                batch_size=2,
                learning_rate=0.003,
                lora_rank=2,
                lora_alpha=4,
            ),
        )
        trainer.train([example("approve", "a"), example("reject", "b")])
        with tempfile.TemporaryDirectory() as temp:
            result = model.choice(state="test", choices=["approve", "reject"])
            trainer.save_pretrained(temp)
            self.assertTrue((Path(temp) / "config.json").exists())
            self.assertFalse((Path(temp) / "adapter_config.json").exists())
            merged_decision_rows = (
                model.backbone.get_input_embeddings().weight[model.readout.token_ids].detach().cpu()
            )
            self.assertFalse(torch.equal(original_decision_rows, merged_decision_rows))
            loaded = DecisionModel.from_pretrained(temp)
            self.assertAlmostEqual(
                result.confidence,
                loaded.choice(state="test", choices=["approve", "reject"]).confidence,
                places=5,
            )

    def test_qlora_requires_4_bit_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "4-bit"):
            SFTTrainer(tiny_model(), config=SFTConfig(method="qlora")).train([example()])

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes"),
        "4-bit CUDA backend not installed",
    )
    def test_qlora_adapter_saves_and_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            adapter = Path(temp) / "adapter"
            tiny_untied_model().save_pretrained(base)
            model = DecisionModel.from_pretrained(
                base,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                device_map="cuda:0",
            )
            self.assertTrue(model.backbone.is_loaded_in_4bit)
            trainer = SFTTrainer(
                model,
                config=SFTConfig(
                    method="qlora",
                    precision="bf16" if os.environ.get("FUTURIX_TEST_BF16") == "1" else "no",
                    lora_rank=2,
                    lora_alpha=4,
                    batch_size=2,
                    learning_rate=0.003,
                    gradient_checkpointing=True,
                ),
            )
            trainer.train([example("approve", "a"), example("reject", "b")])
            trainer.save_pretrained(adapter)
            self.assertTrue((adapter / "adapter_config.json").is_file())
            self.assertFalse((adapter / "config.json").is_file())
            loaded = DecisionModel.from_pretrained(adapter)
            result = loaded.choice(state="test", choices=["approve", "reject"])
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
            self.assertEqual(loaded.config.decision_tokens, 16)


if __name__ == "__main__":
    unittest.main()
