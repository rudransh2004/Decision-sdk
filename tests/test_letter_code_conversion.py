"""Fixed letter-code conversion tests without downloading released weights."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from futurix_decision import (
    DecisionInput,
    DecisionTrainer,
    RLCDConfig,
    SFTConfig,
    evaluate,
    load_model,
)
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.letter_code import answer_codes


def raw_checkpoint(path: Path) -> None:
    words = ["[UNK]", "[EOS]", "approve", "reject", "escalate", "ticket"]
    words.extend(chr(code) for code in range(ord("A"), ord("Z") + 1))
    tokenizer = Tokenizer(
        models.WordLevel(
            {word: index for index, word in enumerate(words)},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    torch.manual_seed(19)
    backbone = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(wrapped),
            n_positions=512,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=wrapped.eos_token_id,
            eos_token_id=wrapped.eos_token_id,
        )
    )
    backbone.save_pretrained(path)
    wrapped.save_pretrained(path)


class LetterCodeConversionTests(unittest.TestCase):
    def test_raw_conversion_initializes_readout_from_lm_head_and_masks_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            original = GPT2LMHeadModel.from_pretrained(base)
            model = load_model(
                base,
                convert=True,
                strategy="letter_code",
                decision_tokens=8,
                max_sequence_length=512,
            )
            self.assertEqual(model.head_type, "letter_code")
            self.assertEqual(model.head_prompt_version, "letter-code-v1")
            self.assertEqual(model.attention_mode, "causal")
            self.assertNotIn("<|decision_000|>", model.tokenizer.get_vocab())
            self.assertEqual(len(model.tokenizer), original.config.vocab_size)
            ids = model.head_config["token_ids"]
            self.assertTrue(
                torch.equal(
                    model.decision_head.readout.weight,
                    original.lm_head.weight[ids],
                )
            )
            rows = [
                DecisionInput(state="ticket", question="Route?", options=["approve", "reject"]),
                DecisionInput(
                    state="ticket", question="Route?", options=["approve", "reject", "escalate"]
                ),
            ]
            batch = model.forward_decisions(rows)
            self.assertEqual(batch.slots, [[0, 1], [0, 1, 2]])
            self.assertEqual(batch.logits.shape, (2, 8))
            self.assertTrue(torch.isneginf(batch.logits[0, 2:]).all())
            for result in model.batch_decide(rows):
                self.assertAlmostEqual(sum(result.probabilities.values()), 1, places=6)
            with tempfile.TemporaryDirectory() as output:
                model.save_pretrained(output)
                saved = json.loads((Path(output) / "decision_config.json").read_text())
                self.assertEqual(saved["head_type"], "letter_code")
                self.assertEqual(saved["head_prompt_version"], "letter-code-v1")
                restored = load_model(output)
                expected = model.batch_decide(rows)
                for before, after in zip(expected, restored.batch_decide(rows)):
                    for option in before.probabilities:
                        self.assertAlmostEqual(
                            before.probabilities[option],
                            after.probabilities[option],
                            places=6,
                        )

    def test_pplx_names_are_rejected_with_a_migration_hint(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            with self.assertRaisesRegex(ConfigError, "strategy must be"):
                load_model(base, convert=True, strategy="pplx")
            model = load_model(
                base,
                convert=True,
                strategy="letter_code",
                decision_tokens=8,
                max_sequence_length=512,
            )
            exported = Path(temp) / "legacy"
            model.save_pretrained(exported)
            metadata_path = exported / "decision_config.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["head_type"] = "pplx"
            metadata["head_prompt_version"] = "pplx-v1"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(CheckpointError, "set head_type to 'letter_code'"):
                load_model(exported)

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
    def test_sft_then_rlcd_trains_and_exports_reloadable_readout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            model = load_model(
                base,
                convert=True,
                strategy="letter_code",
                decision_tokens=8,
                max_sequence_length=512,
                train_backbone=False,
            )
            rows = [
                {
                    "id": "a",
                    "state": "ticket",
                    "question": "Route?",
                    "options": ["approve", "reject"],
                    "target": [0.8, 0.2],
                },
                {
                    "id": "b",
                    "state": "ticket",
                    "question": "Route?",
                    "options": ["approve", "reject"],
                    "target": "reject",
                },
            ]
            trainer = DecisionTrainer(
                model=model,
                dataset=rows,
                calibration_dataset=[
                    {
                        "id": "heldout",
                        "state": "review",
                        "question": "Route?",
                        "options": ["approve", "reject"],
                        "target": [0.6, 0.4],
                    }
                ],
                sft_config=SFTConfig(method="full", epochs=1, batch_size=2),
                rlcd_config=RLCDConfig(
                    method="full",
                    epochs=1,
                    batch_size=2,
                    group_size=2,
                    kl_beta=0.01,
                ),
            )
            history = trainer.train(stages=["decision_sft", "calibration", "rlcd"])
            self.assertTrue(history["decision_sft"])
            self.assertTrue(history["calibration"])
            self.assertTrue(history["rlcd"])
            output = Path(temp) / "trained"
            trainer.export(output)
            restored = load_model(output)
            self.assertEqual(restored.head_type, "letter_code")
            self.assertEqual(evaluate(restored, rows)["count"], 2)

    @unittest.skipUnless(
        importlib.util.find_spec("peft") and importlib.util.find_spec("accelerate"),
        "LoRA training extras not installed",
    )
    def test_lora_then_rlcd_merges_with_fixed_readout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            model = load_model(
                base,
                convert=True,
                strategy="letter_code",
                decision_tokens=8,
                max_sequence_length=512,
            )
            examples = [
                {
                    "id": "a",
                    "state": "ticket",
                    "question": "Route?",
                    "options": ["approve", "reject"],
                    "target": "approve",
                },
            ]
            trainer = DecisionTrainer(
                model=model,
                dataset=examples,
                sft_config=SFTConfig(method="lora", batch_size=1, lora_rank=2, lora_alpha=4),
                rlcd_config=RLCDConfig(method="lora", batch_size=1, group_size=2, kl_beta=0.0),
            )
            trainer.train(stages=["decision_sft", "rlcd"])
            output = Path(temp) / "merged"
            trainer.export(output)
            restored = load_model(output)
            self.assertEqual(restored.head_type, "letter_code")
            self.assertEqual(evaluate(restored, examples)["count"], 1)

    @unittest.skipUnless(
        int(version("transformers").split(".")[0]) >= 5,
        "Qwen3.5 architecture requires Transformers 5.17+",
    )
    def test_small_qwen_multimodal_text_noncausal_roundtrip(self):
        from transformers import (
            Qwen3_5Config,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5TextConfig,
            Qwen3_5VisionConfig,
        )

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(base)
            text = Qwen3_5TextConfig(
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                layer_types=["linear_attention", "full_attention"],
                max_position_embeddings=512,
            )
            vision = Qwen3_5VisionConfig(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=4,
                out_hidden_size=32,
            )
            Qwen3_5ForConditionalGeneration(
                Qwen3_5Config(
                    text_config=text,
                    vision_config=vision,
                )
            ).save_pretrained(base)
            model = load_model(
                base,
                convert=True,
                strategy="letter_code",
                decision_tokens=8,
                max_sequence_length=512,
            )
            self.assertEqual(model.attention_mode, "noncausal_full_attention")
            self.assertTrue(model.multimodal_loader)
            before = model.choice(state="ticket", question="Route?", choices=["approve", "reject"])
            self.assertAlmostEqual(sum(before.probabilities.values()), 1, places=6)
            if importlib.util.find_spec("accelerate"):
                method = "lora" if importlib.util.find_spec("peft") else "full"
                trainer = DecisionTrainer(
                    model=model,
                    dataset=[
                        {
                            "id": "q1",
                            "state": "ticket",
                            "question": "Route?",
                            "options": ["approve", "reject"],
                            "target": "approve",
                        },
                    ],
                    sft_config=SFTConfig(method=method, batch_size=1, lora_rank=2, lora_alpha=4),
                    rlcd_config=RLCDConfig(
                        method=method,
                        batch_size=1,
                        group_size=2,
                        kl_beta=0.0,
                    ),
                )
                history = trainer.train(stages=["decision_sft", "rlcd"])
                self.assertTrue(history["decision_sft"])
                self.assertTrue(history["rlcd"])
            trained = model.choice(state="ticket", question="Route?", choices=["approve", "reject"])
            output = Path(temp) / "export"
            model.save_pretrained(output)
            after = load_model(output).choice(
                state="ticket",
                question="Route?",
                choices=["approve", "reject"],
            )
            for option in trained.probabilities:
                self.assertAlmostEqual(
                    trained.probabilities[option],
                    after.probabilities[option],
                    places=5,
                )

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
    def test_cli_converts_and_trains_before_export(self):
        from futurix_decision.cli import main

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            dataset = Path(temp) / "decisions.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "state": "ticket",
                        "question": "Route?",
                        "options": ["approve", "reject"],
                        "target": "approve",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for strategy in ("letter_code",):
                with self.subTest(strategy=strategy):
                    output = Path(temp) / strategy
                    with redirect_stdout(io.StringIO()):
                        main(
                            [
                                "train",
                                "--model",
                                str(base),
                                "--dataset",
                                str(dataset),
                                "--output",
                                str(output),
                                "--convert-strategy",
                                strategy,
                                "--decision-tokens",
                                "8",
                                "--finetune-method",
                                "full",
                            ]
                        )
                    self.assertEqual(load_model(output).head_type, "letter_code")

    def test_qwen3_5_noncausal_routing_is_explicit_and_mimo_fails_closed(self):
        config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            is_encoder_decoder=False,
            vision_config=object(),
        )
        with (
            patch("futurix_decision.utils.config.load_decision_config", return_value={}),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=config,
            ),
            patch("futurix_decision.models.native._requires_transformers"),
            patch(
                "futurix_decision.models.decision_model.DecisionModel.from_pretrained",
                return_value="prepared",
            ) as loader,
        ):
            self.assertEqual(
                load_model("Qwen/Qwen3.8-27B", convert=True, strategy="letter_code"),
                "prepared",
            )
            loader.assert_called_once_with(
                "Qwen/Qwen3.8-27B",
                head_type="letter_code",
                head_prompt_version="letter-code-v1",
                decision_tokens=255,
                attention_mode="noncausal_full_attention",
                multimodal_loader=True,
                train_backbone=True,
            )
            config.model_type = "mimo_v2"
            config.architectures = ["MiMoV2ForConditionalGeneration"]
            with self.assertRaisesRegex(CheckpointError, "multimodal"):
                load_model("XiaomiMiMo/MiMo-V2.6-Flash-RL", convert=True, strategy="letter_code")
        with self.assertRaisesRegex(ValueError, "known decision-model family"):
            load_model("perplexity-ai/pplx-decider-v1.1-27b", convert=True, strategy="letter_code")

    def test_codebook_and_attention_fail_before_large_model_loading(self):
        raw = Tokenizer(models.WordLevel({"[UNK]": 0, "A": 1}, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
        with self.assertRaisesRegex(ValueError, "only 1 distinct"):
            answer_codes(tokenizer, 4)
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            raw_checkpoint(base)
            with self.assertRaisesRegex(ValueError, "Qwen3.5-family"):
                load_model(
                    base,
                    convert=True,
                    strategy="letter_code",
                    attention_mode="noncausal_full_attention",
                )


if __name__ == "__main__":
    unittest.main()
