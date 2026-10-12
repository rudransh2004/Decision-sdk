"""Candidate-aware decision heads and the complete checkpoint lifecycle."""

import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from futurix_decision import (
    DecisionInput,
    DecisionModel,
    DecisionTrainer,
    SFTConfig,
    evaluate,
    load_model,
    register_decision_head,
)
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.bosun import BOSUN_PROMPT_SCHEMA
from futurix_decision.models.custom_head import HEAD_WEIGHTS
from futurix_decision.trainers import RLCDConfig


def make_model(*, head_type="pointer", head_config=None, base_checkpoint=None):
    torch.manual_seed(17)
    vocab = {
        "[UNK]": 0,
        "[EOS]": 1,
        "approve": 2,
        "reject": 3,
        "escalate": 4,
        "accept": 5,
        "true": 6,
        "false": 7,
    }
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
    backbone = GPT2LMHeadModel(
        GPT2Config(vocab_size=len(vocab), n_positions=256, n_embd=32, n_layer=1, n_head=2)
    )
    return DecisionModel(
        backbone,
        tokenizer,
        head_type=head_type,
        head_config=head_config,
        decision_tokens=16,
        max_sequence_length=256,
        base_checkpoint=base_checkpoint,
    )


def row(target="approve", state="review"):
    from futurix_decision.datasets import DecisionExample

    return DecisionExample(
        id=state,
        kind="choice",
        state=state,
        question="Next?",
        options=["approve", "reject", "escalate"],
        target=target,
    )


class CustomHeadTests(unittest.TestCase):
    def test_custom_head_is_candidate_aware_and_does_not_add_decision_tokens(self):
        model = make_model()
        self.assertNotIn("<|decision_000|>", model.tokenizer.get_vocab())
        self.assertEqual(model.readout.token_ids, [])
        inputs = [
            DecisionInput(state="first", question="Choose", options=["approve", "reject"]),
            DecisionInput(
                state="second", question="Choose", options=["reject", "approve", "escalate"]
            ),
        ]
        base = model.backbone.base_model
        with patch.object(base, "forward", wraps=base.forward) as forward:
            batch = model.forward_decisions(inputs, assignment_seed=13)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(batch.logits.shape, (2, 3))
        self.assertTrue(torch.isneginf(batch.logits[0, 2]))
        self.assertEqual(
            batch.slots[0], model.forward_decisions(inputs, assignment_seed=13).slots[0]
        )
        self.assertEqual(len(set(batch.slots[1])), 3)
        loss = -torch.log_softmax(batch.logits[0, batch.slots[0]], dim=-1)[0]
        loss.backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.decision_head.parameters()
            )
        )

    def test_inference_evaluation_and_safe_roundtrip(self):
        model = make_model(head_config={"width": 16})
        inputs = [
            DecisionInput(
                state={"source": "billing"}, question="Choose", options=["approve", "reject"]
            ),
            DecisionInput(
                state="second", question="Choose", options=["approve", "reject", "escalate"]
            ),
        ]
        results = model.batch_decide(inputs)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(sum(results[1].probabilities.values()), 1.0, places=6)
        self.assertIsInstance(model.noul(state="ready").decision, bool)
        self.assertIsInstance(model.score(state="ready").decision, int)
        self.assertEqual(evaluate(model, [row()], batch_size=1)["count"], 1)
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            config = json.loads((Path(temp) / "decision_config.json").read_text())
            self.assertEqual(config["head_type"], "pointer")
            self.assertEqual(config["head_config"], {"width": 16})
            self.assertTrue((Path(temp) / HEAD_WEIGHTS).is_file())
            restored = DecisionModel.from_pretrained(temp)
            for original, other in zip(results, restored.batch_decide(inputs), strict=True):
                for option in original.probabilities:
                    self.assertAlmostEqual(
                        original.probabilities[option], other.probabilities[option], places=6
                    )
            with self.assertRaisesRegex(ConfigError, "cannot override"):
                DecisionModel.from_pretrained(temp, head_type="tokens")
            (Path(temp) / HEAD_WEIGHTS).unlink()
            with self.assertRaisesRegex(CheckpointError, "missing decision_head.safetensors"):
                DecisionModel.from_pretrained(temp)

    def test_unregistered_heads_and_bosun_prompts_are_rejected(self):
        model = make_model()
        with self.assertRaisesRegex(ConfigError, "unknown decision head"):
            make_model(head_type="unregistered")
        with self.assertRaisesRegex(ConfigError, "require prompt_format"):
            DecisionModel(
                model.backbone, model.tokenizer, head_type="pointer", prompt_format="bosun-v3"
            )

    def test_registered_custom_head_reconstructs_without_dynamic_imports(self):
        class AdditiveHead(torch.nn.Module):
            def __init__(self, hidden_size, width=8):
                super().__init__()
                self.projection = torch.nn.Linear(hidden_size, width)
                self.score = torch.nn.Linear(width, 1)

            def forward(self, state, options, available):
                return self.score(
                    torch.tanh(self.projection(options + state.unsqueeze(1)))
                ).squeeze(-1)

        register_decision_head("test_additive", AdditiveHead)
        with self.assertRaisesRegex(ConfigError, "already registered"):
            register_decision_head("test_additive", AdditiveHead)
        model = make_model(head_type="test_additive", head_config={"width": 4})
        with tempfile.TemporaryDirectory() as temp:
            expected = model.choice(state="review", choices=["approve", "reject"])
            model.save_pretrained(temp)
            restored = DecisionModel.from_pretrained(temp)
            actual = restored.choice(state="review", choices=["approve", "reject"])
            self.assertEqual(type(restored.decision_head), AdditiveHead)
            for name in expected.probabilities:
                self.assertAlmostEqual(
                    expected.probabilities[name], actual.probabilities[name], places=6
                )

    def test_vllm_rejects_custom_head_before_initializing_engine(self):
        from futurix_decision.vllm.engine import VLLMEngine

        with tempfile.TemporaryDirectory() as temp:
            make_model().save_pretrained(temp)
            with self.assertRaisesRegex(ConfigError, "does not support custom"):
                VLLMEngine(temp)


class GeneralConversionTests(unittest.TestCase):
    def _raw_checkpoint(self, directory: Path, family: str) -> None:
        vocab = {
            "[UNK]": 0,
            "[EOS]": 1,
            "approve": 2,
            "reject": 3,
            "ticket": 4,
            "yes": 5,
            "no": 6,
        }
        raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]"
        )
        if family == "gpt2":
            backbone = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(tokenizer),
                    n_positions=256,
                    n_embd=32,
                    n_layer=1,
                    n_head=2,
                )
            )
        else:
            backbone = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=256,
                )
            )
        backbone.save_pretrained(directory)
        tokenizer.save_pretrained(directory)

    def test_text_causal_families_convert_without_resizing_and_roundtrip(self):
        for family in ("gpt2", "llama"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                base = Path(temp) / "base"
                base.mkdir()
                self._raw_checkpoint(base, family)
                model = load_model(base, convert=True, decision_tokens=16, max_sequence_length=256)
                self.assertEqual(model.config.head_type, "pointer")
                self.assertEqual(model.config.head_prompt_version, "span-v1")
                self.assertEqual(
                    model.backbone.get_input_embeddings().num_embeddings,
                    len(model.tokenizer),
                )
                self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
                unfrozen = load_model(
                    base,
                    convert=True,
                    train_backbone=True,
                    decision_tokens=16,
                    max_sequence_length=256,
                )
                self.assertTrue(any(p.requires_grad for p in unfrozen.backbone.parameters()))
                values = model.choice(state="ticket", choices=["approve", "reject"])
                self.assertAlmostEqual(sum(values.probabilities.values()), 1.0, places=6)
                with tempfile.TemporaryDirectory() as output:
                    model.save_pretrained(output)
                    restored = DecisionModel.from_pretrained(output)
                    self.assertEqual(restored.config.head_prompt_version, "span-v1")
                    self.assertFalse(restored.config.train_backbone)
                    self.assertFalse(any(p.requires_grad for p in restored.backbone.parameters()))
                    for option, probability in values.probabilities.items():
                        self.assertAlmostEqual(
                            restored.choice(
                                state="ticket",
                                choices=["approve", "reject"],
                            ).probabilities[option],
                            probability,
                            places=6,
                        )
                    with self.assertRaisesRegex(ConfigError, "head_prompt_version"):
                        DecisionModel.from_pretrained(output, head_prompt_version="marker-v1")

    def test_quantized_pointer_initialization_needs_no_tokenizer_resize(self):
        model = make_model()
        model.backbone.is_loaded_in_4bit = True
        length = len(model.tokenizer)
        with patch.object(
            model.backbone,
            "resize_token_embeddings",
            side_effect=AssertionError("must not resize quantized weights"),
        ):
            converted = DecisionModel(
                model.backbone,
                model.tokenizer,
                head_type="pointer",
                head_prompt_version="span-v1",
                decision_tokens=16,
                max_sequence_length=256,
            )
        self.assertEqual(len(converted.tokenizer), length)
        self.assertEqual(
            converted.backbone.get_input_embeddings().num_embeddings,
            length,
        )
        self.assertEqual(
            set(converted.choice(state="ticket", choices=["approve", "reject"]).probabilities),
            {"approve", "reject"},
        )

    def test_bosun_strategy_converts_text_causal_bases_and_keeps_null_reserved(self):
        for family in ("gpt2", "llama"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                base = Path(temp) / "raw"
                base.mkdir()
                self._raw_checkpoint(base, family)
                model = load_model(
                    base,
                    convert=True,
                    strategy="bosun",
                    max_sequence_length=256,
                )
                self.assertEqual(model.config.head_type, "tokens")
                self.assertEqual(model.config.prompt_format, "bosun-v3")
                self.assertEqual(model.config.prompt_schema, BOSUN_PROMPT_SCHEMA)
                self.assertEqual(len(model.readout.token_ids), 256)
                self.assertTrue(model.config.bosun_template_fallback)
                self.assertTrue(any(p.requires_grad for p in model.backbone.parameters()))
                self.assertEqual(model.tokenizer.padding_side, "left")
                prompt, slots = model.readout.prompt(
                    DecisionInput(state="ticket", question="Route", options=["approve", "reject"]),
                    None,
                )
                self.assertIn("[system]", prompt)
                self.assertNotIn(255, slots)
                result = model.choice(state="ticket", choices=["approve", "reject"])
                self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
                self.assertIsInstance(model.noul(state="ready").decision, bool)
                with tempfile.TemporaryDirectory() as output:
                    model.save_pretrained(output)
                    restored = load_model(Path(output))
                    self.assertEqual(restored.config.prompt_format, "bosun-v3")
                    self.assertTrue(restored.config.bosun_template_fallback)
                    for option, probability in result.probabilities.items():
                        self.assertAlmostEqual(
                            restored.choice(
                                state="ticket",
                                choices=["approve", "reject"],
                            ).probabilities[option],
                            probability,
                            places=5,
                        )

    def test_bosun_strategy_rejects_incompatible_options_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "raw"
            base.mkdir()
            self._raw_checkpoint(base, "gpt2")
            with self.assertRaisesRegex(ConfigError, "exactly 256"):
                load_model(base, convert=True, strategy="bosun", decision_tokens=16)
            with self.assertRaisesRegex(ConfigError, "head_type='tokens'"):
                load_model(base, convert=True, strategy="bosun", head_type="pointer")
            with self.assertRaisesRegex(ConfigError, "before quantizing"):
                load_model(
                    base,
                    convert=True,
                    strategy="bosun",
                    quantization_config=SimpleNamespace(load_in_4bit=True),
                )
            with self.assertRaisesRegex(ConfigError, "only for converting"):
                load_model(base, strategy="bosun")
            with self.assertRaisesRegex(ConfigError, "strategy must be"):
                load_model(base, convert=True, strategy="unknown")
            with self.assertRaisesRegex(ConfigError, "already contains a trained"):
                load_model("Hanno-Labs/bosun-v3.1-0.6b", convert=True, strategy="bosun")

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
    def test_bosun_strategy_full_and_lora_train_export_reload(self):
        for method in ("full", "lora"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                base = Path(temp) / "raw"
                base.mkdir()
                self._raw_checkpoint(base, "gpt2")
                model = load_model(
                    base,
                    convert=True,
                    strategy="bosun",
                    max_sequence_length=256,
                )
                trainer = DecisionTrainer(
                    model=model,
                    dataset=[row("approve", "a"), row("reject", "b")],
                    sft_config=SFTConfig(
                        method=method,
                        batch_size=2,
                        learning_rate=0.01,
                        lora_rank=2,
                        lora_alpha=4,
                    ),
                )
                trainer.train(stages=["decision_sft"])
                output = Path(temp) / "trained"
                trainer.export(output, merge_lora=False)
                restored = load_model(output)
                self.assertEqual(restored.config.prompt_format, "bosun-v3")
                self.assertEqual(restored.config.head_type, "tokens")
                if method == "lora":
                    self.assertTrue((output / "adapter_config.json").is_file())
                expected = model.choice(state="ticket", choices=["approve", "reject"])
                actual = restored.choice(state="ticket", choices=["approve", "reject"])
                for option in expected.probabilities:
                    self.assertAlmostEqual(
                        expected.probabilities[option],
                        actual.probabilities[option],
                        places=5,
                    )

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
    def test_head_only_and_lora_train_export_and_reload(self):
        for method in ("full", "lora"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                base = Path(temp) / "raw"
                base.mkdir()
                self._raw_checkpoint(base, "gpt2")
                model = load_model(
                    base,
                    convert=True,
                    decision_tokens=16,
                    max_sequence_length=256,
                )
                trainer = DecisionTrainer(
                    model=model,
                    dataset=[row("approve", "a"), row("reject", "b")],
                    sft_config=SFTConfig(
                        method=method,
                        batch_size=2,
                        learning_rate=0.01,
                        lora_rank=2,
                        lora_alpha=4,
                    ),
                )
                trainer.train(stages=["decision_sft"])
                output = Path(temp) / "export"
                trainer.export(output, merge_lora=False)
                self.assertEqual(
                    DecisionModel.from_pretrained(output).config.head_prompt_version,
                    "span-v1",
                )
                self.assertFalse(DecisionModel.from_pretrained(output).config.train_backbone)
                expected = model.choice(state="ticket", choices=["approve", "reject"])
                actual = DecisionModel.from_pretrained(output).choice(
                    state="ticket",
                    choices=["approve", "reject"],
                )
                for option in expected.probabilities:
                    self.assertAlmostEqual(
                        expected.probabilities[option],
                        actual.probabilities[option],
                        places=6,
                    )

    @unittest.skipUnless(
        os.environ.get("FUTURIX_RUN_QLORA_SMOKE") == "1"
        and torch.cuda.is_available()
        and importlib.util.find_spec("bitsandbytes"),
        "explicit CUDA QLoRA smoke test",
    )
    def test_qlora_converts_and_reloads_untouched_four_bit_base(self):
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            base.mkdir()
            self._raw_checkpoint(base, "llama")
            model = load_model(
                base,
                convert=True,
                decision_tokens=16,
                max_sequence_length=256,
                quantization_config=quantization,
                device_map={"": 0},
            )
            self.assertTrue(model.backbone.is_loaded_in_4bit)
            trainer = DecisionTrainer(
                model=model,
                dataset=[row("approve", "a"), row("reject", "b")],
                sft_config=SFTConfig(
                    method="qlora",
                    batch_size=2,
                    lora_rank=2,
                    lora_alpha=4,
                    learning_rate=0.003,
                ),
            )
            trainer.train(stages=["decision_sft"])
            output = Path(temp) / "adapter"
            trainer.export(output, merge_lora=False)
            self.assertTrue((output / "adapter_config.json").is_file())
            self.assertTrue((output / HEAD_WEIGHTS).is_file())
            restored = DecisionModel.from_pretrained(
                output,
                quantization_config=quantization,
                device_map={"": 0},
            )
            self.assertEqual(restored.config.head_prompt_version, "span-v1")
            self.assertAlmostEqual(
                sum(
                    restored.choice(
                        state="ticket",
                        choices=["approve", "reject"],
                    ).probabilities.values()
                ),
                1.0,
                places=5,
            )

    @unittest.skipUnless(
        os.environ.get("FUTURIX_RUN_QLORA_SMOKE") == "1"
        and torch.cuda.is_available()
        and importlib.util.find_spec("bitsandbytes"),
        "explicit CUDA Bosun-style QLoRA smoke test",
    )
    def test_prepared_bosun_tokens_can_train_with_qlora(self):
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            raw.mkdir()
            self._raw_checkpoint(raw, "llama")
            prepared = Path(temp) / "prepared"
            unquantized = load_model(
                raw,
                convert=True,
                strategy="bosun",
                max_sequence_length=256,
            )
            unquantized.save_pretrained(prepared)
            del unquantized
            model = load_model(
                prepared,
                quantization_config=quantization,
                device_map={"": 0},
            )
            self.assertTrue(model.backbone.is_loaded_in_4bit)
            self.assertEqual(model.config.prompt_format, "bosun-v3")
            trainer = DecisionTrainer(
                model=model,
                dataset=[row("approve", "a"), row("reject", "b")],
                sft_config=SFTConfig(
                    method="qlora",
                    batch_size=2,
                    lora_rank=2,
                    lora_alpha=4,
                    learning_rate=0.003,
                ),
            )
            trainer.train(stages=["decision_sft"])
            output = Path(temp) / "adapter"
            trainer.export(output, merge_lora=False)
            restored = load_model(
                output,
                quantization_config=quantization,
                device_map={"": 0},
            )
            self.assertEqual(restored.config.prompt_format, "bosun-v3")
            self.assertAlmostEqual(
                sum(
                    restored.choice(
                        state="ticket",
                        choices=["approve", "reject"],
                    ).probabilities.values()
                ),
                1.0,
                places=5,
            )

    @unittest.skipUnless(
        os.environ.get("FUTURIX_RUN_RELEASED_CONVERSION") == "1",
        "opt-in published-weight conversion smoke test",
    )
    def test_released_causal_model_converts_trains_and_reloads(self):
        from huggingface_hub import HfApi

        checkpoint = "HuggingFaceTB/SmolLM2-135M"
        revision = HfApi().model_info(checkpoint).sha
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "hub"
            model = load_model(
                checkpoint,
                revision=revision,
                cache_dir=str(cache),
                convert=True,
                max_sequence_length=256,
                decision_tokens=16,
            )
            self.assertEqual(model.config.head_prompt_version, "span-v1")
            trainer = DecisionTrainer(
                model=model,
                dataset=[row("approve", "a"), row("reject", "b")],
                sft_config=SFTConfig(
                    method="full",
                    batch_size=2,
                    learning_rate=0.01,
                ),
            )
            trainer.train(stages=["decision_sft"])
            destination = Path(temp) / "trained"
            trainer.export(destination)
            del trainer, model
            restored = load_model(destination)
            self.assertEqual(restored.config.source_checkpoint, checkpoint)
            self.assertEqual(restored.config.source_revision, revision)
            self.assertEqual(restored.config.head_prompt_version, "span-v1")
            self.assertAlmostEqual(
                sum(
                    restored.choice(
                        state="ticket",
                        choices=["approve", "reject"],
                    ).probabilities.values()
                ),
                1.0,
                places=5,
            )

    @unittest.skipUnless(
        os.environ.get("FUTURIX_RUN_RELEASED_CONVERSION") == "1",
        "opt-in published-weight Bosun-style smoke test",
    )
    def test_released_causal_model_trains_bosun_style_tokens(self):
        from huggingface_hub import HfApi

        checkpoint = "HuggingFaceTB/SmolLM2-135M"
        revision = HfApi().model_info(checkpoint).sha
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "hub"
            model = load_model(
                checkpoint,
                revision=revision,
                cache_dir=str(cache),
                convert=True,
                strategy="bosun",
                max_sequence_length=256,
                device_map={"": 0} if torch.cuda.is_available() else None,
            )
            self.assertEqual(model.config.prompt_format, "bosun-v3")
            trainer = DecisionTrainer(
                model=model,
                dataset=[row("approve", "a"), row("reject", "b")],
                sft_config=SFTConfig(
                    method="lora",
                    batch_size=2,
                    lora_rank=2,
                    lora_alpha=4,
                ),
            )
            trainer.train(stages=["decision_sft"])
            destination = Path(temp) / "bosun-trained"
            trainer.export(destination, merge_lora=False)
            del model, trainer
            restored = load_model(destination, cache_dir=str(cache))
            self.assertEqual(restored.config.prompt_format, "bosun-v3")
            self.assertEqual(len(restored.readout.token_ids), 256)
            self.assertAlmostEqual(
                sum(
                    restored.choice(
                        state="ticket",
                        choices=["approve", "reject"],
                    ).probabilities.values()
                ),
                1.0,
                places=5,
            )


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class CustomHeadTrainingTests(unittest.TestCase):
    def test_head_only_full_training_updates_head_and_exports(self):
        model = make_model()
        model.backbone.requires_grad_(False)
        before = {k: value.clone() for k, value in model.decision_head.state_dict().items()}
        trainer = DecisionTrainer(
            model=model,
            dataset=[row("approve", "a"), row("reject", "b")],
            sft_config=SFTConfig(method="full", batch_size=2, learning_rate=0.01),
        )
        self.assertTrue(trainer.train(stages=["decision_sft"]))
        self.assertTrue(
            all(math.isfinite(item["loss"]) for item in trainer.history["decision_sft"])
        )
        self.assertTrue(
            any(
                not torch.equal(value.detach().cpu(), before[name])
                for name, value in model.decision_head.state_dict().items()
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            trainer.export(temp)
            result = model.choice(state="fresh", choices=["approve", "reject"])
            loaded = DecisionModel.from_pretrained(temp)
            self.assertAlmostEqual(
                result.probabilities["approve"],
                loaded.choice(state="fresh", choices=["approve", "reject"]).probabilities[
                    "approve"
                ],
                places=6,
            )

    def test_three_stage_lora_checkpoint_reference_and_export(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as temp:
            stages = Path(temp) / "stages"
            trainer = DecisionTrainer(
                model=model,
                dataset=[row("approve", "a"), row("reject", "b")],
                calibration_dataset=[row("approve", "heldout")],
                sft_config=SFTConfig(method="lora", batch_size=2, lora_rank=2, lora_alpha=4),
                rlcd_config=RLCDConfig(method="lora", batch_size=2, group_size=2, kl_beta=0.01),
                checkpoint_dir=stages,
            )
            trainer.train(stages=["decision_sft", "calibration", "rlcd"])
            reference = trainer._active.reference_model
            self.assertIsNotNone(reference.decision_head)
            self.assertIsNot(reference.decision_head, model.decision_head)
            self.assertFalse(any(p.requires_grad for p in reference.decision_head.parameters()))
            for stage in ("decision_sft", "calibration", "rlcd"):
                checkpoint = DecisionModel.from_pretrained(stages / stage)
                self.assertEqual(evaluate(checkpoint, [row()])["count"], 1)
            trainer.export(Path(temp) / "merged")
            merged = DecisionModel.from_pretrained(Path(temp) / "merged")
            self.assertEqual(merged.config.head_type, "pointer")
            self.assertTrue((Path(temp) / "merged" / HEAD_WEIGHTS).is_file())

    def test_unmerged_lora_export_preserves_adapter_and_custom_head(self):
        with tempfile.TemporaryDirectory() as temp:
            base = make_model()
            base.backbone.save_pretrained(Path(temp) / "base")
            base.tokenizer.save_pretrained(Path(temp) / "base")
            model = DecisionModel.from_pretrained(
                Path(temp) / "base",
                head_type="pointer",
                decision_tokens=16,
                max_sequence_length=256,
            )
            trainer = DecisionTrainer(
                model=model,
                dataset=[row("approve")],
                sft_config=SFTConfig(method="lora", batch_size=1, lora_rank=2, lora_alpha=4),
            )
            trainer.train(stages=["decision_sft"])
            trainer.export(Path(temp) / "unmerged", merge_lora=False)
            output = Path(temp) / "unmerged"
            self.assertTrue((output / "adapter_config.json").is_file())
            self.assertTrue((output / HEAD_WEIGHTS).is_file())
            expected = model.choice(state="new", choices=["approve", "reject"])
            restored = DecisionModel.from_pretrained(output)
            actual = restored.choice(state="new", choices=["approve", "reject"])
            for name in expected.probabilities:
                self.assertAlmostEqual(
                    expected.probabilities[name], actual.probabilities[name], places=6
                )

    def test_untied_backbone_lora_trains_custom_head_without_lm_head_modules_to_save(self):
        from test_sft import tiny_untied_model

        base = tiny_untied_model()
        model = DecisionModel(
            base.backbone,
            base.tokenizer,
            head_type="pointer",
            decision_tokens=16,
            max_sequence_length=256,
        )
        trainer = DecisionTrainer(
            model=model,
            dataset=[row("approve")],
            sft_config=SFTConfig(method="lora", batch_size=1, lora_rank=2, lora_alpha=4),
        )
        trainer.train(stages=["decision_sft"])
        self.assertTrue(model.backbone.peft_config)
        self.assertTrue(model.decision_head.state_dict())


if __name__ == "__main__":
    unittest.main()
