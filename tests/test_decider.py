"""Native letter-contract checks with a tiny local LM (no upstream weights)."""

import importlib.util
import json
import math
import os
import string
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from futurix_decision import DecisionInput, DecisionModel, DecisionTrainer, SFTConfig
from futurix_decision.datasets.schema import DecisionExample
from futurix_decision.errors import CheckpointError
from futurix_decision.models.decider import (
    DeciderDecisionModel,
    letter_table,
    prepare_decider_rows,
)
from futurix_decision.models.families import FAMILIES
from futurix_decision.vllm.engine import VLLMEngine


def make_native(path: Path) -> None:
    labels = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
    ]
    words = [
        "[UNK]",
        "[EOS]",
        "Context",
        "Question",
        "Options",
        "Answer",
        "no",
        "yes",
        "approve",
        "reject",
        "rate",
        "one",
        "two",
        "three",
    ]
    vocab = {word: index for index, word in enumerate(words + labels)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
    torch.manual_seed(23)
    backbone = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=512,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
    )
    backbone.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "decider_config.json").write_text(
        json.dumps(
            {
                "version": "4b-v2.1",
                "layout": "plain",
                "schema_first": False,
                "isolated_levels": True,
                "max_options": 255,
                "temperature_by_type": {"choice": 1.11, "noul": 1.56, "score": 1.287},
            }
        ),
        encoding="utf-8",
    )


class DeciderIntegrationTests(unittest.TestCase):
    def test_known_family_registry_and_fail_closed_loaders(self):
        self.assertEqual(len(FAMILIES), 12)
        self.assertEqual(
            {family.name for family in FAMILIES if family.training},
            {"bosun", "decider", "jpt", "eikos", "kev", "clef", "pplx-decider", "encoder"},
        )
        jpt = next(family for family in FAMILIES if family.name == "jpt")
        self.assertTrue(jpt.supports("kirp/jpt-4b", "training"))
        self.assertFalse(jpt.supports("kirp/jpt-35b-a3b", "training"))
        with patch("futurix_decision.models.decision_model.load_decision_config", return_value={}):
            with self.assertRaisesRegex(CheckpointError, "publisher's native"):
                DecisionModel.from_pretrained("Cloudflare/clef")
        with patch(
            "futurix_decision.models.decision_model.load_decision_config",
            return_value={"architecture": "autojev"},
        ):
            with self.assertRaisesRegex(CheckpointError, "non-Futurix"):
                DecisionModel.from_pretrained("example/unknown-native-decision-model")

    def test_local_import_prompt_all_three_types_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            with self.assertRaisesRegex(CheckpointError, "native decision"):
                DecisionModel.from_pretrained(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            self.assertIsInstance(model, DeciderDecisionModel)
            self.assertEqual(
                model._token_ids[:2],
                [model.tokenizer.encode(c, add_special_tokens=False)[0] for c in "AB"],
            )
            self.assertNotIn("<|decision_000|>", model.tokenizer.get_vocab())
            inputs = [
                DecisionInput(state="review", question="Next?", options=["approve", "reject"]),
                DecisionInput(
                    kind="noul", state="review", question="Valid?", options=["false", "true"]
                ),
                DecisionInput(
                    kind="score", state="review", question="Rate", options=["0", "1", "2"]
                ),
            ]
            prepared = prepare_decider_rows(inputs[2], model.tokenizer)
            self.assertEqual(len(prepared), 3)
            self.assertEqual(
                prepared[0].input_ids,
                model.tokenizer.encode("Context:\nreview", add_special_tokens=False)
                + model.tokenizer.encode(
                    "\n\nQuestion: Rate\nProposed answer: 0\nDoes the proposed answer fit?"
                    "\nOptions:\n(A) no\n(B) yes\nAnswer: (",
                    add_special_tokens=False,
                ),
            )
            batch = model.forward_decisions(inputs)
            self.assertEqual(batch.logits.shape, (3, 3))
            self.assertTrue(torch.isneginf(batch.logits[0, 2]))
            self.assertEqual(batch.slots, [[0, 1], [0, 1], [0, 1, 2]])
            self.assertTrue(
                torch.isfinite(
                    model.forward_decisions(inputs, assignment_seed=71).logits[0, :2]
                ).all()
            )
            probabilities = model.batch_decide(inputs)
            for prediction in probabilities:
                self.assertAlmostEqual(sum(prediction.probabilities.values()), 1, places=6)
            self.assertIsInstance(probabilities[1].decision, bool)
            self.assertIsInstance(probabilities[2].decision, int)
            output = Path(temp) / "export"
            model.save_pretrained(output)
            metadata = json.loads((output / "decision_config.json").read_text())
            self.assertEqual(metadata["prompt_format"], "decider-v2.1")
            self.assertEqual(metadata["type_temperatures"]["noul"], 1.56)
            restored = DecisionModel.from_pretrained(output)
            self.assertIsInstance(restored, DeciderDecisionModel)
            for before, after in zip(probabilities, restored.batch_decide(inputs)):
                for option in before.probabilities:
                    self.assertAlmostEqual(
                        before.probabilities[option], after.probabilities[option], places=6
                    )

    def test_wide_labels_and_untrained_readout_never_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            names, ids = letter_table(model.tokenizer, 255)
            self.assertEqual(names[10], "K")
            self.assertEqual(len(ids), 255)
            row = DecisionInput(
                state="review", question="Choose", options=[f"one {i}" for i in range(30)]
            )
            prepared = prepare_decider_rows(row, model.tokenizer)
            self.assertEqual(prepared[0].letter_ids, ids[:30])
            self.assertIn(ids[26], prepared[0].input_ids)
            self.assertEqual(
                len(
                    model.choice(
                        state="review", choices=[f"one {i}" for i in range(30)]
                    ).probabilities
                ),
                30,
            )
            with self.assertRaisesRegex(ValueError, "at most 10 levels"):
                prepare_decider_rows(
                    DecisionInput(
                        kind="score",
                        state="review",
                        question="Rate",
                        options=[str(i) for i in range(11)],
                    ),
                    model.tokenizer,
                )

    def test_true_first_noul_is_reordered_and_unknown_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            normal = DecisionInput(
                kind="noul",
                state="review",
                question="Valid?",
                options=["false", "true"],
            )
            reversed_row = DecisionInput(
                kind="noul",
                state="review",
                question="Valid?",
                options=["true", "false"],
                candidate_ids=["true", "false"],
                preserve_order=True,
            )
            before = model.decide(normal)
            after = model.decide(reversed_row)
            self.assertEqual(
                prepare_decider_rows(normal, model.tokenizer)[0].input_ids,
                prepare_decider_rows(reversed_row, model.tokenizer)[0].input_ids,
            )
            self.assertAlmostEqual(before.probabilities["false"], after.probabilities["false"])
            invalid = DecisionInput(
                kind="noul",
                state="review",
                question="Valid?",
                options=["c0", "c1"],
                candidate_ids=["c0", "c1"],
                candidate_labels=["Reject", "Approve"],
                preserve_order=True,
            )
            with self.assertRaisesRegex(ValueError, "literal true/false"):
                model.decide(invalid)

    def test_batched_vllm_uses_the_same_prompt_and_temperature(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            export = Path(temp) / "export"
            model.save_pretrained(export)

            class SamplingParams:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            class FakeLLM:
                def generate(self, prompts, sampling_params, use_tqdm=False):
                    self.prompts, self.params = prompts, sampling_params
                    return [
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    logprobs=[
                                        {
                                            token: SimpleNamespace(logprob=math.log(prob))
                                            for token, prob in zip(
                                                params.allowed_token_ids, (0.7, 0.3)
                                            )
                                        }
                                    ]
                                )
                            ]
                        )
                        for params in sampling_params
                    ]

            rows = [
                DecisionInput(state="review", question="Next?", options=["approve", "reject"]),
                DecisionInput(
                    kind="score", state="review", question="Rate", options=["0", "1", "2"]
                ),
            ]
            llm = FakeLLM()
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SamplingParams)}):
                engine = VLLMEngine(export, llm=llm, tokenizer=model.tokenizer)
                result = engine.batch_decide(rows)
            self.assertEqual(len(llm.prompts), 4)
            self.assertEqual(
                llm.prompts[1]["prompt_token_ids"],
                prepare_decider_rows(rows[1], model.tokenizer)[0].input_ids,
            )
            self.assertEqual(result[0].decision, "approve")
            self.assertAlmostEqual(
                result[0].probabilities["approve"],
                0.7 ** (1 / 1.11) / (0.7 ** (1 / 1.11) + 0.3 ** (1 / 1.11)),
            )
            self.assertTrue(all(params.max_tokens == 1 for params in llm.params))
            self.assertTrue(all(not hasattr(params, "logprob_token_ids") for params in llm.params))
            self.assertAlmostEqual(result[1].probabilities["0"], 1 / 3)

    def test_wide_vllm_logprobs_do_not_exceed_explicit_id_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source)
            model.max_sequence_length = 4096
            export = Path(temp) / "export"
            model.save_pretrained(export)

            class SamplingParams:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            class FakeLLM:
                def generate(self, prompts, params, use_tqdm=False):
                    self.params = params
                    return [
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    logprobs=[
                                        {
                                            token: SimpleNamespace(
                                                logprob=-math.log(len(p.allowed_token_ids))
                                            )
                                            for token in p.allowed_token_ids
                                        }
                                    ]
                                )
                            ]
                        )
                        for p in params
                    ]

            fake = FakeLLM()
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SamplingParams)}):
                engine = VLLMEngine(export, llm=fake, tokenizer=model.tokenizer)
                prediction = engine.choice(
                    state="review", choices=[f"choice {i}" for i in range(255)]
                )
            self.assertEqual(len(prediction.probabilities), 255)
            self.assertEqual(fake.params[0].logprobs, 255)
            self.assertFalse(hasattr(fake.params[0], "logprob_token_ids"))

    def test_vllm_true_first_noul_keeps_original_option_order(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            export = Path(temp) / "export"
            model.save_pretrained(export)

            class SamplingParams:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            class FakeLLM:
                def generate(self, prompts, params, use_tqdm=False):
                    selected = params[0].allowed_token_ids
                    return [
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    logprobs=[
                                        {
                                            selected[0]: SimpleNamespace(logprob=math.log(0.8)),
                                            selected[1]: SimpleNamespace(logprob=math.log(0.2)),
                                        }
                                    ]
                                )
                            ]
                        )
                    ]

            row = DecisionInput(
                kind="noul",
                state="review",
                question="Valid?",
                options=["true", "false"],
                candidate_ids=["true", "false"],
                preserve_order=True,
            )
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SamplingParams)}):
                engine = VLLMEngine(export, llm=FakeLLM(), tokenizer=model.tokenizer)
                result = engine.decide(row)
            self.assertIs(result.decision, False)
            self.assertGreater(result.probabilities["false"], result.probabilities["true"])

    @unittest.skipUnless(os.environ.get("FUTURIX_TEST_VLLM") == "1", "opt-in GPU vLLM smoke test")
    def test_real_vllm_wide_letter_readout_matches_pytorch(self):
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            tokenizer = AutoTokenizer.from_pretrained(source)
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=1024,
                    tie_word_embeddings=False,
                )
            ).save_pretrained(source)
            model = DecisionModel.from_decider(source, max_sequence_length=1024)
            output = Path(temp) / "export"
            model.save_pretrained(output)
            options = letter_table(model.tokenizer, 129)[0]
            row = DecisionInput(state="review", question="Choose", options=options)
            reference = model.decide(row)
            engine = VLLMEngine(
                output,
                max_model_len=1024,
                gpu_memory_utilization=0.04,
                enforce_eager=True,
                dtype="float16",
            )
            actual = engine.decide(row)
            for option in options:
                self.assertAlmostEqual(
                    reference.probabilities[option], actual.probabilities[option], delta=0.02
                )

    def test_import_refuses_unknown_native_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            make_native(path)
            config = path / "decider_config.json"
            data = json.loads(config.read_text())
            data["layout"] = "chat"
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(CheckpointError, "only the Mapika decider-4b"):
                DecisionModel.from_decider(path)


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class DeciderTrainingTests(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes"),
        "4-bit CUDA backend not installed",
    )
    def test_native_qlora_merge_uses_recoverable_source_checkpoint(self):
        from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(source)
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=512,
                    tie_word_embeddings=False,
                )
            ).save_pretrained(source)
            model = DecisionModel.from_decider(
                source,
                max_sequence_length=512,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                device_map="cuda:0",
            )
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
                sft_config=SFTConfig(method="qlora", batch_size=1, lora_rank=2, lora_alpha=4),
            )
            trainer.train()
            output = Path(temp) / "merged"
            trainer.export(output)
            self.assertIsInstance(DecisionModel.from_pretrained(output), DeciderDecisionModel)

    def test_cli_native_import_training_and_export(self):
        from futurix_decision.cli import main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "native"
            source.mkdir()
            make_native(source)
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps(
                    DecisionExample(
                        id="train-1",
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
                    "decider",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--method",
                    "sft",
                    "--max-examples",
                    "1",
                ]
            )
            self.assertIsInstance(DecisionModel.from_pretrained(output), DeciderDecisionModel)

    def test_lora_finetuning_preserves_letter_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            trainer = DecisionTrainer(
                model=model,
                dataset=[
                    DecisionExample(
                        id="one",
                        state="review",
                        question="Next?",
                        options=["approve", "reject"],
                        target="approve",
                    )
                ],
                sft_config=SFTConfig(method="lora", lora_rank=2, lora_alpha=4, batch_size=1),
            )
            trainer.train(stages=["decision_sft"])
            self.assertTrue(model.backbone.peft_config)
            expected = model.choice(state="review", choices=["approve", "reject"])
            destination = Path(temp) / "merged"
            trainer.export(destination)
            self.assertEqual(
                json.loads((destination / "decision_config.json").read_text())["prompt_format"],
                "decider-v2.1",
            )
            restored = DecisionModel.from_pretrained(destination)
            self.assertIsInstance(restored, DeciderDecisionModel)
            actual = restored.choice(state="review", choices=["approve", "reject"])
            for option in expected.probabilities:
                self.assertAlmostEqual(
                    expected.probabilities[option], actual.probabilities[option], places=6
                )

    def test_unmerged_native_lora_can_continue_training(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            rows = [
                DecisionExample(
                    id="x",
                    state="review",
                    question="Next?",
                    options=["approve", "reject"],
                    target="approve",
                )
            ]
            config = SFTConfig(method="lora", lora_rank=2, lora_alpha=4, batch_size=1)
            trainer = DecisionTrainer(model=model, dataset=rows, sft_config=config)
            trainer.train()
            unmerged = Path(temp) / "adapter"
            trainer.export(unmerged, merge_lora=False)
            self.assertEqual(
                json.loads((unmerged / "decision_config.json").read_text())["base_checkpoint"],
                str(source),
            )
            with self.assertRaisesRegex(ValueError, "trainable_adapter=True"):
                DecisionTrainer(
                    model=DecisionModel.from_pretrained(unmerged),
                    dataset=rows,
                    sft_config=config,
                ).train()
            trainable = DecisionModel.from_pretrained(unmerged, trainable_adapter=True)
            self.assertTrue(any(param.requires_grad for param in trainable.backbone.parameters()))
            continued = DecisionTrainer(model=trainable, dataset=rows, sft_config=config)
            continued.train()
            output = Path(temp) / "merged"
            continued.export(output)
            self.assertIsInstance(DecisionModel.from_pretrained(output), DeciderDecisionModel)

    def test_sft_calibration_rlcd_and_checkpoint_reload(self):
        from futurix_decision.trainers import RLCDConfig

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "native"
            source.mkdir()
            make_native(source)
            model = DecisionModel.from_decider(source, max_sequence_length=512)
            before = model.backbone.get_output_embeddings().weight.detach().clone()
            rows = [
                DecisionExample(
                    id=f"train-{i}",
                    state=f"review {i}",
                    question="Next?",
                    options=["approve", "reject"],
                    target="approve",
                )
                for i in range(2)
            ]
            trainer = DecisionTrainer(
                model=model,
                dataset=rows,
                calibration_dataset=[
                    DecisionExample(
                        id="heldout",
                        state="review",
                        question="Next?",
                        options=["approve", "reject"],
                        target="approve",
                    )
                ],
                sft_config=SFTConfig(method="full", batch_size=1),
                rlcd_config=RLCDConfig(method="full", batch_size=1, group_size=2, kl_beta=0),
                checkpoint_dir=Path(temp) / "stages",
            )
            trainer.train(stages=["decision_sft", "calibration", "rlcd"])
            self.assertFalse(
                torch.equal(before, model.backbone.get_output_embeddings().weight.detach().cpu())
            )
            for stage in ("decision_sft", "calibration", "rlcd"):
                restored = DecisionModel.from_pretrained(Path(temp) / "stages" / stage)
                self.assertIsInstance(restored, DeciderDecisionModel)
                self.assertAlmostEqual(
                    sum(
                        restored.choice(
                            state="review", choices=["approve", "reject"]
                        ).probabilities.values()
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
