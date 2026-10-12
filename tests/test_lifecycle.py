import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from futurix_decision import DecisionModel, DecisionTrainer, SFTConfig, evaluate
from futurix_decision.trainers import RLCDConfig


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class LifecycleTests(unittest.TestCase):
    def test_rlcd_uses_saved_reference_instead_of_deepcopy(self):
        from test_sft import example, tiny_model

        with tempfile.TemporaryDirectory() as temp:
            checkpoint_dir = Path(temp) / "stages"
            rows = [example([0.8, 0.2], "train")]
            heldout = [example([0.6, 0.4], "calibration")]
            sft = SFTConfig(method="full", batch_size=1)
            rlcd = RLCDConfig(method="full", batch_size=1, group_size=2, kl_beta=0.01)
            trainer = DecisionTrainer(
                model=tiny_model(),
                dataset=rows,
                calibration_dataset=heldout,
                sft_config=sft,
                rlcd_config=rlcd,
                checkpoint_dir=checkpoint_dir,
            )
            trainer.train(stages=["decision_sft", "calibration"])
            resumed = DecisionTrainer(
                model=DecisionModel.from_pretrained(checkpoint_dir / "calibration"),
                dataset=rows,
                calibration_dataset=heldout,
                sft_config=sft,
                rlcd_config=rlcd,
                checkpoint_dir=checkpoint_dir,
            )
            resumed.history.update(trainer.history)
            resumed.train(stages=["rlcd"])
            self.assertIsNotNone(resumed._active.reference_model)
            self.assertIsNot(resumed._active.reference_model.backbone, resumed.model.backbone)
            self.assertEqual(list(resumed.history), ["decision_sft", "calibration", "rlcd"])

    def test_stage_checkpoints_are_reloadable_and_progress_is_atomic(self):
        from test_sft import example, tiny_model

        with tempfile.TemporaryDirectory() as temp:
            checkpoint_dir = Path(temp) / "stages"
            trainer = DecisionTrainer(
                model=tiny_model(),
                dataset=[example([0.8, 0.2], "train")],
                calibration_dataset=[example([0.6, 0.4], "calibration")],
                validation_dataset=[example([0.8, 0.2], "validation")],
                sft_config=SFTConfig(method="full", batch_size=1),
                checkpoint_dir=checkpoint_dir,
            )
            trainer.train(stages=["decision_sft", "calibration"])
            progress = json.loads((checkpoint_dir / "progress.json").read_text())
            self.assertEqual(progress["completed_stages"], ["decision_sft", "calibration"])
            self.assertEqual(set(progress["validation_metrics"]), {"decision_sft", "calibration"})
            self.assertIn(trainer.select_best(metric="accuracy"), progress["completed_stages"])
            self.assertEqual(
                progress["checkpoint"], str((checkpoint_dir / "calibration").resolve())
            )
            for stage in progress["completed_stages"]:
                loaded = DecisionModel.from_pretrained(checkpoint_dir / stage)
                self.assertEqual(evaluate(loaded, [example([0.8, 0.2])])["count"], 1)

    def test_three_stages_continue_existing_lora_adapter(self):
        from test_sft import example, tiny_model

        rows = [example([0.7, 0.3], "train-a"), example([0.2, 0.8], "train-b")]
        trainer = DecisionTrainer(
            model=tiny_model(),
            dataset=rows,
            calibration_dataset=[example([0.7, 0.3], "heldout")],
            sft_config=SFTConfig(method="lora", batch_size=2, lora_rank=2, lora_alpha=4),
        )
        result = trainer.train(stages=["decision_sft", "calibration", "rlcd"])
        self.assertEqual(list(result), ["decision_sft", "calibration", "rlcd"])
        with tempfile.TemporaryDirectory() as temp:
            trainer.export(temp, merge_lora=True)
            self.assertTrue((Path(temp) / "config.json").is_file())
            self.assertFalse((Path(temp) / "adapter_config.json").exists())

    def test_three_stages_export_and_evaluation(self):
        from test_sft import example, tiny_model

        rows = [example([0.7, 0.3], "valid"), example([0.2, 0.8], "invalid")]
        model = tiny_model()
        trainer = DecisionTrainer(
            model=model,
            dataset=rows,
            calibration_dataset=[example([0.7, 0.3], "holdout")],
            sft_config=SFTConfig(batch_size=2, learning_rate=0.001),
            rlcd_config=RLCDConfig(batch_size=2, group_size=8, learning_rate=0.001, kl_beta=0.01),
        )
        history = trainer.train(stages=["decision_sft", "calibration", "rlcd"])
        self.assertEqual(set(history), {"decision_sft", "calibration", "rlcd"})
        self.assertIn("ece", history["rlcd"][0])
        with tempfile.TemporaryDirectory() as temp:
            trainer.export(temp, merge_lora=True, safetensors=True)
            output = Path(temp)
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "generation_config.json").is_file())
            self.assertTrue((output / "decision_config.json").is_file())
            self.assertTrue((output / "training_metadata.json").is_file())
            self.assertTrue(list(output.glob("*.safetensors")))
            meta = json.loads((output / "training_metadata.json").read_text())
            self.assertEqual(meta["stages"], ["decision_sft", "calibration", "rlcd"])
            restored = DecisionModel.from_pretrained(temp)
            self.assertAlmostEqual(restored.config.temperature, model.config.temperature)
            self.assertEqual(evaluate(restored, rows, batch_size=2)["count"], 2)

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes"),
        "4-bit CUDA backend not installed",
    )
    def test_qlora_export_merges_into_full_checkpoint(self):
        from test_sft import example, tiny_untied_model

        with tempfile.TemporaryDirectory() as temp:
            base, output = Path(temp) / "base", Path(temp) / "export"
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
            trainer = DecisionTrainer(
                model=model,
                dataset=[example("approve", "one"), example("reject", "two")],
                sft_config=SFTConfig(
                    method="qlora", batch_size=2, lora_rank=2, lora_alpha=4, learning_rate=0.002
                ),
            )
            trainer.train(stages=["decision_sft"])
            trainer.export(output, merge_lora=True, safetensors=True)
            self.assertTrue((output / "config.json").is_file())
            self.assertFalse((output / "adapter_config.json").exists())
            loaded = DecisionModel.from_pretrained(output)
            self.assertEqual(loaded.config.decision_tokens, 16)
            self.assertAlmostEqual(
                sum(
                    loaded.choice(
                        state="test", choices=["approve", "reject"]
                    ).probabilities.values()
                ),
                1.0,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
