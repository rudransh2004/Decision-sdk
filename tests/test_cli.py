import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from futurix_decision.cli import main
from futurix_decision.utils.config import load_config


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class CLITests(unittest.TestCase):
    def test_config_train_evaluate_and_export(self):
        from test_sft import example, tiny_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, output, copied = root / "base", root / "trained", root / "copied"
            tiny_model().save_pretrained(base)
            data = root / "decisions.jsonl"
            data.write_text(
                "\n".join(
                    json.dumps(row.model_dump())
                    for row in [example("approve", "one"), example([0.3, 0.7], "two")]
                )
                + "\n",
                encoding="utf-8",
            )
            config = root / "config.yaml"
            config.write_text(
                f"model:\n  name: {base}\n"
                f"dataset: {data}\noutput: {output}\n"
                "training:\n  sft:\n    enabled: true\n    method: full\n"
                "    epochs: 1\n    batch_size: 2\n    precision: 'no'\n",
                encoding="utf-8",
            )
            self.assertEqual(load_config(config)["model"]["name"], str(base))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--config", str(config)])
            self.assertTrue((output / "training_metadata.json").is_file())
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                main(["evaluate", "--model", str(output), "--dataset", str(data)])
            result = json.loads(captured.getvalue())
            self.assertEqual(result["count"], 2)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["export", "--model", str(output), "--output", str(copied)])
            self.assertTrue((copied / "model.safetensors").is_file())
            self.assertTrue((copied / "training_metadata.json").is_file())

    def test_rtx_yaml_settings_are_real(self):
        config = load_config(Path(__file__).parents[1] / "configs" / "rtx_pro_6000.yaml")
        self.assertEqual(config["training"]["rlcd"]["group_size"], 8)
        self.assertEqual(config["model"]["decision_tokens"], 256)

    def test_yaml_can_train_and_evaluate_pointer_head(self):
        from futurix_decision import DecisionModel
        from test_sft import example, tiny_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, output, data = root / "base", root / "headed", root / "train.jsonl"
            raw = tiny_model()
            raw.backbone.save_pretrained(base)
            raw.tokenizer.save_pretrained(base)
            data.write_text(json.dumps(example("approve").model_dump()) + "\n")
            config = root / "pointer.yaml"
            config.write_text(
                f"model:\n  name: {base}\n  head_type: pointer\n"
                "  head_config:\n    width: 16\n  decision_tokens: 16\n"
                f"dataset: {data}\noutput: {output}\n"
                "training:\n  sft:\n    enabled: true\n    method: full\n    batch_size: 1\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--config", str(config)])
            self.assertEqual(DecisionModel.from_pretrained(output).head_type, "pointer")
            self.assertTrue((output / "decision_head.safetensors").is_file())
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                main(["evaluate", "--model", str(output), "--dataset", str(data)])
            self.assertEqual(json.loads(captured.getvalue())["count"], 1)

    def test_three_stage_cli_from_yaml(self):
        from test_sft import example, tiny_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, output, data, heldout = (
                root / "base",
                root / "final",
                root / "train.jsonl",
                root / "calibration.jsonl",
            )
            tiny_model().save_pretrained(base)
            data.write_text(json.dumps(example([0.7, 0.3]).model_dump()) + "\n", encoding="utf-8")
            heldout.write_text(
                json.dumps(example([0.7, 0.3], "heldout").model_dump()) + "\n", encoding="utf-8"
            )
            config = root / "all.yaml"
            config.write_text(
                f"model:\n  name: {base}\ndataset: {data}\noutput: {output}\n"
                "training:\n  sft:\n    enabled: true\n    method: full\n    batch_size: 1\n"
                "  calibration:\n    enabled: true\n    method: full\n    batch_size: 1\n"
                "  rlcd:\n    enabled: true\n    method: full\n    batch_size: 1\n"
                "    group_size: 8\nrewards:\n  log: 1.0\n  spherical: 0.5\n  ranked: 1.0\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--config", str(config), "--calibration-dataset", str(heldout)])
            metadata = json.loads((output / "training_metadata.json").read_text())
            self.assertEqual(metadata["stages"], ["decision_sft", "calibration", "rlcd"])

    def test_cli_resumes_from_reloadable_stage(self):
        from test_sft import example, tiny_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, output, data, heldout, stages = (
                root / "base",
                root / "final",
                root / "train.jsonl",
                root / "calibration.jsonl",
                root / "stages",
            )
            tiny_model().save_pretrained(base)
            data.write_text(json.dumps(example([0.8, 0.2], "one").model_dump()) + "\n")
            heldout.write_text(json.dumps(example([0.6, 0.4], "heldout").model_dump()) + "\n")
            config = root / "config.yaml"
            config.write_text(
                f"model:\n  name: {base}\ndataset: {data}\noutput: {output}\n"
                f"checkpoint_dir: {stages}\n"
                "training:\n  sft:\n    enabled: true\n    batch_size: 1\n"
                "  calibration:\n    enabled: true\n    batch_size: 1\n"
                "  rlcd:\n    enabled: true\n    batch_size: 1\n    kl_beta: 0.0\n"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--config", str(config), "--stages", "decision_sft"])
                main(
                    [
                        "train",
                        "--config",
                        str(config),
                        "--resume",
                        "--calibration-dataset",
                        str(heldout),
                    ]
                )
            progress = json.loads((stages / "progress.json").read_text())
            self.assertEqual(progress["completed_stages"], ["decision_sft", "calibration", "rlcd"])
            self.assertEqual(
                json.loads((output / "training_metadata.json").read_text())["stages"],
                ["decision_sft", "calibration", "rlcd"],
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes"),
        "4-bit CUDA backend not installed",
    )
    def test_qlora_cli_loads_4bit_and_merges(self):
        from test_sft import example, tiny_untied_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, output, data = root / "base", root / "output", root / "decisions.jsonl"
            tiny_untied_model().save_pretrained(base)
            data.write_text(json.dumps(example().model_dump()) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "train",
                        "--model",
                        str(base),
                        "--dataset",
                        str(data),
                        "--output",
                        str(output),
                        "--method",
                        "sft",
                        "--finetune-method",
                        "qlora",
                    ]
                )
            self.assertTrue((output / "config.json").is_file())
            self.assertFalse((output / "adapter_config.json").exists())


if __name__ == "__main__":
    unittest.main()
