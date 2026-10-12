import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from futurix_decision import NativeDecisionModel, NativeSFTConfig
from test_systemone_dataset import GROUP


class NativeTrainingTests(unittest.TestCase):
    def test_laya_groups_soft_targets_and_reloads_native_checkpoint(self):
        def finetune(data, model_dir, output_dir, config=None, device="auto"):
            rows = [json.loads(line) for line in Path(data).read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(list(rows[0]["questions"]), ["action", "valid", "rating"])
            self.assertEqual(rows[0]["gold"]["action"]["probabilities"]["reject"], 0.8)
            self.assertEqual(rows[0]["gold"]["valid"]["probabilities"]["false"], 1)
            self.assertEqual(rows[0]["questions"]["action"]["criteria"]["reject"], "Stop")
            Path(output_dir).mkdir()
            (Path(output_dir) / "model.safetensors").write_bytes(b"native model")
            (Path(output_dir) / "rl_agent_config.json").write_text("{}")
            return {"train_items": 3}

        package = types.ModuleType("laya")
        package.__path__ = []
        package.load = lambda *_args, **_kwargs: object()
        training = types.ModuleType("laya.train")
        training.finetune = finetune
        training.TrainConfig = lambda **options: types.SimpleNamespace(**options)
        with patch.dict(sys.modules, {"laya": package, "laya.train": training}):
            model = NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", object())
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "laya-out"
                tuned = model.finetune([GROUP], output)
                self.assertEqual(tuned.family, "laya")
                self.assertEqual(tuned.training_report["train_items"], 1)
                self.assertEqual(
                    json.loads((output / "futurix_native.json").read_text())["family"], "laya"
                )
                self.assertTrue((output / "model.safetensors").is_file())
                configured = model.finetune(
                    [GROUP],
                    Path(temp) / "laya-configured",
                    config=NativeSFTConfig(epochs=1, batch_size=2, train_backbone=False),
                )
                self.assertEqual(configured.family, "laya")

    def test_gliner_refuses_soft_labels_without_explicit_conversion(self):
        class Extractor:
            @staticmethod
            def from_pretrained(checkpoint):
                return object()

        class Classifier:
            @staticmethod
            def from_pretrained(checkpoint, **kwargs):
                return types.SimpleNamespace(to=lambda **_kwargs: None)

        class Config:
            def __init__(self, output_dir, **options):
                self.output_dir = output_dir
                self.options = options

        class Trainer:
            def __init__(self, model, config):
                self.config = config

            def train(self, train_data):
                row = json.loads(Path(train_data).read_text().splitlines()[0])
                self.assertEqual(row["output"]["classifications"][0]["true_label"], ["reject"])
                final = Path(self.config.output_dir) / "final"
                final.mkdir(parents=True)
                (final / "model.safetensors").write_bytes(b"native gliner")
                (final / "config.json").write_text("{}")
                (final / "encoder_config").mkdir()
                (final / "encoder_config" / "config.json").write_text("{}")
                return {"loss": [0.5]}

        # Trainer is a mock publisher implementation; capture the test assertion.
        Trainer.assertEqual = self.assertEqual
        package = types.ModuleType("gliner2")
        package.__path__ = []
        package.AutoExtractor = Extractor
        classification = types.ModuleType("gliner2.classification")
        classification.Classifier = Classifier
        native_train = types.ModuleType("gliner2.training")
        native_train.__path__ = []
        training = types.ModuleType("gliner2.training.trainer")
        training.ExtractorTrainer = Trainer
        training.TrainingConfig = Config
        with patch.dict(
            sys.modules,
            {
                "gliner2": package,
                "gliner2.classification": classification,
                "gliner2.training": native_train,
                "gliner2.training.trainer": training,
            },
        ):
            model = NativeDecisionModel.from_runtime("fastino/GLiNER2.5-Decide", "gliner", object())
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, "only accepts hard labels"):
                    model.finetune([GROUP], Path(temp) / "bad")
                output = Path(temp) / "good"
                tuned = model.finetune([GROUP], output, target_policy="argmax")
                self.assertEqual(tuned.family, "gliner")
                self.assertTrue((output / "final" / "futurix_native.json").is_file())
                tuned = model.finetune(
                    [GROUP],
                    Path(temp) / "configured",
                    target_policy="argmax",
                    config=NativeSFTConfig(epochs=1, batch_size=2),
                )
                self.assertEqual(tuned.family, "gliner")
                with self.assertRaisesRegex(ValueError, "readout-only"):
                    model.finetune(
                        [GROUP],
                        Path(temp) / "unsupported",
                        target_policy="argmax",
                        config=NativeSFTConfig(train_backbone=False),
                    )

    def test_kev_passes_native_lora_and_pointer_head_artifact(self):
        class Checkpoint:
            def __init__(self, source):
                self.meta = types.SimpleNamespace(
                    base="Qwen/Qwen3.5-9B-Base",
                    base_revision="b" * 40,
                )

            def load(self, device, opts):
                return object(), object()

        class LoadOptions:
            def __init__(self, **kwargs):
                pass

        package = types.ModuleType("kev")
        package.__path__ = []
        checkpoint = types.ModuleType("kev.checkpoint")
        checkpoint.Checkpoint = Checkpoint
        checkpoint.LoadOptions = LoadOptions
        with patch.dict(sys.modules, {"kev": package, "kev.checkpoint": checkpoint}):

            def run(command, check):
                self.assertEqual(command[:3], [sys.executable, "-m", "kev.train"])
                self.assertEqual(command[command.index("--base") + 1], "Qwen/Qwen3.5-9B-Base")
                self.assertEqual(
                    command[command.index("--init_from") + 1], "jaredpalmer/kev-9b@" + "a" * 40
                )
                self.assertEqual(command[command.index("--base_revision") + 1], "b" * 40)
                self.assertEqual(command[command.index("--epochs") + 1], "2")
                rows = [
                    json.loads(line)
                    for line in Path(command[command.index("--data") + 1]).read_text().splitlines()
                ]
                self.assertEqual(rows[0]["questions"]["action"]["target"]["reject"], 0.8)
                self.assertEqual(rows[0]["questions"]["rating"]["label"], 2)
                output = Path(command[command.index("--out") + 1])
                output.mkdir()
                (output / "head.pt").write_bytes(b"native pointer head")
                (output / "adapter_config.json").write_text("{}")
                (output / "adapter_model.safetensors").write_bytes(b"native adapter")

            with patch("futurix_decision.trainers.native.subprocess.run", side_effect=run):
                model = NativeDecisionModel.from_runtime(
                    "jaredpalmer/kev-9b", "kev", (object(), object())
                )
                model.source_revision = "a" * 40
                with tempfile.TemporaryDirectory() as temp:
                    tuned = model.finetune([GROUP], Path(temp) / "kev-out", config={"epochs": 2})
                    self.assertTrue((Path(temp) / "kev-out" / "head.pt").is_file())
                    self.assertEqual(tuned.family, "kev")

    def test_pplx_trains_native_readout_and_saves_reloadable_config(self):
        class PplxModel(torch.nn.Module):
            def __init__(self, checkpoint=None, train=False, device=None):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1)
                self.readout = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
                self.checkpoint = checkpoint

            def prepare(self, records):
                return records

            def forward(self, records):
                return torch.stack([self.readout for _ in records])

            def save(self, path):
                path = Path(path)
                path.mkdir()
                (path / "readout.safetensors").write_bytes(b"native readout")
                (path / "decision_config.json").write_text(
                    '{"attention_mode":"noncausal_full_attention"}'
                )
                (path / "config.json").write_text("{}")
                (path / "model.safetensors").write_bytes(b"native backbone")
                (path / "tokenizer.json").write_text("{}")

        autojev = types.ModuleType("autojev")
        autojev.__path__ = []
        module = types.ModuleType("autojev.model")
        module.DecisionModel = PplxModel
        with patch.dict(sys.modules, {"autojev": autojev, "autojev.model": module}):
            native = PplxModel()
            model = NativeDecisionModel.from_runtime(
                "perplexity-ai/pplx-decider-v1.1-27b",
                "pplx-decider",
                native,
            )
            model.source_revision = "a" * 40
            training_rows = [
                {
                    "state": "A ticket",
                    "question": "Which team?",
                    "options": ["billing", "other"],
                    "target": "billing",
                }
            ]
            with tempfile.TemporaryDirectory() as temp:
                with patch("futurix_decision.models.native._requires_transformers"):
                    tuned = model.finetune(
                        training_rows,
                        Path(temp) / "pplx-out",
                        config=NativeSFTConfig(
                            epochs=2, batch_size=1, learning_rate=0.1, train_backbone=False
                        ),
                    )
                self.assertLess(native.readout[1].item(), 1)
                self.assertEqual(tuned.source_revision, "a" * 40)
                self.assertTrue((Path(temp) / "pplx-out" / "readout.safetensors").is_file())

    def test_clef_aligns_sorted_native_options_and_exports_both_heads(self):
        class LanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

            def save_pretrained(self, directory, **kwargs):
                (directory / "model.safetensors").write_bytes(b"merged backbone")

        class JointModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = LanguageModel()
                self.head = torch.nn.Linear(1, 2, bias=False)
                torch.nn.init.zeros_(self.head.weight)

            def forward(self, batch):
                return [[self.head(torch.ones(1)).flatten()] for _ in batch]

        class Processor:
            tokenizer = types.SimpleNamespace(pad_token_id=0)

            def save_pretrained(self, directory):
                (directory / "processor_config.json").write_text("{}")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "joint_schema_model.py").write_text("# pinned publisher code\n")
            (source / "joint_head_config.json").write_text("{}")
            module = types.SimpleNamespace(
                __file__=str(source / "joint_schema_model.py"),
                encode_record=lambda tok, record, processor: types.SimpleNamespace(
                    questions=[types.SimpleNamespace(option_ids=("billing", "technical"))]
                ),
                collate_records=lambda records, pad_token_id, device: records,
            )
            native = JointModel()
            processor = Processor()
            model = NativeDecisionModel.from_runtime(
                "Cloudflare/clef-flash",
                "clef",
                (module, native, processor),
            )
            model.source_revision = "b" * 40
            data = [
                {
                    "state": "Refund needed",
                    "question": "Which team?",
                    "options": ["technical", "billing"],
                    "target": [0.2, 0.8],
                }
            ]
            with patch(
                "futurix_decision.trainers.native.NativeDecisionModel.from_pretrained",
                return_value=model,
            ):
                tuned = model.finetune(
                    data,
                    root / "out",
                    config=NativeSFTConfig(
                        epochs=2,
                        learning_rate=0.1,
                        train_backbone=False,
                    ),
                )
            self.assertGreater(native.head.weight[0, 0].item(), 0)
            self.assertFalse(native.language_model.weight.requires_grad)
            self.assertEqual(tuned.training_report["train_items"], 1)
            self.assertTrue((root / "out" / "joint_head.safetensors").is_file())
            self.assertTrue((root / "out" / "model.safetensors").is_file())
            self.assertTrue((root / "out" / "joint_schema_model.py").is_file())
            self.assertTrue((root / "out" / "futurix_native.json").is_file())


if __name__ == "__main__":
    unittest.main()
