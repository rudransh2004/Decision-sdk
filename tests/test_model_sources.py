"""Hub/local routing and missing-artifact checks without downloading giant weights."""

import json
import tempfile
import types
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.errors import EntryNotFoundError

from futurix_decision import DecisionModel, load_model
from futurix_decision.errors import CheckpointError
from futurix_decision.models.sources import (
    checkpoint_source,
    infer_local_family,
    require_checkpoint_files,
)
from test_bosun import bosun_tiny_model
from test_decision_model import tiny_checkpoint


def put(root: Path, filename: str, value: str = "{}") -> None:
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def clef_copy(root: Path) -> None:
    put(root, "config.json")
    put(root, "tokenizer.json")
    put(root, "joint_head_config.json")
    put(root, "joint_head.safetensors")
    put(root, "model.safetensors")
    put(
        root,
        "joint_schema_model.py",
        (
            "def load_release_model(directory, device='cpu', **kwargs):\n"
            "    return object(), object()\n"
        ),
    )


def pplx_copy(root: Path) -> None:
    put(root, "config.json")
    put(root, "tokenizer.json")
    put(root, "model.safetensors")
    put(root, "readout.safetensors")
    put(
        root,
        "decision_config.json",
        json.dumps(
            {
                "format_version": 1,
                "attention_mode": "noncausal_full_attention",
            }
        ),
    )


def cached_file(root: Path, filename: str) -> str:
    candidate = root / filename
    if not candidate.is_file():
        raise EntryNotFoundError(f"missing {filename}")
    return str(candidate)


class SourceTests(unittest.TestCase):
    def test_explicit_missing_local_path_and_non_directory_never_use_hub(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Hub used")):
                with self.assertRaisesRegex(FileNotFoundError, "local checkpoint directory"):
                    load_model(root / "absent", family="clef", trust_remote_code=True)
                with self.assertRaisesRegex(NotADirectoryError, "must be a directory"):
                    load_model(Path(__file__), family="clef", trust_remote_code=True)
            self.assertEqual(checkpoint_source("Cloudflare/clef"), ("Cloudflare/clef", False))

    def test_local_clef_copy_loads_without_manifest_or_hub_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "renamed-custom-clef"
            root.mkdir()
            clef_copy(root)
            with (
                patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Hub used")),
                patch("futurix_decision.models.native._requires_transformers"),
            ):
                loaded = load_model(root, trust_remote_code=True)
                self.assertEqual(loaded.family, "clef")
                self.assertIsNone(loaded.source_revision)
                self.assertEqual(loaded.checkpoint, str(root))
                with self.assertRaisesRegex(CheckpointError, "belongs to clef"):
                    load_model(root, family="pplx-decider", trust_remote_code=True)
                with self.assertRaisesRegex(ValueError, "trust_remote_code=True"):
                    load_model(root)
            (root / "joint_head.safetensors").unlink()
            with self.assertRaisesRegex(CheckpointError, "joint_head.safetensors"):
                load_model(root, trust_remote_code=True)

    def test_hub_clef_downloads_and_checks_missing_head_before_executing_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clef_copy(root)
            revision = "a" * 40
            with (
                patch("huggingface_hub.snapshot_download", return_value=str(root)) as download,
                patch("futurix_decision.models.native._requires_transformers"),
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=lambda repo, filename, **kwargs: cached_file(root, filename),
                ),
            ):
                loaded = load_model(
                    "Cloudflare/clef-flash",
                    revision=revision,
                    trust_remote_code=True,
                )
                self.assertEqual(loaded.family, "clef")
                download.assert_called_once_with("Cloudflare/clef-flash", revision=revision)
                (root / "joint_head.safetensors").unlink()
                with self.assertRaisesRegex(
                    CheckpointError,
                    r"Cloudflare/clef-flash.*joint_head.safetensors",
                ):
                    load_model("clef-flash", revision=revision, trust_remote_code=True)

    def test_pplx_local_and_hub_missing_readout_and_shards(self):
        package = types.ModuleType("autojev")
        package.__path__ = []
        module = types.ModuleType("autojev.model")

        class FakePplx:
            def __init__(self, *, checkpoint, train, device, **kwargs):
                self.checkpoint = Path(checkpoint)

        module.DecisionModel = FakePplx
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                "sys.modules",
                {"autojev": package, "autojev.model": module},
            ),
            patch("futurix_decision.models.native._requires_transformers"),
        ):
            root = Path(temp)
            pplx_copy(root)
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Hub used")):
                self.assertEqual(load_model(root, trust_remote_code=True).runtime.checkpoint, root)
            (root / "model.safetensors").unlink()
            put(
                root,
                "model.safetensors.index.json",
                json.dumps(
                    {
                        "weight_map": {"layer": "model-00001-of-00002.safetensors"},
                    }
                ),
            )
            with self.assertRaisesRegex(CheckpointError, "model-00001-of-00002.safetensors"):
                load_model(root, trust_remote_code=True)
            put(
                root,
                "model.safetensors.index.json",
                json.dumps(
                    {
                        "weight_map": {"layer": "../outside.safetensors"},
                    }
                ),
            )
            with self.assertRaisesRegex(CheckpointError, "inside its directory"):
                require_checkpoint_files(root, "pplx-decider")
            (root / "model.safetensors.index.json").unlink()
            put(root, "model.safetensors")
            (root / "readout.safetensors").unlink()
            with (
                patch("huggingface_hub.snapshot_download", return_value=str(root)),
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=lambda repo, filename, **kwargs: cached_file(root, filename),
                ),
            ):
                with self.assertRaisesRegex(
                    CheckpointError,
                    r"perplexity-ai/pplx-decider-v1.1-27b.*readout.safetensors",
                ):
                    load_model(
                        "pplx-decider",
                        revision="b" * 40,
                        trust_remote_code=True,
                    )

    def test_kev_hub_reports_missing_pointer_head(self):
        package = types.ModuleType("kev")
        package.__path__ = []
        checkpoint = types.ModuleType("kev.checkpoint")
        checkpoint.Checkpoint = object
        checkpoint.LoadOptions = object
        with (
            patch.dict("sys.modules", {"kev": package, "kev.checkpoint": checkpoint}),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=EntryNotFoundError("No such file"),
            ),
        ):
            with self.assertRaisesRegex(
                CheckpointError,
                r"jaredpalmer/kev-9b.*missing: head.pt",
            ):
                load_model("kev", revision="c" * 40)

    @unittest.skipIf(
        int(version("transformers").split(".")[0]) >= 5,
        "tiny Bosun tokenizer fixture uses Transformers 4",
    )
    def test_bosun_local_copy_preserves_decision_contract(self):
        base = bosun_tiny_model()
        package = types.SimpleNamespace(
            model=base.backbone,
            tokenizer=base.tokenizer,
            config=types.SimpleNamespace(
                prompt_schema="bosun-decision-prompt-v3-stable-slots",
                decision_token_count=256,
                null_token_index=255,
                base_model_name_or_path="Qwen/Qwen3-0.6B",
                base_model_revision="a" * 40,
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            put(
                root,
                "config.json",
                json.dumps(
                    {
                        "model_type": "bosun",
                        "decision_embeddings_file": "decision_embeddings.safetensors",
                        "adapter_subfolder": "adapter",
                        "tokenizer_subfolder": "tokenizer",
                    }
                ),
            )
            put(root, "adapter/adapter_config.json")
            put(root, "adapter/adapter_model.safetensors")
            put(root, "tokenizer/tokenizer.json")
            with self.assertRaisesRegex(CheckpointError, "decision_embeddings.safetensors"):
                load_model(root)
            put(root, "decision_embeddings.safetensors")
            with patch(
                "futurix_decision.models.decision_model.AutoModelForCausalLM.from_pretrained",
                return_value=package,
            ) as loader:
                model = load_model(root)
            loader.assert_called_once_with(str(root), trust_remote_code=True)
            self.assertEqual(model.prompt_format, "bosun-v3")
            self.assertEqual(model.source_checkpoint, str(root))

    def test_local_letter_variants_and_futurix_export_route_to_their_readouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for family, marker in (
                ("decider", "decider_config.json"),
                ("eikos", "decision_config.json"),
                ("jpt", "README.md"),
            ):
                with self.subTest(family=family):
                    variant = root / family
                    variant.mkdir()
                    put(
                        variant,
                        marker,
                        (
                            '{"prompt_version":"letter-v1-semif"}'
                            if family == "eikos"
                            else "# JPT-4B"
                            if family == "jpt"
                            else "{}"
                        ),
                    )
                    put(variant, "config.json")
                    put(variant, "model.safetensors")
                    put(variant, "tokenizer.json")
                    if family == "eikos":
                        with self.assertRaisesRegex(CheckpointError, "calib.json"):
                            load_model(variant)
                        put(variant, "calib.json")
                    marker_value = object()
                    with patch.object(
                        DecisionModel, f"from_{family}", return_value=marker_value
                    ) as loader:
                        self.assertIs(load_model(variant), marker_value)
                    loader.assert_called_once_with(str(variant))

    @unittest.skipIf(
        int(version("transformers").split(".")[0]) >= 5,
        "tiny Futurix tokenizer fixture uses Transformers 4",
    )
    def test_local_futurix_export_checks_metadata_and_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = root / "futurix"
            tiny_checkpoint().save_pretrained(saved)
            self.assertIsInstance(load_model(saved), DecisionModel)
            (saved / "decision_config.json").unlink()
            with self.assertRaisesRegex(CheckpointError, "decision_config.json"):
                load_model(saved, family="futurix")

    def test_local_family_override_and_missing_laya_or_gliner_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            put(root, "model.safetensors")
            with self.assertRaisesRegex(
                CheckpointError, "cannot identify local decision checkpoint"
            ):
                load_model(root)
            with self.assertRaisesRegex(CheckpointError, "rl_agent_config.json"):
                load_model(root, family="laya")
            with self.assertRaisesRegex(CheckpointError, "encoder_config/config.json"):
                load_model(root, family="gliner")

    def test_semantic_router_local_packages_use_nested_head_and_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            for version, family in ((1, "nox"), (2, "lux")):
                with self.subTest(version=version):
                    root = Path(temp) / f"renamed-{version}"
                    root.mkdir()
                    metadata = {
                        "decision_format": "vllm-sr-decision",
                        "format_version": version,
                        "model_name": f"Decision-{version}.0-{family.title()}-4B",
                        "model_config": "decision_config.json",
                        "decision_weights": {"decision_head": "decision_head.safetensors"},
                        "backbone": {
                            "config": "backbone/config.json",
                            "index": "backbone/model.safetensors.index.json",
                            "weights": ["backbone/model-00001-of-00001.safetensors"],
                        },
                        "tokenizer": {
                            "json": "tokenizer.json",
                            "config": "tokenizer_config.json",
                            "chat_template": "chat_template.jinja",
                        },
                        "calibration": {"temperature_file": "temperature.json"}
                        if version == 1
                        else None,
                        **({"manifest": "MODEL_MANIFEST.json"} if version == 2 else {}),
                    }
                    put(root, "config.json", json.dumps(metadata))
                    for filename in (
                        "decision_config.json",
                        "decision_head.safetensors",
                        "backbone/config.json",
                        "backbone/model-00001-of-00001.safetensors",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                    ):
                        put(root, filename)
                    put(
                        root,
                        "backbone/model.safetensors.index.json",
                        json.dumps(
                            {
                                "weight_map": {"layer": "model-00001-of-00001.safetensors"},
                            }
                        ),
                    )
                    put(root, "temperature.json" if version == 1 else "MODEL_MANIFEST.json")
                    self.assertEqual(infer_local_family(root), family)
                    require_checkpoint_files(root, family)
                    (root / "decision_head.safetensors").unlink()
                    with self.assertRaisesRegex(CheckpointError, "decision_head.safetensors"):
                        require_checkpoint_files(root, family)
                    put(root, "decision_head.safetensors")
                    (root / "backbone/model-00001-of-00001.safetensors").unlink()
                    with self.assertRaisesRegex(
                        CheckpointError,
                        "backbone/model-00001-of-00001.safetensors",
                    ):
                        require_checkpoint_files(root, family)

    def test_liquid_local_package_preserves_remote_code_and_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            put(
                root,
                "config.json",
                '{"model_type":"lfm2_vl","auto_map":{"AutoModel":"modeling_d1.D1Model"}}',
            )
            put(root, "model.safetensors")
            self.assertEqual(infer_local_family(root), "liquid-d1")
            with self.assertRaisesRegex(CheckpointError, "modeling_d1.py"):
                require_checkpoint_files(root, "liquid-d1")
            for filename in (
                "modeling_d1.py",
                "lfm2_vl.py",
                "runner.py",
                "hybrid.py",
                "api.py",
                "prompt.py",
                "tokenizer.json",
                "tokenizer_config.json",
                "processor_config.json",
                "chat_template.jinja",
            ):
                put(root, filename)
            require_checkpoint_files(root, "liquid-d1")
            (root / "tokenizer.json").unlink()
            with self.assertRaisesRegex(CheckpointError, "tokenizer.json"):
                require_checkpoint_files(root, "liquid-d1")

    def test_new_hub_releases_reject_incomplete_snapshots_before_model_execution(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("futurix_decision.models.native._requires_transformers"),
            patch("huggingface_hub.snapshot_download", return_value=temp) as download,
            patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=AssertionError("must not execute remote code on incomplete snapshots"),
            ),
        ):
            root = Path(temp)
            for name, version, size in (
                ("vllm-sr/Decision-1.0-Nox-4B", 1, "Nox-4B"),
                ("vllm-sr/Decision-2.0-Lux-9B", 2, "Lux-9B"),
            ):
                with self.subTest(name=name):
                    put(
                        root,
                        "config.json",
                        json.dumps(
                            {
                                "decision_format": "vllm-sr-decision",
                                "format_version": version,
                                "model_name": f"Decision-{version}.0-{size}",
                                "model_config": "decision_config.json",
                                **({"manifest": "MODEL_MANIFEST.json"} if version == 2 else {}),
                                "decision_weights": {"decision_head": "decision_head.safetensors"},
                                "backbone": {
                                    "config": "backbone/config.json",
                                    "index": "backbone/model.safetensors.index.json",
                                    "weights": ["backbone/model-00001-of-00001.safetensors"],
                                },
                                "tokenizer": {
                                    "json": "tokenizer.json",
                                    "config": "tokenizer_config.json",
                                    "chat_template": "chat_template.jinja",
                                },
                            }
                        ),
                    )
                    with self.assertRaisesRegex(
                        CheckpointError,
                        rf"{name}.*decision_head.safetensors",
                    ):
                        load_model(name, revision="a" * 40, trust_remote_code=True)
                    download.assert_called_with(name, revision="a" * 40)
            put(root, "config.json", '{"auto_map":{"AutoModel":"modeling_d1.D1Model"}}')
            with self.assertRaisesRegex(CheckpointError, r"LiquidAI/d1-3B.*modeling_d1.py"):
                load_model("liquid-d1", revision="b" * 40, trust_remote_code=True)

    def test_supported_remote_letter_release_uses_its_native_importer(self):
        with patch.object(DecisionModel, "from_decider", return_value="native-readout") as loader:
            self.assertEqual(load_model("Mapika/decider-4b"), "native-readout")
        loader.assert_called_once_with("Mapika/decider-4b")
        with self.assertRaisesRegex(ValueError, "unknown selected"):
            load_model("Cloudflare/clef-omni")

    def test_hub_futurix_export_is_detected_without_replacing_its_decision_head(self):
        with (
            patch(
                "futurix_decision.utils.config.load_decision_config",
                return_value={
                    "architecture": "futurix-decision",
                },
            ),
            patch.object(DecisionModel, "from_pretrained", return_value="loaded") as loader,
        ):
            self.assertEqual(load_model("example/typed-decisions"), "loaded")
        loader.assert_called_once_with("example/typed-decisions")
        with patch("futurix_decision.utils.config.load_decision_config", return_value={}):
            with self.assertRaisesRegex(CheckpointError, "decision_config.json"):
                load_model("example/typed-decisions", family="futurix")

    def test_generic_hub_conversion_routes_only_guarded_causal_architectures(self):
        raw = types.SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            is_encoder_decoder=False,
            vision_config=None,
        )
        converted = object()
        with (
            patch("futurix_decision.utils.config.load_decision_config", return_value={}),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=raw,
            ) as config,
            patch.object(
                DecisionModel,
                "from_pretrained",
                return_value=converted,
            ) as loader,
        ):
            self.assertIs(
                load_model("example/text-llm", convert=True, revision="a" * 40),
                converted,
            )
            config.assert_called_once_with("example/text-llm", revision="a" * 40)
            loader.assert_called_once_with(
                "example/text-llm",
                head_type="pointer",
                head_prompt_version="span-v1",
                train_backbone=False,
                revision="a" * 40,
            )
            loader.reset_mock()
            self.assertIs(
                load_model("example/text-llm", convert=True, strategy="bosun"),
                converted,
            )
            loader.assert_called_once_with(
                "example/text-llm",
                head_type="tokens",
                prompt_format="bosun-v3",
                bosun_template_fallback=True,
                train_backbone=True,
            )
            raw.architectures = ["BertForMaskedLM"]
            with self.assertRaisesRegex(CheckpointError, "text causal LM"):
                load_model("example/encoder", convert=True)
            raw.architectures = ["LlamaForCausalLM"]
            raw.is_encoder_decoder = True
            with self.assertRaisesRegex(CheckpointError, "text-only causal"):
                load_model("example/seq2seq", convert=True)
            raw.is_encoder_decoder = False
            raw.vision_config = object()
            with self.assertRaisesRegex(CheckpointError, "multimodal"):
                load_model("example/vision", convert=True)
            with self.assertRaisesRegex(ValueError, "40-character revision"):
                load_model("example/custom", convert=True, trust_remote_code=True)

    def test_qwen_multimodal_conversion_accepts_explicit_bosun_strategy(self):
        with (
            patch("futurix_decision.models.native._requires_transformers"),
            patch.object(
                DecisionModel,
                "from_pretrained",
                return_value="bosun-style",
            ) as loader,
        ):
            self.assertEqual(
                load_model("qwen3.5-9b", convert=True, strategy="bosun"),
                "bosun-style",
            )
            loader.assert_called_once_with(
                "Qwen/Qwen3.5-9B",
                head_type="tokens",
                prompt_format="bosun-v3",
                bosun_template_fallback=True,
                train_backbone=True,
            )
            with self.assertRaisesRegex(ValueError, "head_type='tokens'"):
                load_model(
                    "qwen3.5-9b",
                    convert=True,
                    strategy="bosun",
                    head_type="pointer",
                )

    def test_conversion_never_replaces_a_known_native_or_saved_decision_head(self):
        with self.assertRaisesRegex(ValueError, "discard its native readout"):
            load_model("Cloudflare/clef", convert=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            put(root, "head.pt")
            with self.assertRaisesRegex(ValueError, "discard its native readout"):
                load_model(root, convert=True)
        with patch(
            "futurix_decision.utils.config.load_decision_config",
            return_value={
                "architecture": "futurix-decision",
            },
        ):
            with self.assertRaisesRegex(ValueError, "already a trained Futurix"):
                load_model("example/trained", convert=True)


if __name__ == "__main__":
    unittest.main()
