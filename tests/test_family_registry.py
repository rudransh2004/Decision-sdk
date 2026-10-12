"""Family adapters: the registry that load_model and the CLI dispatch through."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from futurix_decision import (
    CheckpointError,
    ConfigError,
    DecisionResult,
    FamilyAdapter,
    load_model,
    register_family,
)
from futurix_decision.models.families import FAMILIES
from futurix_decision.models.families.publisher import PUBLISHER_CHECKPOINTS
from futurix_decision.models.registry import family_for_checkpoint, get_family

REGISTRY = "futurix_decision.models.registry._FAMILIES"


class UniformBackend:
    """Answers every decision with equal probabilities."""

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint, self.kwargs = checkpoint, kwargs

    def decide(self, example, *, confidence_threshold=None):
        share = 1 / len(example.options)
        return DecisionResult.from_probabilities(
            example,
            [share] * len(example.options),
            latency_ms=0.0,
            confidence_threshold=confidence_threshold,
        )

    def batch_decide(self, examples, *, confidence_threshold=None):
        return [self.decide(row, confidence_threshold=confidence_threshold) for row in examples]

    def choice(self, **kwargs):
        raise NotImplementedError

    def noul(self, **kwargs):
        raise NotImplementedError

    def score(self, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


class AcmeAdapter(FamilyAdapter):
    name = "acme"
    checkpoints = ("acme/decider-1",)

    def __init__(self):
        self.checked = []

    def load(self, checkpoint, *, device=None, **kwargs):
        return UniformBackend(checkpoint, device=device, **kwargs)

    def requires(self, path):
        self.checked.append(path)
        if not (path / "acme.json").is_file():
            raise CheckpointError(f"incomplete acme checkpoint {path}; missing: acme.json")


class RegistryTests(unittest.TestCase):
    def test_load_model_dispatches_to_a_registered_family(self):
        adapter = AcmeAdapter()
        with patch.dict(REGISTRY):
            register_family(adapter)
            model = load_model("acme/decider-1", device="cpu", revision="abc")
            self.assertIsInstance(model, UniformBackend)
            self.assertEqual(model.checkpoint, "acme/decider-1")
            self.assertEqual(model.kwargs, {"device": "cpu", "revision": "abc"})
            with self.assertRaisesRegex(ConfigError, "unknown selected"):
                load_model("acme/decider-2", family="acme")
            with self.assertRaisesRegex(ConfigError, "would discard its native readout"):
                load_model("acme/decider-1", convert=True)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with self.assertRaisesRegex(CheckpointError, "missing: acme.json"):
                    load_model(root, family="acme")
                (root / "acme.json").write_text("{}")
                self.assertIsInstance(load_model(root, family="acme"), UniformBackend)
                self.assertEqual(adapter.checked, [root, root])
        self.assertIsNone(family_for_checkpoint("acme/decider-1"))

    def test_register_family_rejects_duplicates_and_reserved_names(self):
        with patch.dict(REGISTRY):
            register_family(AcmeAdapter())
            with self.assertRaisesRegex(ConfigError, "already registered"):
                register_family(AcmeAdapter())
            for name in ("", "standard", "native"):
                adapter = AcmeAdapter()
                adapter.name = name
                with self.assertRaisesRegex(ConfigError, "invalid family name"):
                    register_family(adapter)
            with self.assertRaisesRegex(ConfigError, "FamilyAdapter"):
                register_family(object())
        with self.assertRaisesRegex(ConfigError, "unknown decision-model family 'acme'"):
            get_family("acme")

    def test_builtin_adapters_cover_every_implemented_checkpoint(self):
        for family in FAMILIES:
            for checkpoint in family.supported_checkpoints:
                adapter = family_for_checkpoint(checkpoint)
                self.assertIsNotNone(adapter, checkpoint)
                expected = PUBLISHER_CHECKPOINTS.get(checkpoint, family.name)
                self.assertEqual(adapter.name, expected)

    def test_default_finetune_and_training_options_refuse(self):
        adapter = AcmeAdapter()
        with self.assertRaises(NotImplementedError):
            adapter.finetune(UniformBackend("acme/decider-1"), [], "out")
        self.assertEqual(adapter.training_kwargs({"name": "acme/decider-1"}, method="full"), {})
        with self.assertRaisesRegex(ConfigError, "head_type"):
            adapter.training_kwargs({"head_type": "pointer"}, method="full")


if __name__ == "__main__":
    unittest.main()
