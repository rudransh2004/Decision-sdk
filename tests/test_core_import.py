"""The core package must import and work without torch or transformers."""

import subprocess
import sys
import textwrap
import unittest

# Runs in a fresh interpreter with torch and transformers made unimportable, so
# this proves the core is torch-free even in an environment that has torch.
SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {"torch", "transformers"}

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, Block())

    import futurix_decision
    from futurix_decision import DecisionInput, DecisionResult, MissingDependencyError
    from futurix_decision.evaluation import compute_metrics
    from futurix_decision.utils.config import load_config

    example = DecisionInput(state="sky", question="Which color?", options=["blue", "red"])
    assert example.options == ["blue", "red"]
    result = DecisionResult(
        decision="blue",
        confidence=0.9,
        probabilities={"blue": 0.9, "red": 0.1},
        abstained=False,
        latency_ms=1.0,
    )
    assert result.to_dict()["decision"] == "blue"
    assert compute_metrics([[0.9, 0.1]], [[1.0, 0.0]])["accuracy"] == 1.0

    for name in ("DecisionModel", "SFTTrainer", "evaluate", "fit_temperature"):
        try:
            getattr(futurix_decision, name)
        except MissingDependencyError as error:
            assert "futurix-decision[torch]" in str(error), error
            assert isinstance(error, ImportError)
        else:
            raise AssertionError(f"{name} imported without torch")

    # Native runtimes (HTTP, publisher packages) do not need torch to load.
    assert callable(futurix_decision.load_model)
    assert futurix_decision.NativeDecisionModel.__name__ == "NativeDecisionModel"

    # A transformers checkpoint names the missing extra instead of failing on `import torch`.
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "decision_config.json").write_text(json.dumps({"architecture": "futurix-decision"}))
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (root / name).write_text("{}")
        try:
            futurix_decision.load_model(root)
        except MissingDependencyError as error:
            assert "futurix-decision[torch]" in str(error), error
        else:
            raise AssertionError("load_model loaded a transformers checkpoint without torch")

    leaked = sorted(m for m in sys.modules if m.partition(".")[0] in BLOCKED)
    assert not leaked, leaked
    print("ok")
    """
)


class CoreImportTests(unittest.TestCase):
    def test_core_imports_without_torch(self):
        completed = subprocess.run(
            [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ok")

    def test_torch_backed_names_resolve_with_torch(self):
        import futurix_decision
        from futurix_decision.models.decision_model import DecisionModel

        self.assertIs(futurix_decision.DecisionModel, DecisionModel)
        with self.assertRaises(AttributeError):
            futurix_decision.NotAName  # noqa: B018


if __name__ == "__main__":
    unittest.main()
