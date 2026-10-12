import contextlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from futurix_decision import NativeDecisionModel
from futurix_decision.cli import main
from test_native_models import answer


class NativeCLITests(unittest.TestCase):
    def test_native_evaluation_uses_native_inference(self):
        class Agent:
            def predict(self, state, questions):
                return answer(state, questions)

        native = NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", Agent())
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "eval.jsonl"
            dataset.write_text(json.dumps({
                "state": "Duplicate charge", "question": "Which team?",
                "options": ["billing", "technical"], "label": "billing",
            }) + "\n")
            with patch("futurix_decision.cli.load_model", return_value=native), \
                    contextlib.redirect_stdout(io.StringIO()) as captured:
                main(["evaluate", "--model", "laya", "--model-format", "native",
                      "--dataset", str(dataset), "--max-rows", "1"])
            self.assertEqual(json.loads(captured.getvalue())["count"], 1)

    def test_native_training_uses_native_checkpoint_export(self):
        native = NativeDecisionModel.from_runtime(
            "convaiinnovations/laya", "laya", types.SimpleNamespace(),
        )
        output = types.SimpleNamespace(
            family="laya", training_report={"train_items": 2},
        )
        with tempfile.TemporaryDirectory() as temp, \
                patch("futurix_decision.cli.load_model", return_value=native), \
                patch.object(native, "finetune", return_value=output) as finetune, \
                contextlib.redirect_stdout(io.StringIO()) as captured:
            dataset = Path(temp) / "train.jsonl"
            dataset.write_text('{"state":"s","question":"q","options":["x","y"],"label":"x"}\n')
            main(["train", "--model", "laya", "--model-format", "native",
                  "--dataset", str(dataset), "--output", str(Path(temp) / "model")])
            self.assertEqual(json.loads(captured.getvalue())["family"], "laya")
            finetune.assert_called_once_with(str(dataset), str(Path(temp) / "model"))


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class NativeServingTests(unittest.TestCase):
    def test_native_systemone_keeps_question_contract(self):
        from fastapi.testclient import TestClient
        from futurix_decision.vllm.api import create_app

        class Agent:
            def predict(self, state, questions):
                self.question = questions["route"]
                return {"answers": {
                    "route": {"type": "choice", "choice": "billing",
                              "probabilities": {"billing": 0.9, "support": 0.1}},
                }}

        runtime = Agent()
        client = TestClient(create_app(NativeDecisionModel.from_runtime(
            "convaiinnovations/laya", "laya", runtime,
        )))
        payload = {"state": "Card charged twice", "questions": {"route": {
            "type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "Refunds", "support": "Other"},
        }}}
        response = client.post("/v1/systemone", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(runtime.question["instructions"], "Which team?")
        self.assertEqual(runtime.question["criteria"]["billing"], "Refunds")
        self.assertEqual(response.json()["answers"]["route"]["choice"], "billing")
        self.assertEqual(client.post(
            "/v1/systemone", json={**payload, "prompt_profile": "semantic"},
        ).status_code, 422)


if __name__ == "__main__":
    unittest.main()
