import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from futurix_decision import DecisionInput, NativeDecisionModel, load_model
from futurix_decision.models.native import _question


CHOICE = DecisionInput(
    state={"ticket": "duplicate charge"}, question="Which team?",
    options=["billing", "technical"],
    candidate_descriptions=["Refunds and charges", "Software bugs"],
)


def answer(state, questions):
    return {"answers": {"decision": {
        "type": questions["decision"]["type"],
        "probabilities": {"billing": 0.8, "technical": 0.2},
        "choice": "billing",
    }}}


class NativeInferenceTests(unittest.TestCase):
    def test_shared_choice_noul_score_interface(self):
        class Agent:
            def predict(self, state, questions):
                if questions["decision"]["type"] == "choice":
                    return answer(state, questions)
                if questions["decision"]["type"] == "noul":
                    return {"answers": {"decision": {"type": "noul", "noul": 0.92}}}
                return {"answers": {"decision": {
                    "type": "score", "score": 0.9,
                    "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1},
                }}}

        model = NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", Agent())
        self.assertEqual(model.decide(CHOICE).decision, "billing")
        abstained = model.choice(
            state="A ticket", question="Which team?", choices=["billing", "technical"],
            confidence_threshold=0.85,
        )
        self.assertIsNone(abstained.decision)
        self.assertTrue(abstained.abstained)
        self.assertIs(model.noul(state="Text").decision, True)
        rating = model.score(state="Text", min_score=2, max_score=4)
        self.assertEqual(rating.decision, 3)
        self.assertEqual(list(rating.probabilities), ["2", "3", "4"])
        self.assertEqual(len(model.batch_decide([CHOICE, CHOICE])), 2)
        self.assertTrue(model.capabilities.training)
        self.assertFalse(model.capabilities.vllm_inference)

    def test_publisher_probabilities_cannot_be_invented(self):
        for values in (
            {"choice": "billing"},
            {"probabilities": {"billing": 0.6, "other": 0.4}},
            {"probabilities": {"billing": 0.3, "technical": 0.3}},
        ):
            class Agent:
                def predict(self, _state, _questions):
                    return {"answers": {"decision": {"type": "choice", **values}}}
            with self.subTest(values=values), self.assertRaises(ValueError):
                NativeDecisionModel.from_runtime(
                    "convaiinnovations/laya", "laya", Agent()
                ).decide(CHOICE)

    def test_http_openjev_verifies_checkpoint_and_calibration(self):
        def transport(request):
            if request.url.path == "/v1/version":
                return httpx.Response(200, json={
                    "model_dir": "OpenJev-Flash-9B", "T": 1.07,
                    "noul_t": 1.074766, "noul_bias": 0,
                    "flags": {"targeted": True, "instr_style": "pyrepr"},
                })
            assert request.url.path == "/v1/systemone"
            body = json.loads(request.content)
            self.assertEqual(body["questions"]["decision"]["criteria"]["billing"], "Refunds and charges")
            return httpx.Response(200, json=answer(body["state"], body["questions"]))

        client = httpx.Client(base_url="http://localhost:3000", transport=httpx.MockTransport(transport))
        model = load_model("openjev-flash", base_url="http://localhost:3000", client=client)
        self.assertEqual(model.decide(CHOICE).decision, "billing")
        self.assertTrue(model.capabilities.vllm_inference)
        with self.assertRaisesRegex(ValueError, "does not serve"):
            load_model("openjev", base_url="http://localhost:3000", client=client)
        with self.assertRaisesRegex(ValueError, "base_url"):
            load_model("openjev")

    def test_gliner_uses_exclusive_schema_probabilities(self):
        class Schema:
            def single(self, task, labels, instruction):
                self.labels, self.instruction = labels, instruction
                return self

        class Result:
            def __getitem__(self, name):
                return types.SimpleNamespace(exclusive=True)

            def probabilities(self, name):
                return {"billing": 0.85, "technical": 0.15}

        class Classifier:
            def classify(self, text, schema):
                self.text, self.schema = text, schema
                return Result()

        fake = types.ModuleType("gliner2.classification")
        fake.ClassificationSchema = Schema
        classifier = Classifier()
        with patch.dict("sys.modules", {"gliner2": types.ModuleType("gliner2"),
                                        "gliner2.classification": fake}):
            model = NativeDecisionModel.from_runtime(
                "fastino/GLiNER2.5-Decide", "gliner", classifier,
            )
            self.assertEqual(model.decide(CHOICE).confidence, 0.85)
            self.assertEqual(classifier.schema.instruction, "Which team?")
            self.assertEqual(classifier.schema.labels["billing"], "Refunds and charges")

    def test_gliner_moves_extractor_to_requested_device(self):
        class Classifier:
            @staticmethod
            def from_pretrained(checkpoint, *, device, revision):
                self.assertEqual(checkpoint, "fastino/GLiNER2.5-Decide")
                self.assertEqual(device, "cuda:1")
                self.assertEqual(revision, "a" * 40)
                return types.SimpleNamespace(to=lambda *, device: moved.append(device))

        moved = []
        package = types.ModuleType("gliner2")
        package.__path__ = []
        classification = types.ModuleType("gliner2.classification")
        classification.Classifier = Classifier
        with patch.dict("sys.modules", {"gliner2": package, "gliner2.classification": classification}):
            model = load_model(
                "gliner", device="cuda:1", revision="a" * 40,
            )
        self.assertEqual(model.family, "gliner")
        self.assertEqual(moved, ["cuda:1"])

    def test_other_native_head_dispatch_and_remote_code_guard(self):
        for family in ("nox", "lux", "liquid-d1"):
            runtime = types.SimpleNamespace(system_one=answer)
            model = NativeDecisionModel.from_runtime("checkpoint", family, runtime)
            self.assertEqual(model.decide(CHOICE).decision, "billing")
            self.assertFalse(model.capabilities.training)
            with self.assertRaises(NotImplementedError):
                model.finetune([], "unused")
        pplx = NativeDecisionModel.from_runtime(
            "perplexity-ai/pplx-decider-v1.1-27b", "pplx-decider",
            types.SimpleNamespace(predict=lambda records: [[0.8, 0.2]]),
        )
        self.assertEqual(pplx.decide(CHOICE).decision, "billing")
        clef = NativeDecisionModel.from_runtime(
            "Cloudflare/clef-flash", "clef",
            (types.SimpleNamespace(systemone=lambda *_args: answer(None, {"decision": {"type": "choice"}})),
             object(), object()),
        )
        self.assertEqual(clef.decide(CHOICE).decision, "billing")
        self.assertTrue(clef.capabilities.training)
        with self.assertRaisesRegex(ValueError, "40-character revision"):
            load_model("nox", trust_remote_code=True, revision="main")
        with self.assertRaisesRegex(ValueError, "not decision-trained"):
            load_model("qwen3.5-9b")

    def test_candidate_and_verbalizer_runtimes_require_keyword_only_system_one(self):
        class Publisher:
            def system_one(self, *, state, questions, images=None):
                self.state, self.questions, self.images = state, questions, images
                return {"answers": {
                    qid: {
                        "type": "choice",
                        "probabilities": {"billing": 0.8, "technical": 0.2},
                    } for qid in questions
                }}

        for family in ("nox", "lux", "liquid-d1"):
            with self.subTest(family=family):
                publisher = Publisher()
                model = NativeDecisionModel.from_runtime("checkpoint", family, publisher)
                self.assertEqual(model.decide(CHOICE).decision, "billing")
                grouped = model.system_one(
                    "Charged twice",
                    {"route": {"type": "choice", "instructions": "Who handles this?",
                               "criteria": {"billing": "Payments", "technical": "App"}}},
                )
                self.assertEqual(grouped["answers"]["route"]["probabilities"]["billing"], 0.8)
                self.assertEqual(publisher.state, "Charged twice")
                self.assertEqual(list(publisher.questions), ["route"])

    def test_native_question_does_not_reorder_choices(self):
        request = _question(CHOICE)
        self.assertEqual(list(request["criteria"]), CHOICE.options)
        with self.assertRaisesRegex(ValueError, "explicit false/true"):
            NativeDecisionModel.from_runtime(
                "convaiinnovations/laya", "laya",
                types.SimpleNamespace(predict=lambda *_: {}),
            ).decide(DecisionInput(
                kind="noul", state="text", question="Correct?",
                options=["a", "b"], candidate_ids=["a", "b"], preserve_order=True,
            ))

    def test_only_media_capable_native_models_receive_images(self):
        class Vision:
            def system_one(self, state, questions, images=None):
                self.images = images
                return answer(state, questions)

        vision = Vision()
        row = CHOICE.model_copy(update={"images": ["./screenshot.png"]})
        with patch("futurix_decision.models.native._images_for_native", return_value=["loaded image"]):
            self.assertEqual(
                NativeDecisionModel.from_runtime("LiquidAI/d1-3B", "liquid-d1", vision)
                .decide(row).decision, "billing",
            )
        self.assertEqual(vision.images, ["loaded image"])
        with self.assertRaisesRegex(ValueError, "does not expose an image"):
            NativeDecisionModel.from_runtime(
                "convaiinnovations/laya", "laya", types.SimpleNamespace(predict=answer)
            ).decide(row)

    def test_grouped_native_request_keeps_multiple_questions_one_call(self):
        class Agent:
            calls = 0

            def predict(self, state, questions):
                self.calls += 1
                self.questions = questions
                return {"answers": {
                    "route": {"type": "choice", "choice": "billing",
                              "probabilities": {"billing": 0.7, "other": 0.3}},
                    "urgent": {"type": "noul", "noul": 0.4},
                }}

        native = Agent()
        model = NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", native)
        questions = {
            "route": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "Finance", "other": "Support"}},
            "urgent": {"type": "noul", "instructions": "Urgent?"},
        }
        response = model.system_one("Charged twice", questions)
        self.assertEqual(native.calls, 1)
        self.assertEqual(native.questions, questions)
        self.assertEqual(response["answers"]["urgent"]["noul"], 0.4)

    def test_native_batch_paths_preserve_input_order(self):
        class BatchedLaya:
            calls = 0

            def predict_batch(self, states, questions):
                self.calls += 1
                return [answer(state, questions) for state in states]

        laya = BatchedLaya()
        model = NativeDecisionModel.from_runtime(
            "convaiinnovations/laya", "laya", laya, backend="local",
        )
        rows = [CHOICE, CHOICE.model_copy(update={"state": "different ticket"})]
        self.assertEqual([r.decision for r in model.batch_decide(rows)], ["billing", "billing"])
        self.assertEqual(laya.calls, 1)

        class BatchedPplx:
            calls = 0

            def predict(self, records):
                self.calls += 1
                return [[0.8, 0.2], [0.3, 0.7]]

        pplx = BatchedPplx()
        model = NativeDecisionModel.from_runtime(
            "perplexity-ai/pplx-decider-v1.1-27b", "pplx-decider", pplx,
            backend="local",
        )
        rows = [
            CHOICE, DecisionInput(
                kind="noul", state="Ready", question="Is it ready?",
                options=["false", "true"],
            ),
        ]
        results = model.batch_decide(rows)
        self.assertEqual([r.decision for r in results], ["billing", True])
        self.assertEqual(pplx.calls, 1)


if __name__ == "__main__":
    unittest.main()
