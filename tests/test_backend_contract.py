import importlib.util
import json
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import torch
from pydantic import ValidationError

from futurix_decision import (
    Calibrated,
    DecisionBackend,
    DecisionClient,
    DecisionInput,
    DecisionResult,
    NativeDecisionModel,
    evaluate,
    fit_temperature,
    load_model,
)
from futurix_decision.datasets import DecisionExample
from futurix_decision.datasets.benchmarks import normalize_decisionbench
from futurix_decision.errors import BackendError, ConfigError, InvalidOutputError
from futurix_decision.evaluation import fit_temperature_from_logits
from futurix_decision.models.families.openjev import server_calibration
from futurix_decision.vllm import VLLMEngine
from test_sft import tiny_model

CHOICE = DecisionInput(state="ticket", question="Next?", options=["approve", "reject"])


def rows(count=8):
    """Choice rows whose targets are softer than an overconfident 0.95/0.05 model."""
    return [
        DecisionExample(
            id=str(i),
            state=f"ticket {i}",
            question="Next?",
            options=["approve", "reject"],
            target=[0.7, 0.3] if i % 2 else [0.6, 0.4],
        )
        for i in range(count)
    ]


class Overconfident:
    """A Laya-like publisher runtime that puts 0.95 on the first option."""

    def predict(self, state, questions):
        question = questions["decision"]
        if question["type"] == "noul":
            return {"answers": {"decision": {"type": "noul", "noul": 0.95}}}
        keys = (
            [str(i) for i in range(len(question["criteria"]))]
            if question["type"] == "score"
            else list(question["criteria"])
        )
        values = {key: (0.95 if i == 0 else 0.05 / (len(keys) - 1)) for i, key in enumerate(keys)}
        return {"answers": {"decision": {"type": question["type"], "probabilities": values}}}


def fake_vllm_engine(directory):
    model = tiny_model()
    model.save_pretrained(directory)

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            logprobs=[
                                {
                                    token: SimpleNamespace(logprob=math.log(0.5))
                                    for token in params.allowed_token_ids
                                }
                            ]
                        )
                    ]
                )
                for params in sampling_params
            ]

    return VLLMEngine(directory, llm=FakeLLM(), tokenizer=model.tokenizer)


def sampling_params(**kwargs):
    return SimpleNamespace(**kwargs)


class ProtocolTests(unittest.TestCase):
    def test_every_backend_satisfies_the_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            backends = [
                tiny_model(),
                NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", Overconfident()),
                fake_vllm_engine(temp),
                DecisionClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
            ]
            backends.append(Calibrated(backends[0], 2.0))
            for backend in backends:
                with self.subTest(backend=type(backend).__name__):
                    self.assertIsInstance(backend, DecisionBackend)
                    backend.close()
        self.assertNotIsInstance(object(), DecisionBackend)

    def test_latency_is_per_row_for_batches(self):
        model = tiny_model()
        with patch(
            "futurix_decision.models.decision_model.time.perf_counter", side_effect=[10.0, 11.0]
        ):
            results = model.batch_decide([CHOICE] * 4)
        self.assertEqual([result.latency_ms for result in results], [250.0] * 4)
        with tempfile.TemporaryDirectory() as temp:
            engine = fake_vllm_engine(temp)
            with (
                patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=sampling_params)}),
                patch("futurix_decision.vllm.engine.time.perf_counter", side_effect=[1.0, 1.5]),
            ):
                results = engine.batch_decide([CHOICE] * 2)
        self.assertEqual([result.latency_ms for result in results], [250.0] * 2)


class DecisionTypeTests(unittest.TestCase):
    def test_decision_type_follows_kind_even_when_order_is_preserved(self):
        model = NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", Overconfident())
        self.assertIs(model.noul(state="s").decision, True)
        self.assertEqual(model.score(state="s", min_score=2, max_score=4).decision, 2)
        self.assertEqual(model.decide(CHOICE).decision, "approve")
        local = tiny_model()
        reversed_noul = DecisionInput(
            kind="noul",
            state="s",
            question="Valid?",
            options=["true", "false"],
            candidate_ids=["true", "false"],
            preserve_order=True,
        )
        self.assertIsInstance(local.decide(reversed_noul).decision, bool)
        self.assertIsInstance(local.score(state="s").decision, int)

    def test_opaque_benchmark_ids_have_no_typed_decision(self):
        row = normalize_decisionbench(
            {
                "row_id": "gate-1",
                "primitive": "binary_classification",
                "instruction": "Approve?",
                "state_json": '{"call": "delete"}',
                "candidates_json": json.dumps(
                    [{"id": "c000", "label": "Reject"}, {"id": "c001", "label": "Approve"}]
                ),
                "gold_candidate_id": "c000",
            }
        )
        result = DecisionResult.from_probabilities(row, [0.8, 0.2], latency_ms=1.0)
        self.assertIsNone(result.decision)
        self.assertFalse(result.abstained)
        self.assertEqual(result.kind, "noul")
        self.assertEqual(result.probabilities, {"c000": 0.8, "c001": 0.2})

    def test_result_rejects_probabilities_outside_zero_to_one(self):
        row = DecisionInput(kind="noul", state="s", question="q", options=["false", "true"])
        for probabilities in (
            [math.nan, 0.5],
            [0.5, math.nan],
            [math.inf, 0.0],
            [-0.2, 1.2],
            [0.0, 1.5],
        ):
            for threshold in (None, 0.95):
                with (
                    self.subTest(probabilities=probabilities, threshold=threshold),
                    self.assertRaises(InvalidOutputError),
                ):
                    DecisionResult.from_probabilities(
                        row, probabilities, latency_ms=0.0, confidence_threshold=threshold
                    )
        self.assertEqual(
            DecisionResult.from_probabilities(row, [0.0, 1.0], latency_ms=0.0).decision, True
        )

    def test_result_rejects_a_decision_of_the_wrong_type(self):
        common = {"confidence": 0.9, "probabilities": {}, "latency_ms": 1.0}
        for kind, decision in (("noul", "true"), ("score", True), ("score", "3"), ("choice", 1)):
            with self.subTest(kind=kind, decision=decision), self.assertRaises(ValidationError):
                DecisionResult(kind=kind, decision=decision, abstained=False, **common)
        with self.assertRaises(ValidationError):
            DecisionResult(kind="noul", decision=True, abstained=True, **common)
        self.assertIs(
            DecisionResult(kind="noul", decision=False, abstained=False, **common).decision, False
        )

    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
    def test_server_json_carries_kind_and_typed_decision(self):
        from fastapi.testclient import TestClient

        from futurix_decision.vllm.api import create_app

        http = TestClient(create_app(tiny_model()))
        noul = http.post("/v1/decision/noul", json={"state": "ready"}).json()
        score = http.post("/v1/decision/score", json={"state": "ready"}).json()
        self.assertEqual(noul["kind"], "noul")
        self.assertIs(type(noul["decision"]), bool)
        self.assertIs(type(score["decision"]), int)
        with DecisionClient(http_client=http) as client:
            self.assertIs(type(client.noul(state="ready").decision), bool)

    def test_client_rejects_a_result_of_another_kind(self):
        body = {
            "kind": "choice",
            "decision": "approve",
            "confidence": 0.7,
            "probabilities": {"false": 0.3, "true": 0.7},
            "abstained": False,
            "latency_ms": 1.0,
        }
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        with (
            DecisionClient(transport=transport) as client,
            self.assertRaisesRegex(InvalidOutputError, "choice decision to a noul request"),
        ):
            client.noul(state="s")


class CalibrationTests(unittest.TestCase):
    def test_calibrated_matches_logit_temperature_exactly(self):
        model = tiny_model()
        noul = DecisionInput(kind="noul", state="s", question="q", options=["false", "true"])
        calibrated = Calibrated(model, 2.5).batch_decide([CHOICE, noul])
        model.config.temperature = 2.5
        scaled = model.batch_decide([CHOICE, noul])
        for expected, actual in zip(scaled, calibrated, strict=True):
            self.assertEqual(expected.decision, actual.decision)
            for option, value in expected.probabilities.items():
                self.assertAlmostEqual(actual.probabilities[option], value, places=6)

    def test_calibrated_recomputes_abstention_from_calibrated_confidence(self):
        native = NativeDecisionModel.from_runtime("convaiinnovations/laya", "laya", Overconfident())
        self.assertFalse(native.decide(CHOICE, confidence_threshold=0.9).abstained)
        softened = Calibrated(native, 4.0).decide(CHOICE, confidence_threshold=0.9)
        self.assertTrue(softened.abstained)
        self.assertIsNone(softened.decision)
        self.assertLess(softened.confidence, 0.9)
        self.assertAlmostEqual(sum(softened.probabilities.values()), 1.0)

    def test_fit_on_probabilities_equals_fit_on_logits(self):
        model = tiny_model()
        data = rows()
        batch = model.forward_decisions(data)
        logits = [batch.logits[i, slots].detach() for i, slots in enumerate(batch.slots)]
        targets = [torch.tensor(row.target_distribution()) for row in data]
        expected = fit_temperature_from_logits(logits, targets)
        self.assertAlmostEqual(fit_temperature(model, data), expected, places=4)

    def test_calibration_applies_the_zero_probabilities_the_fit_assumed(self):
        from futurix_decision.models.protocol import DecisionShortcuts

        class Certain(DecisionShortcuts):
            """Puts all mass on the first option; the second underflowed to 0."""

            def decide(self, example, *, confidence_threshold=None):
                return DecisionResult.from_probabilities(example, [1.0, 0.0], latency_ms=0.0)

            def batch_decide(self, examples, *, confidence_threshold=None):
                return [self.decide(example) for example in examples]

            def close(self):
                pass

        temperature = fit_temperature(Certain(), rows())
        self.assertGreater(temperature, 1.0)
        plain = evaluate(Certain(), rows())
        calibrated = evaluate(Calibrated(Certain(), temperature), rows())
        # The fit softened the zero; calibrating with its temperature must too.
        self.assertLess(calibrated["nll"], plain["nll"])

    def test_load_fit_and_evaluate_a_local_model(self):
        with tempfile.TemporaryDirectory() as temp:
            tiny_model().save_pretrained(temp)
            model = load_model(temp, device="cpu")
            with self.assertRaisesRegex(ConfigError, "device or device_map"):
                load_model(temp, device="cpu", device_map={"": "cpu"})
            temperature = fit_temperature(model, rows())
            self.assertGreater(temperature, 0)
            plain = evaluate(model, rows())
            calibrated = evaluate(Calibrated(model, temperature), rows())
            self.assertLessEqual(calibrated["nll"], plain["nll"] + 1e-6)

    def test_load_fit_and_evaluate_a_native_http_backend(self):
        runtime = Overconfident()

        def serve(request):
            body = json.loads(request.content)
            return httpx.Response(200, json=runtime.predict(body["state"], body["questions"]))

        client = httpx.Client(base_url="http://laya.test", transport=httpx.MockTransport(serve))
        with self.assertRaisesRegex(ConfigError, "head_type applies"):
            load_model("laya", base_url="http://laya.test", client=client, head_type="pointer")
        model = load_model("laya", base_url="http://laya.test", client=client)
        temperature = fit_temperature(model, rows())
        # 0.95 against targets near 0.65 is overconfident: the fit softens it.
        self.assertGreater(temperature, 1.5)
        plain = evaluate(model, rows())
        calibrated = evaluate(Calibrated(model, temperature), rows())
        self.assertLess(calibrated["nll"], plain["nll"])

    def test_openjev_server_calibration_is_pinned(self):
        flash = server_calibration("openjev/OpenJev-Flash-9B")
        base = server_calibration("openjev/openjev")
        self.assertEqual((flash.temperature, flash.noul_temperature), (1.07, 1.074766))
        self.assertEqual((base.temperature, base.noul_temperature), (0.85, 1.829074))
        # A local copy is matched by its directory name, as the server reports it.
        self.assertEqual(server_calibration("/models/OpenJev-Flash-9B/"), flash)

        def transport(temperature):
            def handle(request):
                if request.url.path == "/v1/version":
                    return httpx.Response(
                        200,
                        json={
                            "model_dir": "openjev",
                            "T": temperature,
                            "noul_t": 1.829074,
                            "noul_bias": 0,
                            "flags": {"targeted": True, "instr_style": "pyrepr"},
                        },
                    )
                return httpx.Response(
                    200, json={"answers": {"decision": {"type": "noul", "noul": 0.83}}}
                )

            return httpx.Client(base_url="http://jev.test", transport=httpx.MockTransport(handle))

        model = load_model("openjev", base_url="http://jev.test", client=transport(0.85))
        # The server applies the published temperatures; the SDK must not apply them again.
        self.assertEqual(model.noul(state="paid").probabilities, {"false": 1 - 0.83, "true": 0.83})
        with self.assertRaisesRegex(BackendError, "does not serve"):
            load_model("openjev", base_url="http://jev.test", client=transport(0.851))


if __name__ == "__main__":
    unittest.main()
