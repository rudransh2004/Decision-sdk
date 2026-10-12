import importlib.util
import json
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from futurix_decision import (
    DecisionBackend,
    DecisionClient,
    DecisionInput,
    DecisionResult,
    NativeDecisionModel,
)
from futurix_decision.datasets.benchmarks import normalize_decisionbench
from futurix_decision.errors import InvalidOutputError
from futurix_decision.vllm import VLLMEngine
from test_sft import tiny_model

CHOICE = DecisionInput(state="ticket", question="Next?", options=["approve", "reject"])


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


if __name__ == "__main__":
    unittest.main()
