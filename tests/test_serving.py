import importlib.util
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

from futurix_decision.datasets import DecisionInput
from futurix_decision.inference.schemas import DecisionRequest
from futurix_decision.vllm.logits_processor import probabilities_from_logprobs


class SchemaTests(unittest.TestCase):
    def test_api_schema_and_abstention_validation(self):
        row = DecisionRequest(
            state={"expired": True},
            question="Next?",
            choices=["approve", "reject"],
            confidence_threshold=0.8,
        )
        self.assertEqual(row.to_input().options, ["approve", "reject"])
        with self.assertRaises(ValueError):
            DecisionRequest(state="state", choices=["a", "a"])
        with self.assertRaises(ValueError):
            DecisionRequest(state="state", choices=["a", "b"], confidence_threshold=1.5)

    def test_restricted_logprob_normalization(self):
        result = probabilities_from_logprobs({12: -1.0, 31: -2.0}, [12, 31], temperature=2.0)
        self.assertAlmostEqual(result[0], 1 / (1 + math.exp(-0.5)), places=6)
        with self.assertRaisesRegex(ValueError, "omitted"):
            probabilities_from_logprobs({12: -1.0}, [12, 31])


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from futurix_decision.vllm.api import create_app
        from test_sft import tiny_model

        cls.client = TestClient(create_app(tiny_model()))

    def test_endpoints_and_systemone(self):
        payload = {
            "state": {"gst_certificate": "expired"},
            "question": "Next?",
            "choices": ["approve", "reject"],
        }
        choice = self.client.post("/v1/decision/choice", json=payload)
        self.assertEqual(choice.status_code, 200, choice.text)
        self.assertAlmostEqual(sum(choice.json()["probabilities"].values()), 1.0, places=6)
        self.assertEqual(self.client.post("/v1/decision", json=payload).status_code, 200)
        self.assertEqual(
            self.client.post("/v1/decision/noul", json={"state": "ready"}).status_code, 200
        )
        self.assertEqual(
            self.client.post("/v1/decision/score", json={"state": "ready"}).status_code, 200
        )
        self.assertEqual(self.client.post("/v1/decision", json={"state": "ready"}).status_code, 422)
        abstain = self.client.post(
            "/v1/decision/choice", json={**payload, "confidence_threshold": 1.0}
        ).json()
        self.assertTrue(abstain["abstained"])
        self.assertIsNone(abstain["decision"])
        response = self.client.post(
            "/v1/systemone",
            json={
                "model": "Futurix-Decision-1.7B",
                "state": "certificate expired",
                "questions": {
                    "approve": {
                        "type": "choice",
                        "instructions": "Next?",
                        "criteria": {
                            "request_document": "Ask for current certificate",
                            "reject": "Stop",
                        },
                    },
                    "valid": {"type": "noul", "instructions": "Is this valid?"},
                    "rating": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": ["bad", "ok", "good"],
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(set(response.json()["answers"]), {"approve", "valid", "rating"})
        semantic = self.client.post(
            "/v1/systemone",
            json={
                "state": ["invoice", "expired certificate"],
                "prompt_profile": "semantic",
                "demonstration_format": "compact",
                "questions": {"valid": {"type": "noul", "instructions": "Valid?"}},
            },
        )
        self.assertEqual(semantic.status_code, 200, semantic.text)
        self.assertEqual(semantic.json()["prompt_profile"], "semantic")
        self.assertEqual(
            self.client.post(
                "/v1/systemone",
                json={"state": "state", "questions": {"x": {"type": "noul"}}, "media": ["image"]},
            ).status_code,
            422,
        )
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertIn('decision_requests_total{kind="choice"}', self.client.get("/metrics").text)

    @unittest.skipUnless(
        os.environ.get("FUTURIX_TEST_SERVER") == "1", "opt-in HTTP server smoke test"
    )
    def test_cli_server_over_http(self):
        import httpx

        from test_sft import tiny_model

        with tempfile.TemporaryDirectory() as temp:
            tiny_model().save_pretrained(temp)
            with socket.socket() as address:
                address.bind(("127.0.0.1", 0))
                port = address.getsockname()[1]
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "futurix_decision.cli",
                    "serve",
                    "--model",
                    temp,
                    "--device",
                    "cpu",
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                for _ in range(80):
                    if process.poll() is not None:
                        raise AssertionError(process.stderr.read().decode())
                    try:
                        response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.3)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        time.sleep(0.1)
                else:
                    self.fail("decision server did not start")
                decision = httpx.post(
                    f"http://127.0.0.1:{port}/v1/decision/choice",
                    json={"state": "ready", "choices": ["approve", "reject"]},
                    timeout=10,
                )
                self.assertEqual(decision.status_code, 200, decision.text)
            finally:
                process.terminate()
                process.wait(timeout=10)


@unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM extra not installed")
class VLLMUnitTests(unittest.TestCase):
    def test_one_restricted_logprob_step_per_batched_prompt(self):
        from futurix_decision.vllm.engine import VLLMEngine
        from test_sft import tiny_model

        class FakeLLM:
            calls = 0

            def generate(self, prompts, sampling_params, use_tqdm=False):
                self.calls += 1
                assert len(prompts) == len(sampling_params) == 2
                assert all(
                    params.max_tokens == 1 and params.logprobs == len(params.allowed_token_ids)
                    for params in sampling_params
                )
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                logprobs=[
                                    {
                                        token: SimpleNamespace(logprob=math.log(probability))
                                        for token, probability in zip(
                                            params.allowed_token_ids, [0.8, 0.2]
                                        )
                                    }
                                ]
                            )
                        ]
                    )
                    for params in sampling_params
                ]

        with tempfile.TemporaryDirectory() as temp:
            model = tiny_model()
            model.save_pretrained(temp)
            llm = FakeLLM()
            engine = VLLMEngine(temp, llm=llm, tokenizer=model.tokenizer)
            rows = [
                DecisionInput(state=str(i), question="Next?", options=["approve", "reject"])
                for i in range(2)
            ]
            results = engine.batch_decide(rows)
            self.assertEqual(llm.calls, 1)
            self.assertEqual(results[0].decision, "approve")
            self.assertAlmostEqual(results[1].confidence, 0.8, places=6)

    @unittest.skipUnless(os.environ.get("FUTURIX_TEST_VLLM") == "1", "opt-in GPU vLLM smoke test")
    def test_real_vllm_engine_on_local_checkpoint(self):
        from futurix_decision.vllm.engine import VLLMEngine
        from test_sft import tiny_untied_model

        with tempfile.TemporaryDirectory() as temp:
            tiny_untied_model().save_pretrained(temp)
            from futurix_decision import DecisionModel

            reference = DecisionModel.from_pretrained(temp).choice(
                state="ready", choices=["approve", "reject"]
            )
            engine = VLLMEngine(
                temp,
                gpu_memory_utilization=0.04,
                enforce_eager=True,
                max_model_len=256,
                dtype="float16",
            )
            result = engine.choice(state="ready", choices=["approve", "reject"])
            self.assertEqual(set(result.probabilities), {"approve", "reject"})
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=5)
            for choice in result.probabilities:
                self.assertAlmostEqual(
                    result.probabilities[choice],
                    reference.probabilities[choice],
                    delta=0.03,
                )


if __name__ == "__main__":
    unittest.main()
