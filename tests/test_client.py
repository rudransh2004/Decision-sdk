"""DecisionClient and AsyncDecisionClient against the real server and stub transports."""

import asyncio
import importlib.util
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from futurix_decision import (
    AsyncDecisionClient,
    BackendError,
    ConfigError,
    DecisionClient,
    DecisionInput,
    DecisionResult,
    InvalidOutputError,
)

DECISION = {
    "kind": "choice",
    "decision": "approve",
    "confidence": 0.7,
    "probabilities": {"approve": 0.7, "reject": 0.3},
    "abstained": False,
    "latency_ms": 1.5,
}
CHOICE = DecisionInput(state="expired", question="Next?", options=["approve", "reject"])


class FakeEngine:
    """Puts 0.7 on the first option; decision types follow the decision kind."""

    def decide(self, example, *, confidence_threshold=None):
        rest = 0.3 / (len(example.options) - 1)
        probabilities = {
            option: 0.7 if index == 0 else rest for index, option in enumerate(example.options)
        }
        return DecisionResult.from_probabilities(
            example,
            list(probabilities.values()),
            latency_ms=2.0,
            confidence_threshold=confidence_threshold,
        )


def handler_sequence(*responses):
    """A MockTransport handler that replays responses and records requests."""
    requests = []

    def handle(request):
        requests.append(request)
        item = responses[min(len(requests), len(responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    return handle, requests


def ok():
    return httpx.Response(200, json=DECISION)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class ServerRoundTripTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from futurix_decision.vllm.api import create_app

        self.app = create_app(FakeEngine())
        self.http = TestClient(self.app)

    def test_every_method_parses_server_responses(self):
        client = DecisionClient(http_client=self.http)
        result = client.decide(CHOICE)
        self.assertEqual(result.decision, "approve")
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
        self.assertEqual(client.choice(state="s", choices=["a", "b", "c"]).decision, "a")
        self.assertIs(client.noul(state="s").decision, False)
        score = client.score(state="s", min_score=1, max_score=3)
        self.assertEqual(score.decision, 1)
        self.assertEqual(set(score.probabilities), {"1", "2", "3"})
        abstained = client.choice(state="s", choices=["a", "b"], confidence_threshold=0.9)
        self.assertTrue(abstained.abstained)
        self.assertIsNone(abstained.decision)
        batch = client.batch_decide(
            [CHOICE, DecisionInput(kind="noul", state="s", question="q", options=["false", "true"])]
        )
        self.assertEqual([row.decision for row in batch], ["approve", False])
        self.assertEqual(client.health(), {"status": "ok", "backend": "FakeEngine"})
        client.close()
        self.assertFalse(self.http.is_closed)  # Injected clients stay open.

    def test_async_client_round_trip(self):
        async def run():
            http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://test"
            )
            async with AsyncDecisionClient(http_client=http) as client:
                self.assertIs((await client.noul(state="s")).decision, False)
                self.assertEqual((await client.score(state="s")).decision, 0)
                rows = [
                    DecisionInput(state="s", question="q", options=[f"x{i}", "y"]) for i in range(5)
                ]
                results = await client.batch_decide(rows, max_concurrency=2)
                self.assertEqual([r.decision for r in results], [f"x{i}" for i in range(5)])
                self.assertEqual((await client.health())["status"], "ok")
            self.assertFalse(http.is_closed)
            await http.aclose()

        asyncio.run(run())


@patch("time.sleep")
class RetryTests(unittest.TestCase):
    def client(self, handler, **kwargs):
        return DecisionClient(
            "http://decisions.test", transport=httpx.MockTransport(handler), **kwargs
        )

    def test_retries_503_then_succeeds(self, sleep):
        handler, requests = handler_sequence(httpx.Response(503), ok())
        with self.client(handler) as client:
            self.assertEqual(client.decide(CHOICE).decision, "approve")
        self.assertEqual(len(requests), 2)
        self.assertEqual(sleep.call_count, 1)

    def test_persistent_503_raises_after_max_retries(self, sleep):
        handler, requests = handler_sequence(httpx.Response(503, text="overloaded"))
        with self.client(handler, max_retries=2) as client:
            with self.assertRaises(BackendError) as caught:
                client.decide(CHOICE)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.body, "overloaded")
        self.assertEqual(len(requests), 3)

    def test_client_error_is_not_retried(self, sleep):
        handler, requests = handler_sequence(httpx.Response(400, json={"detail": "bad state"}))
        with self.client(handler) as client:
            with self.assertRaises(BackendError) as caught:
                client.decide(CHOICE)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("bad state", caught.exception.body)
        self.assertEqual(len(requests), 1)
        sleep.assert_not_called()

    def test_retry_after_is_honored(self, sleep):
        handler, _ = handler_sequence(httpx.Response(429, headers={"Retry-After": "7"}), ok())
        with self.client(handler) as client:
            client.decide(CHOICE)
        sleep.assert_called_once_with(7.0)

    def test_connect_errors_are_retried(self, sleep):
        handler, requests = handler_sequence(httpx.ConnectError("refused"), ok())
        with self.client(handler) as client:
            self.assertEqual(client.decide(CHOICE).confidence, 0.7)
        self.assertEqual(len(requests), 2)

    def test_read_timeout_raises_backend_error_without_retry(self, sleep):
        handler, requests = handler_sequence(httpx.ReadTimeout("slow"))
        with self.client(handler) as client:
            with self.assertRaises(BackendError) as caught:
                client.decide(CHOICE)
        self.assertIsNone(caught.exception.status_code)
        self.assertIn("ReadTimeout", str(caught.exception))
        self.assertEqual(len(requests), 1)

    def test_timeout_and_api_key_reach_the_request(self, sleep):
        handler, requests = handler_sequence(ok())
        with self.client(handler, api_key="secret", timeout=4.0) as client:
            client.decide(CHOICE, confidence_threshold=0.5)
        request = requests[0]
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        self.assertEqual(request.extensions["timeout"]["read"], 4.0)
        self.assertEqual(request.url.path, "/v1/decision")
        self.assertEqual(
            json.loads(request.content),
            {
                "kind": "choice",
                "state": "expired",
                "question": "Next?",
                "choices": ["approve", "reject"],
                "confidence_threshold": 0.5,
            },
        )

    def test_malformed_success_raises_invalid_output(self, sleep):
        handler, _ = handler_sequence(httpx.Response(200, json={"decision": "approve"}))
        with self.client(handler) as client:
            with self.assertRaises(InvalidOutputError) as caught:
                client.decide(CHOICE)
        self.assertEqual(caught.exception.status_code, 200)

    def test_owned_client_is_closed_on_exit(self, sleep):
        handler, _ = handler_sequence(ok())
        with self.client(handler) as client:
            pass
        with self.assertRaises(RuntimeError):
            client.decide(CHOICE)


class ArgumentTests(unittest.TestCase):
    def test_rejects_fields_the_server_cannot_carry(self):
        client = DecisionClient(transport=httpx.MockTransport(lambda request: ok()))
        example = CHOICE.model_copy(update={"candidate_descriptions": ["yes", "no"]})
        with self.assertRaisesRegex(ConfigError, "candidate_descriptions"):
            client.decide(example)
        with self.assertRaises(ConfigError):
            client.decide(CHOICE, confidence_threshold=2.0)
        with self.assertRaises(ConfigError):
            DecisionClient(max_retries=-1)
        with self.assertRaises(ConfigError):
            DecisionClient(api_key="k", http_client=httpx.Client())
        client.close()


class AsyncRetryTests(unittest.TestCase):
    def run_client(self, handler, call, **kwargs):
        async def run():
            async with AsyncDecisionClient(
                "http://decisions.test", transport=httpx.MockTransport(handler), **kwargs
            ) as client:
                return await call(client)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            return asyncio.run(run()), sleep

    def test_retries_then_succeeds_honoring_retry_after(self):
        handler, requests = handler_sequence(
            httpx.Response(503, headers={"Retry-After": "2"}), ok()
        )
        result, sleep = self.run_client(handler, lambda client: client.decide(CHOICE))
        self.assertEqual(result.decision, "approve")
        self.assertEqual(len(requests), 2)
        sleep.assert_awaited_once_with(2.0)

    def test_persistent_503_and_timeout_raise_backend_error(self):
        handler, requests = handler_sequence(httpx.Response(503))
        with self.assertRaises(BackendError) as caught:
            self.run_client(handler, lambda client: client.decide(CHOICE), max_retries=1)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(len(requests), 2)

        handler, requests = handler_sequence(httpx.ReadTimeout("slow"))
        with self.assertRaises(BackendError):
            self.run_client(handler, lambda client: client.noul(state="s"))
        self.assertEqual(len(requests), 1)

    def test_batch_concurrency_is_bounded(self):
        active = 0
        peak = 0

        async def handle(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            for _ in range(3):
                await asyncio.sleep(0)
            active -= 1
            return ok()

        async def run():
            async with AsyncDecisionClient(transport=httpx.MockTransport(handle)) as client:
                return await client.batch_decide([CHOICE] * 10, max_concurrency=3)

        results = asyncio.run(run())
        self.assertEqual(len(results), 10)
        self.assertEqual(peak, 3)


if __name__ == "__main__":
    unittest.main()
