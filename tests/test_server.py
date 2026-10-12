"""The ``futurix-decision serve`` app: error statuses, auth, and micro-batching."""

import asyncio
import importlib.util
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from futurix_decision import DecisionInput
from futurix_decision.client import AsyncDecisionClient, DecisionClient
from futurix_decision.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    InvalidOutputError,
)
from futurix_decision.inference.schemas import ERROR_HEADER
from futurix_decision.models.protocol import DecisionShortcuts

CHOICE = {"state": "ticket", "question": "Next?", "choices": ["approve", "reject"]}


class Failing(DecisionShortcuts):
    """A backend whose every call raises ``error``."""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def decide(self, example, *, confidence_threshold=None):
        return self.batch_decide([example], confidence_threshold=confidence_threshold)[0]

    def batch_decide(self, examples, *, confidence_threshold=None):
        self.calls += 1
        raise self.error

    def close(self):
        pass


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class ErrorStatusTests(unittest.TestCase):
    def post(self, error):
        from fastapi.testclient import TestClient

        from futurix_decision.vllm.api import create_app

        backend = Failing(error)
        http = TestClient(create_app(backend))
        return http.post("/v1/decision/choice", json=CHOICE), http, backend

    def test_each_sdk_error_has_its_status_and_body(self):
        cases = [
            (ConfigError("prompt too long"), 422, "config"),
            (CheckpointError("missing head"), 500, "checkpoint"),
            (BackendError("upstream down"), 502, "backend"),
            (InvalidOutputError("probabilities sum to 2"), 502, "invalid_output"),
        ]
        for error, status, code in cases:
            with self.subTest(error=type(error).__name__):
                response, _, _ = self.post(error)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), {"detail": str(error), "error": code})
                self.assertEqual(response.headers[ERROR_HEADER], code)

    @patch("time.sleep")
    def test_client_does_not_retry_invalid_output(self, sleep):
        _, http, backend = self.post(InvalidOutputError("bad probabilities"))
        backend.calls = 0
        with (
            DecisionClient(http_client=http) as client,
            self.assertRaises(InvalidOutputError) as raised,
        ):
            client.choice(state="ticket", choices=["approve", "reject"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(backend.calls, 1)
        sleep.assert_not_called()

    @patch("time.sleep")
    def test_client_retries_a_backend_error(self, sleep):
        _, http, backend = self.post(BackendError("upstream down"))
        backend.calls = 0
        with DecisionClient(http_client=http, max_retries=2) as client:
            with self.assertRaises(BackendError) as raised:
                client.choice(state="ticket", choices=["approve", "reject"])
        self.assertNotIsInstance(raised.exception, InvalidOutputError)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(backend.calls, 3)

    def test_client_maps_config_error_to_backend_error_422(self):
        _, http, _ = self.post(ConfigError("prompt too long"))
        with DecisionClient(http_client=http) as client:
            with self.assertRaises(BackendError) as raised:
                client.choice(state="ticket", choices=["approve", "reject"])
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("prompt too long", raised.exception.body)


def app_for(backend, **kwargs):
    from futurix_decision.vllm.api import create_app

    return create_app(backend, **kwargs)


def async_client(app, **kwargs):
    return AsyncDecisionClient(
        base_url="http://test", transport=httpx.ASGITransport(app=app), **kwargs
    )


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class MicroBatchingTests(unittest.TestCase):
    def test_64_concurrent_requests_share_forward_passes(self):
        from test_sft import tiny_model

        model = tiny_model()
        forwards = []
        model.backbone.register_forward_hook(lambda *args: forwards.append(1))
        rows = [
            DecisionInput(state=f"ticket {i}", question="Next?", options=["approve", "reject"])
            for i in range(64)
        ]
        app = app_for(model, api_key="secret")

        async def run():
            async with async_client(app, api_key="secret") as client:
                results = await client.batch_decide(rows, max_concurrency=64)
                metrics = await client._http.get("/metrics")
            return results, metrics.text

        results, metrics = asyncio.run(run())
        self.assertEqual(len(results), 64)
        self.assertLess(len(forwards), 64)
        self.assertIn(f"decision_batches_total {len(forwards)}", metrics)
        self.assertIn('decision_requests_total{kind="choice"} 64', metrics)
        # Batching does not change an answer (this backbone pads on the right).
        for row, result in zip(rows[::9], results[::9], strict=True):
            expected = model.decide(row)
            for option, value in expected.probabilities.items():
                self.assertAlmostEqual(result.probabilities[option], value, places=5)

    def test_each_request_keeps_its_own_confidence_threshold(self):
        from test_sft import tiny_model

        model = tiny_model()
        calls = []
        original = model.batch_decide
        model.batch_decide = lambda rows, **kwargs: calls.append(len(rows)) or original(rows)
        app = app_for(model, batch_window_ms=50)

        async def run():
            async with async_client(app) as client:
                return await asyncio.gather(
                    client.choice(state="s", choices=["a", "b"], confidence_threshold=1.0),
                    client.choice(state="s", choices=["a", "b"]),
                )

        strict, plain = asyncio.run(run())
        self.assertEqual(calls, [2])
        self.assertTrue(strict.abstained)
        self.assertIsNone(strict.decision)
        self.assertFalse(plain.abstained)
        self.assertEqual(strict.probabilities, plain.probabilities)

    def test_a_failing_row_fails_only_its_own_request(self):
        from test_sft import tiny_model

        model = tiny_model()
        calls = []
        original = model.batch_decide

        def batch_decide(rows, **kwargs):
            calls.append(len(rows))
            if any(row.state == "too long" for row in rows):
                raise ConfigError("prompt exceeds max_sequence_length")
            return original(rows)

        model.batch_decide = batch_decide
        app = app_for(model, batch_window_ms=50)
        states = ["ok 1", "too long", "ok 2"]

        async def run():
            async with async_client(app) as client:
                return await asyncio.gather(
                    *(client.choice(state=state, choices=["a", "b"]) for state in states),
                    return_exceptions=True,
                )

        first, failed, last = asyncio.run(run())
        # One batch fails, then each row is retried alone.
        self.assertEqual(calls, [3, 1, 1, 1])
        self.assertEqual(first.kind, "choice")
        self.assertEqual(last.kind, "choice")
        self.assertIsInstance(failed, BackendError)
        self.assertEqual(failed.status_code, 422)
        self.assertIn("max_sequence_length", failed.body)

    def test_a_malformed_result_fails_its_request_and_the_server_keeps_serving(self):
        from futurix_decision import DecisionResult

        class Partial(Failing):
            def batch_decide(self, examples, *, confidence_threshold=None):
                return [
                    DecisionResult(
                        kind="choice",
                        decision="a",
                        confidence=1.0,
                        probabilities={"a": 1.0} if row.state == "bad" else {"a": 1.0, "b": 0.0},
                        abstained=False,
                        latency_ms=1.0,
                    )
                    for row in examples
                ]

        app = app_for(Partial(None))

        async def run():
            async with async_client(app, max_retries=0) as client:
                with self.assertRaises(InvalidOutputError):
                    await client.choice(state="bad", choices=["a", "b"])
                return await client.choice(state="s", choices=["a", "b"])

        self.assertEqual(asyncio.run(run()).decision, "a")

    def test_systemone_questions_share_one_batch(self):
        from fastapi.testclient import TestClient

        from test_sft import tiny_model

        model = tiny_model()
        calls = []
        original = model.batch_decide
        model.batch_decide = lambda rows, **kwargs: calls.append(len(rows)) or original(rows)
        response = TestClient(app_for(model, batch_window_ms=50)).post(
            "/v1/systemone",
            json={
                "state": "certificate expired",
                "questions": {
                    "valid": {"type": "noul", "instructions": "Valid?"},
                    "next": {"type": "choice", "criteria": {"approve": "", "reject": ""}},
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(calls, [2])


class Recording(DecisionShortcuts):
    """A backend that records each batch and fails rows whose state names an error."""

    def __init__(self, errors=None, delay=0.0, result=None):
        self.errors = errors or {}
        self.delay = delay
        self.result = result
        self.batches = []

    def decide(self, example, *, confidence_threshold=None):
        return self.batch_decide([example])[0]

    def batch_decide(self, examples, *, confidence_threshold=None):
        from futurix_decision import DecisionResult

        self.batches.append([row.state for row in examples])
        time.sleep(self.delay)
        for row in examples:
            if row.state in self.errors:
                raise self.errors[row.state]
        if self.result is not None:
            return [self.result(row) for row in examples]
        return [
            DecisionResult.from_probabilities(row, [0.25, 0.75], latency_ms=1.0) for row in examples
        ]

    def close(self):
        pass


def rows(*states):
    return [DecisionInput(state=state, question="Next?", options=["a", "b"]) for state in states]


class BatcherTests(unittest.TestCase):
    def decide_all(self, backend, examples, **kwargs):
        from futurix_decision.vllm.batcher import MicroBatcher

        batcher = MicroBatcher(backend, window_ms=50)

        async def run():
            return await asyncio.wait_for(
                asyncio.gather(
                    *(batcher.decide(row, **kwargs) for row in examples), return_exceptions=True
                ),
                10,
            )

        try:
            return asyncio.run(run()), batcher
        finally:
            batcher.close()

    def test_a_backend_outage_fails_the_batch_with_one_call(self):
        error = BackendError("upstream down")
        backend = Recording({f"row {i}": error for i in range(32)})
        results, batcher = self.decide_all(backend, rows(*(f"row {i}" for i in range(32))))
        self.assertEqual(len(backend.batches), 1)
        self.assertTrue(all(result is error for result in results))
        self.assertEqual(batcher.batches, 1)

    def test_retrying_rows_alone_does_not_count_as_batches(self):
        backend = Recording({"too long": ConfigError("prompt too long")})
        results, batcher = self.decide_all(backend, rows("ok", "too long", "ok too"))
        self.assertEqual([len(batch) for batch in backend.batches], [3, 1, 1, 1])
        self.assertIsInstance(results[1], ConfigError)
        self.assertEqual(batcher.batches, 1)

    def test_a_malformed_result_fails_only_its_request(self):
        def result(row):
            if row.state == "bad":
                return {"probabilities": {"a": 0.5, "b": 0.5}}
            return DecisionResult.from_probabilities(row, [0.25, 0.75], latency_ms=1.0)

        from futurix_decision import DecisionResult

        for threshold in (None, 0.5):
            with self.subTest(confidence_threshold=threshold):
                backend = Recording(result=result)
                (good, bad), _ = self.decide_all(
                    backend, rows("good", "bad"), confidence_threshold=threshold
                )
                self.assertEqual(good.decision, "b")
                self.assertIsInstance(bad, InvalidOutputError)

    def test_the_worker_keeps_serving_after_a_malformed_result(self):
        from futurix_decision.vllm.batcher import MicroBatcher

        state = {"broken": True}

        def result(row):
            if state.pop("broken", False):
                return object()
            return DecisionResult.from_probabilities(row, [0.25, 0.75], latency_ms=1.0)

        from futurix_decision import DecisionResult

        batcher = MicroBatcher(Recording(result=result), window_ms=1)

        async def run():
            first = await asyncio.gather(
                batcher.decide(rows("s")[0], confidence_threshold=0.5), return_exceptions=True
            )
            second = await asyncio.wait_for(batcher.decide(rows("s")[0]), 5)
            return first[0], second

        try:
            first, second = asyncio.run(run())
        finally:
            batcher.close()
        self.assertIsInstance(first, InvalidOutputError)
        self.assertEqual(second.decision, "b")

    def test_close_fails_waiting_requests(self):
        from futurix_decision.vllm.batcher import MicroBatcher

        batcher = MicroBatcher(Recording(delay=0.3), window_ms=1, max_batch=1)

        async def run():
            pending = [asyncio.ensure_future(batcher.decide(row)) for row in rows("a", "b", "c")]
            await asyncio.sleep(0.05)
            batcher.close()
            return await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 5)

        results = asyncio.run(run())
        self.assertTrue(all(isinstance(result, BackendError) for result in results), results)
        self.assertTrue(all("shutting down" in str(result) for result in results))


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class LifespanTests(unittest.TestCase):
    def test_the_app_serves_across_repeated_lifespans(self):
        from fastapi.testclient import TestClient

        app = app_for(Recording())
        for _ in range(2):
            with TestClient(app, raise_server_exceptions=False) as http:
                response = http.post("/v1/decision/choice", json=CHOICE)
                self.assertEqual(response.status_code, 200, response.text)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "serve extra not installed")
class AuthTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from test_sft import tiny_model

        self.http = TestClient(app_for(tiny_model(), api_key="secret"))

    def test_endpoints_require_the_bearer_key(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret"}):
            with self.subTest(headers=headers):
                response = self.http.post("/v1/decision/choice", json=CHOICE, headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")
                self.assertEqual(self.http.get("/metrics", headers=headers).status_code, 401)
        ok = self.http.post(
            "/v1/decision/choice", json=CHOICE, headers={"Authorization": "Bearer secret"}
        )
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_health_stays_open(self):
        self.assertEqual(self.http.get("/health").status_code, 200)

    def test_client_sends_its_api_key(self):
        app = self.http.app

        async def run(key):
            async with async_client(app, api_key=key, max_retries=0) as client:
                return await client.noul(state="ready")

        self.assertIs(type(asyncio.run(run("secret")).decision), bool)
        with self.assertRaises(BackendError) as raised:
            asyncio.run(run("wrong"))
        self.assertEqual(raised.exception.status_code, 401)


@unittest.skipUnless(importlib.util.find_spec("uvicorn"), "serve extra not installed")
class ServeCommandTests(unittest.TestCase):
    def serve(self, *flags, env=None):
        from futurix_decision.cli import main
        from test_sft import tiny_model

        with tempfile.TemporaryDirectory() as temp:
            tiny_model().save_pretrained(temp)
            with (
                patch.dict(os.environ, env or {}, clear=False),
                patch("uvicorn.run") as run,
            ):
                if env is None:
                    os.environ.pop("FUTURIX_API_KEY", None)
                main(["serve", "--model", temp, "--device", "cpu", *flags])
        app = run.call_args.args[0]
        return app, run.call_args.kwargs["host"]

    def test_without_a_key_binds_loopback_and_stays_open(self):
        from fastapi.testclient import TestClient

        app, host = self.serve()
        self.assertEqual(host, "127.0.0.1")
        response = TestClient(app).post("/v1/decision/choice", json=CHOICE)
        self.assertEqual(response.status_code, 200)

    def test_key_from_the_environment_requires_auth_and_binds_all_interfaces(self):
        from fastapi.testclient import TestClient

        app, host = self.serve(env={"FUTURIX_API_KEY": "from-env"})
        self.assertEqual(host, "0.0.0.0")
        http = TestClient(app)
        self.assertEqual(http.post("/v1/decision/choice", json=CHOICE).status_code, 401)
        authorized = {"Authorization": "Bearer from-env"}
        response = http.post("/v1/decision/choice", json=CHOICE, headers=authorized)
        self.assertEqual(response.status_code, 200)

    def test_revision_pins_every_model_format(self):
        from futurix_decision.cli import main

        for model_format in ("standard", "bosun", "native"):
            with (
                self.subTest(model_format=model_format),
                patch("futurix_decision.cli.load_model") as load,
                patch("uvicorn.run"),
            ):
                main(
                    ["serve", "--model", "acme/model", "--device", "cpu"]
                    + ["--model-format", model_format, "--revision", "a" * 40]
                )
                self.assertEqual(load.call_args.kwargs["revision"], "a" * 40)

    def test_vllm_refuses_a_revision_it_cannot_pin(self):
        from futurix_decision.cli import main

        with patch("uvicorn.run") as run, self.assertRaises(ConfigError):
            main(["serve", "--model", "acme/model", "--backend", "vllm", "--revision", "a" * 40])
        run.assert_not_called()

    def test_flags_set_the_key_host_and_batching(self):
        app, host = self.serve(
            "--api-key", "flag", "--host", "10.0.0.5", "--batch-window-ms", "2", "--max-batch", "8"
        )
        self.assertEqual(host, "10.0.0.5")
        batcher = app.state.batcher
        self.assertEqual((batcher.window, batcher.max_batch), (0.002, 8))


if __name__ == "__main__":
    unittest.main()
