"""The ``futurix-decision serve`` app: error statuses, auth, and micro-batching."""

import importlib.util
import unittest
from unittest.mock import patch

from futurix_decision.client import DecisionClient
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


if __name__ == "__main__":
    unittest.main()
