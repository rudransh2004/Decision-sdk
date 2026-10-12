# Serve decisions over HTTP

`futurix-decision serve` puts any backend behind a small FastAPI app.
`DecisionClient` and `AsyncDecisionClient` call it from Python without
installing torch.

## Start a server

```bash
python -m pip install "futurix-decision[laya,serve]"
futurix-decision serve --model laya --model-format native --port 8000
```

`--model` takes a Hub ID, an alias, or a local checkpoint. `--model-format`
names the family (`standard` for a checkpoint you trained and exported,
`bosun`, `decider`, `jpt`, `eikos`, any registered family, or `native` for a
publisher runtime). `--backend vllm` serves an exported checkpoint through
`VLLMEngine`; see [inference backends](inference-backends.md). For a publisher
helper such as OpenJev, pass `--native-base-url`.

Without an API key the server binds `127.0.0.1`. With a key it binds
`0.0.0.0`. Set `--host` to choose. Serving on another address without a key
logs a warning.

From Python, `create_app` builds the same app around any `DecisionBackend`
that implements `batch_decide`:

<!-- test -->
```python
from futurix_decision import load_model
from futurix_decision.vllm.api import create_app

app = create_app(load_model("./my-decision-model"), api_key="secret", max_batch=16)
# uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Endpoints

| Endpoint | Purpose |
|:--|:--|
| `POST /v1/decision` | Any kind; the body's `kind` picks it |
| `POST /v1/decision/choice` | `choices` required |
| `POST /v1/decision/noul` | Yes/no |
| `POST /v1/decision/score` | `min_score` to `max_score` levels |
| `POST /v1/systemone` | Several typed questions over one state |
| `GET /health` | `{"status": "ok", "backend": ...}`; never needs a key |
| `GET /metrics` | Prometheus text: request counts, latency sums, `decision_batches_total` |

```bash
curl -s http://127.0.0.1:8000/v1/decision/choice \
  -H 'content-type: application/json' \
  -d '{"state":{"gst_certificate":"expired"},"question":"Next?","choices":["approve","request_document","reject"],"confidence_threshold":0.8}'
```

The response is a `DecisionResult` as JSON: `kind`, `decision`, `confidence`,
`probabilities`, `abstained`, and `latency_ms`. Its `kind` field is required,
and `DecisionClient` checks it against the request. Upgrade the client and the
server together.

System One requests use
`{"state": ..., "questions": {"id": {"type": "choice", "instructions": "...",
"criteria": {"option": "description"}}}}`. The server translates them into
canonical `DecisionInput` objects, including the criteria descriptions. Text-only
backends reject media.

## Authentication

`--api-key` (or the `FUTURIX_API_KEY` environment variable) requires
`Authorization: Bearer <key>` on every endpoint except `/health`. A missing or
wrong key gets HTTP 401. `DecisionClient(api_key=...)` sends it.

## Batching

The server micro-batches concurrent requests for every backend. The first
waiting row holds a batch open for `--batch-window-ms` (default 5) or until
`--max-batch` rows (default 32), and the batch runs as one `batch_decide` call
(one `generate` call with `--backend vllm`). Each row keeps its own
`confidence_threshold`. If a batch call raises, its rows are retried one at a
time, so an error reaches only the requests whose rows cause it.

One worker thread makes every backend call, so the event loop keeps accepting
requests while a batch runs. A row's result does not depend on the rows
batched with it.

## Errors

The server maps SDK errors to HTTP statuses. The body is
`{"detail": ..., "error": ...}`, and the `X-Futurix-Error` header repeats
`error`.

| SDK error | Status | `error` |
|:--|:--|:--|
| `ConfigError` | 422 | `config` |
| `CheckpointError` | 500 | `checkpoint` |
| `BackendError` | 502 | `backend` |
| `InvalidOutputError` | 502 | `invalid_output` |

## Clients

`DecisionClient` implements the same `DecisionBackend` methods as a local
model, so code that takes a backend runs against a remote one unchanged. Install
only the base package for it: `pip install futurix-decision`.

<!-- test -->
```python
from futurix_decision import DecisionClient

with DecisionClient("http://127.0.0.1:8000", api_key="secret") as client:
    print(client.health())
    result = client.choice(state="Certificate expired", choices=["approve", "reject"])
    print(result.decision, result.probabilities)
    print(client.noul(state="Invoice paid", question="Was it paid?").decision)  # True or False
```

Settings: `base_url`, `api_key`, `timeout` (30 s), and `max_retries` (3). The
client retries connect errors and HTTP 429, 502, 503, and 504 with
exponential backoff and jitter, and honors `Retry-After` up to 60 s. Other
failures raise `BackendError`, which carries `status_code` and `body`. An
`invalid_output` 502 raises `InvalidOutputError` at once, without a retry.
Pass `transport` or `http_client` to route the connection, for example to an
`httpx.MockTransport` in tests.

The server has no batch endpoint, so `batch_decide` sends one request per
example. The sync client sends them one after another. The async client runs
at most `max_concurrency` (8) at a time:

<!-- test -->
```python
import asyncio

from futurix_decision import AsyncDecisionClient, DecisionInput


async def main():
    rows = [
        DecisionInput(state="Certificate expired", question="Next?", options=["approve", "reject"]),
        DecisionInput(state="Certificate valid", question="Next?", options=["approve", "reject"]),
    ]
    async with AsyncDecisionClient("http://127.0.0.1:8000") as client:
        for result in await client.batch_decide(rows, max_concurrency=4):
            print(result.decision)


asyncio.run(main())
```

A publisher's System One HTTP helper is a different server. Load it with
`load_model("openjev-flash", base_url=...)`; it uses the same retry policy and
raises `BackendError` for failed requests. See [model support](model-support.md).
