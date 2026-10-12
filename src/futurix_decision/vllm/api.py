"""FastAPI Decision API over any ``DecisionBackend``: Transformers, vLLM, or native."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from futurix_decision import __version__
from futurix_decision.datasets import DecisionInput
from futurix_decision.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    FuturixError,
    InvalidOutputError,
)
from futurix_decision.inference.schemas import (
    ERROR_HEADER,
    INVALID_OUTPUT,
    DecisionRequest,
    DecisionResult,
)
from futurix_decision.models.protocol import DecisionBackend
from futurix_decision.vllm.batcher import MicroBatcher

logger = logging.getLogger(__name__)

PROFILE_INSTRUCTIONS = {
    "baseline": "",
    "semantic": "Use the meaning of options rather than their order.",
    "evidence": "Use only evidence in the state; do not invent missing facts.",
    "checklist": "Check every option against the criteria and state.",
    "strict": "Apply the criterion literally without unstated assumptions.",
}


class SystemOneQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "choice"
    instructions: str = ""
    criteria: dict[str, Any] | list[Any] | None = None


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    state: Any
    questions: dict[str, SystemOneQuestion] = Field(min_length=1)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    demonstrations: dict[str, Any] | None = None
    prompt_profile: str = "baseline"
    demonstration_format: str = "chat"
    media: Any | None = None


def _system_one_input(
    state: Any,
    question_id: str,
    question: SystemOneQuestion,
    demonstrations: Any,
    prompt_profile: str,
) -> DecisionInput:
    kind = question.type
    if kind not in ("noul", "choice", "score"):
        raise ConfigError(f"question {question_id!r}: unsupported type {kind!r}")
    criteria = question.criteria
    if kind == "noul":
        options = ["false", "true"]
    elif isinstance(criteria, list):
        options = [str(i) for i in range(len(criteria))]
    elif isinstance(criteria, dict):
        options = sorted(criteria, key=lambda key: int(key)) if kind == "score" else list(criteria)
    else:
        raise ConfigError(f"question {question_id!r}: criteria must list choices")
    prompt = question.instructions or question_id
    if PROFILE_INSTRUCTIONS[prompt_profile]:
        prompt = PROFILE_INSTRUCTIONS[prompt_profile] + "\n" + prompt
    if criteria is not None:
        prompt += "\nCriteria: " + json.dumps(criteria, ensure_ascii=False, sort_keys=True)
    canonical_state = state if isinstance(state, (str, dict)) else {"value": state}
    if demonstrations:
        canonical_state = {"state": canonical_state, "demonstrations": demonstrations}
    return DecisionInput(kind=kind, state=canonical_state, question=prompt, options=options)


# HTTP status, ERROR_HEADER value, and public detail per SDK error a backend raises
# during a request. Starlette picks the handler of the nearest class in the
# exception's MRO. Only a ConfigError, which describes the request, returns its own
# message: the others can name upstream URLs or local paths, so they are logged.
ERROR_STATUS: dict[type[FuturixError], tuple[int, str, str | None]] = {
    ConfigError: (422, "config", None),
    CheckpointError: (500, "checkpoint", "the server's checkpoint failed to load"),
    BackendError: (502, "backend", "the decision backend failed"),
    InvalidOutputError: (502, INVALID_OUTPUT, "the decision backend returned invalid output"),
}


def _error_response(request: Request, exc: Exception) -> JSONResponse:
    status, code, detail = next(
        ERROR_STATUS[cls] for cls in type(exc).__mro__ if cls in ERROR_STATUS
    )
    if detail is None:
        detail = str(exc)
    else:
        logger.error("%s %s failed: %s", request.method, request.url.path, exc, exc_info=exc)
    return JSONResponse(
        {"detail": detail, "error": code}, status_code=status, headers={ERROR_HEADER: code}
    )


def _bearer(api_key: str) -> Callable[[str | None], None]:
    expected = api_key.encode()

    def check(authorization: str | None = Header(default=None)) -> None:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return check


def create_app(
    engine: DecisionBackend,
    *,
    api_key: str | None = None,
    batch_window_ms: float = 5.0,
    max_batch: int = 32,
) -> FastAPI:
    """Serve ``engine`` over HTTP; ``DecisionClient`` is the matching client.

    Concurrent requests are micro-batched: rows that arrive within
    ``batch_window_ms`` of each other, up to ``max_batch``, share one
    ``engine.batch_decide`` call (see ``MicroBatcher``). This works for any
    backend, including ``VLLMEngine``, which then makes one ``generate`` call
    per batch. With ``api_key``, every endpoint except ``/health`` requires
    ``Authorization: Bearer <api_key>``, and ``/docs``, ``/redoc``, and
    ``/openapi.json`` are off.
    """
    batcher = MicroBatcher(engine, window_ms=batch_window_ms, max_batch=max_batch)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        batcher.start()
        try:
            yield
        finally:
            batcher.close()

    # The interactive docs and schema would be open endpoints, so a key turns them off.
    docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if api_key else {}
    app = FastAPI(title="Futurix Decision API", version=__version__, lifespan=lifespan, **docs)
    app.state.batcher = batcher
    for error in ERROR_STATUS:
        app.add_exception_handler(error, _error_response)
    api = APIRouter(dependencies=[Depends(_bearer(api_key))] if api_key else [])
    counters: dict[str, int] = defaultdict(int)
    latency: dict[str, float] = defaultdict(float)

    async def run(rows: list[DecisionInput], threshold: float | None) -> list[DecisionResult]:
        results = await asyncio.gather(
            *(batcher.decide(row, confidence_threshold=threshold) for row in rows)
        )
        for row, result in zip(rows, results, strict=True):
            counters[row.kind] += 1
            latency[row.kind] += result.latency_ms
        return list(results)

    async def run_request(request: DecisionRequest) -> DecisionResult:
        try:
            example = request.to_input()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        (result,) = await run([example], request.confidence_threshold)
        return result

    @api.post("/v1/decision", response_model=DecisionResult)
    async def decide(request: DecisionRequest) -> DecisionResult:
        return await run_request(request)

    @api.post("/v1/decision/choice", response_model=DecisionResult)
    async def choice(request: DecisionRequest) -> DecisionResult:
        return await run_request(request.model_copy(update={"kind": "choice"}))

    @api.post("/v1/decision/noul", response_model=DecisionResult)
    async def noul(request: DecisionRequest) -> DecisionResult:
        return await run_request(request.model_copy(update={"kind": "noul"}))

    @api.post("/v1/decision/score", response_model=DecisionResult)
    async def score(request: DecisionRequest) -> DecisionResult:
        return await run_request(request.model_copy(update={"kind": "score"}))

    @api.post("/v1/systemone")
    async def systemone(body: SystemOneRequest) -> dict[str, Any]:
        if body.media is not None:
            raise HTTPException(
                status_code=422, detail="media is not supported by text-only checkpoints"
            )
        from futurix_decision.models.native import NativeDecisionModel

        if isinstance(engine, NativeDecisionModel):
            if (
                body.prompt_profile != "baseline"
                or body.demonstrations
                or body.confidence_threshold is not None
            ):
                raise HTTPException(
                    status_code=422,
                    detail="native System One inference does not support Futurix prompt overrides",
                )
            questions = {
                qid: question.model_dump(exclude_none=True)
                for qid, question in body.questions.items()
            }
            try:
                return await batcher.call(engine.system_one, body.state, questions)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.prompt_profile not in PROFILE_INSTRUCTIONS:
            raise HTTPException(status_code=422, detail="unsupported prompt_profile")
        if body.demonstration_format not in ("chat", "compact", "minimal"):
            raise HTTPException(status_code=422, detail="unsupported demonstration_format")
        try:
            inputs = [
                _system_one_input(
                    body.state,
                    question_id,
                    question,
                    (body.demonstrations or {}).get(question_id),
                    body.prompt_profile,
                )
                for question_id, question in body.questions.items()
            ]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        results = await run(inputs, body.confidence_threshold)
        answers = {}
        for question_id, row, result in zip(body.questions, inputs, results, strict=True):
            if row.kind == "noul":
                answer = {"type": "noul", "noul": result.probabilities["true"]}
            elif row.kind == "score":
                answer = {
                    "type": "score",
                    "score": sum(
                        int(option) * probability
                        for option, probability in result.probabilities.items()
                    ),
                    "probabilities": result.probabilities,
                }
            else:
                answer = {
                    "type": "choice",
                    "choice": result.decision,
                    "probabilities": result.probabilities,
                }
            answer.update(confidence=result.confidence, abstained=result.abstained)
            answers[question_id] = answer
        return {
            "model": body.model or "futurix-decision",
            "answers": answers,
            "thinking": False,
            "prompt_profile": body.prompt_profile,
            "demonstration_format": body.demonstration_format,
        }

    @api.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return (
            "".join(
                f'decision_requests_total{{kind="{kind}"}} {counters[kind]}\n'
                f'decision_latency_ms_sum{{kind="{kind}"}} {latency[kind]}\n'
                for kind in ("choice", "noul", "score")
            )
            + f"decision_batches_total {batcher.batches}\n"
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": type(engine).__name__}

    app.include_router(api)
    return app
