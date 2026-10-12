"""FastAPI Decision API for the HF or vLLM decision backend."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from futurix_decision import __version__
from futurix_decision.datasets import DecisionInput
from futurix_decision.inference.schemas import DecisionRequest, DecisionResponse

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


def _compat_input(
    state: Any,
    question_id: str,
    question: SystemOneQuestion,
    demonstrations: Any,
    prompt_profile: str,
) -> DecisionInput:
    kind = question.type
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"question {question_id!r}: unsupported type {kind!r}")
    criteria = question.criteria
    if kind == "noul":
        options = ["false", "true"]
    elif isinstance(criteria, list):
        options = [str(i) for i in range(len(criteria))]
    elif isinstance(criteria, dict):
        options = sorted(criteria, key=lambda key: int(key)) if kind == "score" else list(criteria)
    else:
        raise ValueError(f"question {question_id!r}: criteria must list choices")
    prompt = question.instructions or question_id
    if PROFILE_INSTRUCTIONS[prompt_profile]:
        prompt = PROFILE_INSTRUCTIONS[prompt_profile] + "\n" + prompt
    if criteria is not None:
        prompt += "\nCriteria: " + json.dumps(criteria, ensure_ascii=False, sort_keys=True)
    canonical_state = state if isinstance(state, (str, dict)) else {"value": state}
    if demonstrations:
        canonical_state = {"state": canonical_state, "demonstrations": demonstrations}
    return DecisionInput(kind=kind, state=canonical_state, question=prompt, options=options)


def create_app(engine: Any) -> FastAPI:
    app = FastAPI(title="Futurix Decision API", version=__version__)
    lock = threading.Lock()
    counters: dict[str, int] = defaultdict(int)
    latency: dict[str, float] = defaultdict(float)

    def run(request: DecisionRequest) -> DecisionResponse:
        try:
            example = request.to_input()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with lock:
            result = engine.decide(example, confidence_threshold=request.confidence_threshold)
            counters[request.kind] += 1
            latency[request.kind] += result.latency_ms
        return DecisionResponse.model_validate(result.to_dict())

    @app.post("/v1/decision", response_model=DecisionResponse)
    def decide(request: DecisionRequest) -> DecisionResponse:
        return run(request)

    @app.post("/v1/decision/choice", response_model=DecisionResponse)
    def choice(request: DecisionRequest) -> DecisionResponse:
        return run(request.model_copy(update={"kind": "choice"}))

    @app.post("/v1/decision/noul", response_model=DecisionResponse)
    def noul(request: DecisionRequest) -> DecisionResponse:
        return run(request.model_copy(update={"kind": "noul"}))

    @app.post("/v1/decision/score", response_model=DecisionResponse)
    def score(request: DecisionRequest) -> DecisionResponse:
        return run(request.model_copy(update={"kind": "score"}))

    @app.post("/v1/systemone")
    def systemone(body: SystemOneRequest) -> dict[str, Any]:
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
            try:
                return engine.system_one(
                    body.state,
                    {
                        qid: question.model_dump(exclude_none=True)
                        for qid, question in body.questions.items()
                    },
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.prompt_profile not in PROFILE_INSTRUCTIONS:
            raise HTTPException(status_code=422, detail="unsupported prompt_profile")
        if body.demonstration_format not in ("chat", "compact", "minimal"):
            raise HTTPException(status_code=422, detail="unsupported demonstration_format")
        try:
            inputs = [
                _compat_input(
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
        with lock:
            results = engine.batch_decide(inputs, confidence_threshold=body.confidence_threshold)
            for row, result in zip(inputs, results):
                counters[row.kind] += 1
                latency[row.kind] += result.latency_ms
        answers = {}
        for question_id, row, result in zip(body.questions, inputs, results):
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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": type(engine).__name__}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        with lock:
            return "".join(
                f'decision_requests_total{{kind="{kind}"}} {counters[kind]}\n'
                f'decision_latency_ms_sum{{kind="{kind}"}} {latency[kind]}\n'
                for kind in ("choice", "noul", "score")
            )

    return app


def create_vllm_app(checkpoint: str, **kwargs: Any) -> FastAPI:
    from futurix_decision.vllm.engine import VLLMEngine

    return create_app(VLLMEngine(checkpoint, **kwargs))
