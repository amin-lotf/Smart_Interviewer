from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Dict, Any, AsyncIterator

import anyio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse

from smart_interviewer.core import InterviewEngine, initial_state, ClientAction, WhisperTranscriber
from smart_interviewer.services import OpenAITTSService
from smart_interviewer.settings import settings

logger = logging.getLogger("smart_interviewer")

PUBLIC_STATE_BY_SESSION: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    transcriber = WhisperTranscriber(
        model_name=settings.WHISPER_MODEL_NAME,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        language=settings.WHISPER_LANGUAGE,
    )
    app.state.transcriber = transcriber
    app.state.engine = InterviewEngine(transcriber=transcriber)
    app.state.tts_service = OpenAITTSService.from_settings(settings)
    yield


def _sid(x_session_id: str) -> str:
    sid = (x_session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session id (send header: X-Session-Id)")
    return sid


def _public_state(st: Dict[str, Any]) -> Dict[str, Any]:
    # If summary exists, expose it so UI can download.
    summary = None
    if (st.get("summary_data_base64") or ""):
        summary = {
            "filename": st.get("summary_filename") or "interview_summary.json",
            "content_type": st.get("summary_content_type") or "application/json",
            "data_base64": st.get("summary_data_base64") or "",
        }

    return {
        "status": "ok",
        "phase": st.get("phase") or "",
        "turn": int(st.get("turn") or 0),

        "current_level": int(st.get("current_level") or 1),
        "last_passed_level": int(st.get("last_passed_level") or 0),
        "batch_level": int(st.get("batch_level") or 0),
        "batch_index": int(st.get("batch_index") or 0),
        "batch_size": int(st.get("batch_size") or 3),
        "batch_correct": int(st.get("batch_correct") or 0),

        "interview_done": bool(st.get("interview_done") or False),
        "final_level": int(st.get("final_level") or 0),

        "current_question": st.get("current_question") or "",
        "root_question": st.get("root_question") or "",
        "assistant_text": st.get("assistant_text") or "",
        "transcript": (st.get("text") or "").strip(),
        "current_item_id": st.get("current_item_id") or "",
        "current_context": st.get("current_context") or "",
        "current_objective": st.get("current_objective") or "",
        "followups_used": int(st.get("followups_used") or 0),
        "max_followups": int(st.get("max_followups") or 0),
        "last_correct": bool(st.get("last_correct") or False),
        "last_reason": st.get("last_reason") or "",
        "started_at": st.get("started_at") or "",
        "finished_at": st.get("finished_at") or "",
        "turns_log": list(st.get("turns_log") or []),

        "can_proceed": bool(st.get("can_proceed") or False),
        "allowed_actions": list(st.get("allowed_actions") or []),

        "summary": summary,
        "tts_enabled": bool(settings.TTS_ENABLED and settings.OPENAI_API_KEY),
    }


class QuestionTTSRequest(BaseModel):
    text: str = ""


async def _ensure_session_initialized(app: FastAPI, sid: str) -> Dict[str, Any]:
    cached = PUBLIC_STATE_BY_SESSION.get(sid)
    if cached is not None:
        return cached

    engine: InterviewEngine = app.state.engine
    st = await engine.init(thread_id=sid)
    pub = _public_state(dict(st))
    PUBLIC_STATE_BY_SESSION[sid] = pub
    return pub


async def _stream_text_as_ndjson(
    *,
    event_type: str,
    text: str,
    chunk_size: int = 12,
    delay_s: float = 0.01,
) -> AsyncIterator[str]:
    """
    Pseudo-streaming: chunk existing text into small pieces.
    NDJSON lines like {"type": "...", "token": "..."}.
    """
    text = text or ""
    if not text:
        return
        yield  # make it an async generator

    for i in range(0, len(text), chunk_size):
        token = text[i : i + chunk_size]
        yield json.dumps({"type": event_type, "token": token}, ensure_ascii=False) + "\n"
        if delay_s:
            await asyncio.sleep(delay_s)


def create_app() -> FastAPI:
    app = FastAPI(
        title="SmartInterviewer API",
        description="Smart Interviewer API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"https?://("
            r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
            r"10(?:\.\d{1,3}){3}|"
            r"192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}|"
            r"[A-Za-z0-9-]+\.local"
            r")(:\d+)?$"
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"status": "ok", "service": "SmartInterviewer", "version": "0.1.0"}

    @app.post("/v1/session/reset")
    async def reset_session(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        PUBLIC_STATE_BY_SESSION[sid] = _public_state(dict(initial_state()))
        engine: InterviewEngine = app.state.engine
        st = await engine.init(thread_id=sid)
        PUBLIC_STATE_BY_SESSION[sid] = _public_state(dict(st))
        return {"status": "ok", "session_id": sid}

    @app.get("/v1/session/state")
    async def get_session_state(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        pub = await _ensure_session_initialized(app, sid)
        return pub

    # -------------------------
    # Non-streaming endpoints
    # -------------------------
    @app.post("/v1/interview/start")
    async def interview_start(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)
        engine: InterviewEngine = app.state.engine
        st = await engine.resume(thread_id=sid, resume_payload={"action": ClientAction.START})
        pub = _public_state(dict(st))
        PUBLIC_STATE_BY_SESSION[sid] = pub
        return pub

    @app.post("/v1/interview/answer")
    async def interview_answer(
        audio: Annotated[UploadFile, File(description="Audio file")],
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        if not audio.filename:
            raise HTTPException(status_code=400, detail="Missing audio file")
        if audio.content_type and not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail=f"Unsupported content-type: {audio.content_type}")

        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio bytes")

        engine: InterviewEngine = app.state.engine
        st = await engine.resume(
            thread_id=sid,
            resume_payload={
                "action": ClientAction.ANSWER,
                "audio_bytes": data,
                "filename": audio.filename or "",
                "content_type": audio.content_type or "",
            },
        )
        pub = _public_state(dict(st))
        PUBLIC_STATE_BY_SESSION[sid] = pub
        return pub

    @app.post("/v1/interview/next")
    async def interview_next(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        engine: InterviewEngine = app.state.engine
        st = await engine.resume(thread_id=sid, resume_payload={"action": ClientAction.NEXT})
        pub = _public_state(dict(st))
        PUBLIC_STATE_BY_SESSION[sid] = pub
        return pub

    @app.post("/v1/interview/finish")
    async def interview_finish(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        engine: InterviewEngine = app.state.engine
        st = await engine.resume(thread_id=sid, resume_payload={"action": ClientAction.FINISH})
        pub = _public_state(dict(st))
        PUBLIC_STATE_BY_SESSION[sid] = pub
        return pub

    # -------------------------
    # Streaming endpoints
    # -------------------------
    @app.post("/v1/interview/start/stream")
    async def interview_start_stream(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Start interview with streaming question generation.
        Returns NDJSON stream.
        """
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        async def generate():
            engine: InterviewEngine = app.state.engine
            final_state = None

            async for event_type, data in engine.resume_stream(
                thread_id=sid, resume_payload={"action": ClientAction.START}
            ):
                if event_type == "question_token":
                    yield json.dumps({"type": "question_token", "token": data}, ensure_ascii=False) + "\n"
                elif event_type == "values":
                    final_state = data

            if final_state:
                pub = _public_state(dict(final_state))
                PUBLIC_STATE_BY_SESSION[sid] = pub
                yield json.dumps({"type": "final_state", "data": pub}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.post("/v1/interview/next/stream")
    async def interview_next_stream(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Move to next question with streaming.
        Returns NDJSON stream.
        """
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        async def generate():
            engine: InterviewEngine = app.state.engine
            final_state = None

            async for event_type, data in engine.resume_stream(
                thread_id=sid, resume_payload={"action": ClientAction.NEXT}
            ):
                if event_type == "question_token":
                    yield json.dumps({"type": "question_token", "token": data}, ensure_ascii=False) + "\n"
                elif event_type == "values":
                    final_state = data

            if final_state:
                pub = _public_state(dict(final_state))
                PUBLIC_STATE_BY_SESSION[sid] = pub
                yield json.dumps({"type": "final_state", "data": pub}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.post("/v1/interview/answer/stream")
    async def interview_answer_stream(
        audio: Annotated[UploadFile, File(description="Audio file")],
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Submit answer with streaming evaluation feedback.
        Returns NDJSON stream.
        """
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        if not audio.filename:
            raise HTTPException(status_code=400, detail="Missing audio file")
        if audio.content_type and not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail=f"Unsupported content-type: {audio.content_type}")

        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio bytes")

        async def generate():
            engine: InterviewEngine = app.state.engine
            final_state = None

            async for event_type, event_data in engine.resume_stream(
                thread_id=sid,
                resume_payload={
                    "action": ClientAction.ANSWER,
                    "audio_bytes": data,
                    "filename": audio.filename or "",
                    "content_type": audio.content_type or "",
                },
            ):
                if event_type == "evaluation_token":
                    yield json.dumps({"type": "evaluation_token", "token": event_data}, ensure_ascii=False) + "\n"
                elif event_type == "followup_token":
                    yield json.dumps({"type": "followup_token", "token": event_data}, ensure_ascii=False) + "\n"
                elif event_type == "values":
                    final_state = event_data

            if final_state:
                pub = _public_state(dict(final_state))
                PUBLIC_STATE_BY_SESSION[sid] = pub
                yield json.dumps({"type": "final_state", "data": pub}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.post("/v1/tts/question")
    async def question_tts(
        payload: QuestionTTSRequest,
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        tts_service: OpenAITTSService = app.state.tts_service
        audio_path = await anyio.to_thread.run_sync(tts_service.synthesize_to_path, payload.text)
        if audio_path is None:
            return Response(status_code=204)

        return FileResponse(
            path=audio_path,
            media_type=tts_service.media_type,
            filename=audio_path.name,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # -------------------------
    # Errors
    # -------------------------
    @app.exception_handler(HTTPException)
    async def http_exc_handler(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return app
