# smart_interviewer/app.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from starlette.responses import JSONResponse

from smart_interviewer.core import WhisperTranscriber, InterviewEngine, initial_state, ClientAction
from smart_interviewer.settings import settings

logger = logging.getLogger("smart_interviewer")

# Keep only a lightweight cache of the *latest* public state per session
# (LangGraph checkpointer already persists the real state per thread_id)
PUBLIC_STATE_BY_SESSION: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    transcriber = WhisperTranscriber(
        model_name=settings.WHISPER_MODEL_NAME,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        language=settings.WHISPER_LANGUAGE,
    )
    app.state.engine = InterviewEngine(transcriber=transcriber)
    yield


def _sid(x_session_id: str) -> str:
    sid = (x_session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session id (send header: X-Session-Id)")
    return sid

def _extract_interrupt_payload(st: Dict[str, Any]) -> Dict[str, Any] | None:
    for k in ("__interrupt__", "interrupt", "interrupts"):
        payload = st.get(k)
        if payload is None:
            continue
        if isinstance(payload, list) and payload:
            return payload[-1] if isinstance(payload[-1], dict) else None
        if isinstance(payload, dict):
            return payload
    return None

def _public_state(st: Dict[str, Any]) -> Dict[str, Any]:
    intr = _extract_interrupt_payload(st)

    download = None
    if isinstance(intr, dict) and intr.get("type") == "download_file":
        f = intr.get("file")
        if isinstance(f, dict):
            download = {
                "filename": f.get("filename") or "interview_summary.json",
                "content_type": f.get("content_type") or "application/json",
                "data_base64": f.get("data_base64") or "",
            }

    allowed = list(st.get("allowed_actions") or [])
    assistant_text = st.get("assistant_text") or ""

    # ✅ critical: if we're in download interrupt, force FINISH UI state
    if download is not None:
        allowed = [ClientAction.FINISH]
        assistant_text = intr.get("message") or assistant_text

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
        "assistant_text": assistant_text,
        "transcript": (st.get("text") or "").strip(),

        "can_proceed": bool(st.get("can_proceed") or False),
        "allowed_actions": allowed,

        "interrupt": intr,
        "download": download,
        "summary": {
            "filename": st.get("summary_filename") or "",
            "content_type": st.get("summary_content_type") or "",
            "data_base64": st.get("summary_data_base64") or "",
        } if (st.get("summary_data_base64") or "") else None,
    }


async def _ensure_session_initialized(app: FastAPI, sid: str) -> Dict[str, Any]:
    """
    Ensures the graph has been invoked at least once for this sid (thread_id),
    so it reaches the first interrupt (wait_start) and returns a UI-friendly state.
    """
    cached = PUBLIC_STATE_BY_SESSION.get(sid)
    if cached is not None:
        return cached

    engine: InterviewEngine = app.state.engine
    st = await engine.init(thread_id=sid)  # will stop at interrupt wait_start
    pub = _public_state(dict(st))
    PUBLIC_STATE_BY_SESSION[sid] = pub
    return pub


def create_app() -> FastAPI:
    app = FastAPI(
        title="SmartInterviewer API",
        description="Smart Interviewer API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {"status": "ok", "service": "SmartInterviewer", "version": "0.1.0"}

    @app.post("/v1/session/reset")
    async def reset_session(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        sid = _sid(x_session_id)

        # Reset UI cache
        PUBLIC_STATE_BY_SESSION[sid] = _public_state(dict(initial_state()))

        # Also re-init the graph for this thread_id by calling init (it will set the state again)
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

    @app.exception_handler(HTTPException)
    async def http_exc_handler(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return app
