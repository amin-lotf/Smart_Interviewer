# smart_interviewer/app.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from starlette.responses import JSONResponse, StreamingResponse

from smart_interviewer.core import WhisperTranscriber, InterviewEngine, initial_state, ClientAction, _best_effort_suffix
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
    app.state.transcriber = transcriber  # Store transcriber separately
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

    @app.post("/v1/interview/start/stream")
    async def interview_start_stream(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Streams question generation as tokens arrive.
        Format: {"token": "..."} lines followed by {"final_state": {...}}
        """
        import json
        from smart_interviewer.core import _generate_question_for_item_stream, _extract_previous_questions

        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        async def generate():
            # Get current state to determine what question to ask
            engine: InterviewEngine = app.state.engine
            config = {"configurable": {"thread_id": sid}}

            # Advance the graph to generate question context
            st = await engine.resume(thread_id=sid, resume_payload={"action": ClientAction.START})

            level = int(st.get("current_level") or 1)
            turn = int(st.get("turn") or 0)
            item_id = str(st.get("current_item_id") or "")
            context = str(st.get("current_context") or "")
            objective = str(st.get("current_objective") or "")
            messages = list(st.get("messages") or [])
            prev_questions = _extract_previous_questions(messages, limit=8)

            # Stream question generation
            async for token in _generate_question_for_item_stream(
                level=level,
                turn=turn,
                item_id=item_id,
                context=context,
                objective=objective,
                prev_questions=prev_questions,
            ):
                yield json.dumps({"token": token}) + "\n"

            # Send final state
            pub = _public_state(dict(st))
            PUBLIC_STATE_BY_SESSION[sid] = pub
            yield json.dumps({"final_state": pub}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

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

    @app.post("/v1/interview/answer/stream")
    async def interview_answer_stream(
        audio: Annotated[UploadFile, File(description="Audio file")],
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Streams the full answer flow: transcription -> evaluation -> follow-up (if needed).
        Format: {"type": "transcript", "text": "..."}, {"type": "eval_token", "token": "..."},
                {"type": "followup_token", "token": "..."}, {"final_state": {...}}
        """
        import json
        from smart_interviewer.core import _evaluate_answer_stream

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
            # 1. Transcribe (using existing non-streaming method for simplicity)
            import anyio
            suffix = _best_effort_suffix(audio.filename or "", audio.content_type or "")

            # Use the transcriber stored in app.state
            transcriber: WhisperTranscriber = app.state.transcriber
            full_text = await anyio.to_thread.run_sync(transcriber.transcribe_bytes, data, suffix)
            yield json.dumps({"type": "transcript", "text": full_text}) + "\n"

            # 2. Get current question context from graph state
            engine: InterviewEngine = app.state.engine
            config = {"configurable": {"thread_id": sid}}
            state_snapshot = await engine._graph.aget_state(config)
            st = state_snapshot.values if state_snapshot else {}

            current_q = str(st.get("current_question") or "")
            level = int(st.get("current_level") or 1)
            ctx = str(st.get("current_context") or "")
            obj = str(st.get("current_objective") or "")

            # 3. Stream evaluation
            eval_json = ""
            async for token in _evaluate_answer_stream(
                level=level,
                question=current_q,
                answer=full_text,
                context=ctx,
                objective=obj,
            ):
                eval_json += token
                yield json.dumps({"type": "eval_token", "token": token}) + "\n"

            # 4. Parse evaluation and check for follow-up
            try:
                eval_data = json.loads(eval_json)
                verdict = eval_data.get("verdict", "incorrect")
                reason = eval_data.get("reason", "")
                next_q = eval_data.get("next_question", "")

                yield json.dumps({"type": "evaluation", "verdict": verdict, "reason": reason}) + "\n"

                # If needs_more and has follow-up, stream it
                if verdict == "needs_more" and next_q:
                    yield json.dumps({"type": "followup_question", "question": next_q}) + "\n"
            except:
                pass

            # 5. Process through engine to update state
            final_st = await engine.resume(
                thread_id=sid,
                resume_payload={
                    "action": ClientAction.ANSWER,
                    "audio_bytes": data,
                    "filename": audio.filename or "",
                    "content_type": audio.content_type or "",
                },
            )
            pub = _public_state(dict(final_st))
            PUBLIC_STATE_BY_SESSION[sid] = pub

            # Send final state
            yield json.dumps({"final_state": pub}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

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

    @app.post("/v1/interview/next/stream")
    async def interview_next_stream(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Streams next question generation as tokens arrive.
        Format: {"token": "..."} lines followed by {"final_state": {...}}
        """
        import json
        from smart_interviewer.core import _generate_question_for_item_stream, _extract_previous_questions

        sid = _sid(x_session_id)
        await _ensure_session_initialized(app, sid)

        async def generate():
            engine: InterviewEngine = app.state.engine
            st = await engine.resume(thread_id=sid, resume_payload={"action": ClientAction.NEXT})

            # Check if interview is done
            if st.get("interview_done"):
                pub = _public_state(dict(st))
                PUBLIC_STATE_BY_SESSION[sid] = pub
                yield json.dumps({"final_state": pub}) + "\n"
                return

            # Stream question generation
            level = int(st.get("current_level") or 1)
            turn = int(st.get("turn") or 0)
            item_id = str(st.get("current_item_id") or "")
            context = str(st.get("current_context") or "")
            objective = str(st.get("current_objective") or "")
            messages = list(st.get("messages") or [])
            prev_questions = _extract_previous_questions(messages, limit=8)

            async for token in _generate_question_for_item_stream(
                level=level,
                turn=turn,
                item_id=item_id,
                context=context,
                objective=objective,
                prev_questions=prev_questions,
            ):
                yield json.dumps({"token": token}) + "\n"

            pub = _public_state(dict(st))
            PUBLIC_STATE_BY_SESSION[sid] = pub
            yield json.dumps({"final_state": pub}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.post("/v1/interview/evaluate/stream")
    async def interview_evaluate_stream(
        x_session_id: Annotated[str, Header(alias="X-Session-Id")] = "",
    ):
        """
        Streams evaluation feedback as tokens arrive.
        This should be called AFTER transcription completes.
        Format: {"token": "..."} lines followed by {"final_state": {...}}
        """
        import json
        from smart_interviewer.core import _evaluate_answer_stream

        sid = _sid(x_session_id)
        cached = PUBLIC_STATE_BY_SESSION.get(sid)
        if not cached:
            raise HTTPException(status_code=400, detail="No active session state")

        async def generate():
            # Get current question/answer from cached state
            q = cached.get("current_question", "")
            transcript = cached.get("transcript", "")
            level = cached.get("current_level", 1)

            if not q or not transcript:
                yield json.dumps({"error": "Missing question or transcript"}) + "\n"
                return

            # Get context from session (need to fetch from engine state)
            engine: InterviewEngine = app.state.engine
            config = {"configurable": {"thread_id": sid}}
            state_snapshot = await engine._graph.aget_state(config)
            st = state_snapshot.values if state_snapshot else {}

            ctx = str(st.get("current_context") or "")
            obj = str(st.get("current_objective") or "")

            # Stream evaluation JSON
            accumulated = ""
            async for token in _evaluate_answer_stream(
                level=level,
                question=q,
                answer=transcript,
                context=ctx,
                objective=obj,
            ):
                accumulated += token
                yield json.dumps({"token": token}) + "\n"

            # Parse and send structured feedback
            try:
                data = json.loads(accumulated)
                verdict = data.get("verdict", "incorrect")
                reason = data.get("reason", "")
                yield json.dumps({"evaluation": {"verdict": verdict, "reason": reason}}) + "\n"
            except:
                yield json.dumps({"evaluation": {"verdict": "incorrect", "reason": "Could not parse evaluation"}}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

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
