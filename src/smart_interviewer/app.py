import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from smart_interviewer.core import WhisperTranscriber, build_voice_graph
from smart_interviewer.schemas import VoiceTranscriptionResponse
from smart_interviewer.settings import settings

logger = logging.getLogger("smart_interviewer")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pick your defaults (later move to settings/env)
    transcriber = WhisperTranscriber(
        model_name=settings.WHISPER_MODEL_NAME,
        device=settings.WHISPER_DEVICE,  # change to "cuda" if GPU is ready
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        language=settings.WHISPER_LANGUAGE,
    )
    app.state.voice_graph = build_voice_graph(transcriber=transcriber)
    yield
    # cleanup later


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

    # -------------------------
    # Voice ingestion (future-proof)
    # -------------------------
    @app.post("/v1/voice/upload/stream")
    async def upload_voice_stream(
            audio: Annotated[UploadFile, File(description="Audio file")],
    ):
        data = await audio.read()
        graph = app.state.voice_graph

        async def event_gen():
            sent_transcript = False
            sent_assistant = False

            # IMPORTANT: stream_mode="values" yields the FULL state after each step
            async for state in graph.astream(
                    {
                        "audio_bytes": data,
                        "filename": audio.filename or "",
                        "content_type": audio.content_type or "",
                    },
                    stream_mode="values",
            ):
                # Debug: uncomment once if needed
                # logger.info("GRAPH STATE: %s", state)

                # 1) Emit transcript as soon as it exists
                text = (state.get("text") or "").strip()
                if text and not sent_transcript:
                    ev = {"type": "transcript", "text": text}
                    yield "event: transcript\n"
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    sent_transcript = True

                # 2) If you store assistant output in state, emit it here
                # If your fake_llm currently uses events[] instead, change fake_llm to set assistant_text (shown below)
                assistant_text = (state.get("assistant_text") or "").strip()
                if assistant_text and not sent_assistant:
                    ev = {"type": "assistant", "text": assistant_text}
                    yield "event: assistant\n"
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    sent_assistant = True

            yield "event: done\n"
            yield "data: {}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 1) Multipart file upload (best for Streamlit mic -> file-like -> upload)
    @app.post("/v1/voice/upload", response_model=VoiceTranscriptionResponse)
    async def upload_voice(
            audio: Annotated[UploadFile, File(description="Audio file (wav/mp3/webm/m4a/ogg, etc.)")],
    ):
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Missing audio file")

        if audio.content_type and not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail=f"Unsupported content-type: {audio.content_type}")

        data = await audio.read()
        size_bytes = len(data)

        logger.info(
            "Received audio upload: filename=%s content_type=%s size=%d bytes",
            audio.filename,
            audio.content_type,
            size_bytes,
        )

        compiled = app.state.voice_graph

        final_state = await compiled.ainvoke(
            {
                "audio_bytes": data,
                "filename": audio.filename or "",
                "content_type": audio.content_type or "",
            }
        )

        text = (final_state.get("text") or "").strip()

        return VoiceTranscriptionResponse(
            status="ok",
            text=text,
            filename=audio.filename,
            content_type=audio.content_type,
            size_bytes=size_bytes,
        )

    # 2) Raw bytes endpoint (useful for non-browser clients or later WS pipelines)
    @app.post("/v1/voice/raw")
    async def raw_voice(request: Request):
        content_type = request.headers.get("content-type", "")
        body = await request.body()

        if not body:
            raise HTTPException(status_code=400, detail="Empty request body")

        logger.info(
            "Received raw audio: content_type=%s size=%d bytes",
            content_type,
            len(body),
        )

        return {
            "status": "ok",
            "endpoint": "raw",
            "content_type": content_type,
            "size_bytes": len(body),
        }

    # (Optional) a tiny centralized error response shape
    @app.exception_handler(HTTPException)
    async def http_exc_handler(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app

