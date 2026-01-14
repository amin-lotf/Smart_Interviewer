import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from starlette.responses import JSONResponse

logger = logging.getLogger("smart_interviewer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init resources later (models, clients, etc.)
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

    # 1) Multipart file upload (best for Streamlit mic -> file-like -> upload)
    @app.post("/v1/voice/upload")
    async def upload_voice(
        audio: Annotated[UploadFile, File(description="Audio file (wav/mp3/webm/m4a/ogg, etc.)")],
    ):
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Missing audio file")

        # Optional guardrails (keep them lenient for now)
        # - you can tighten later when you know your STT pipeline expectations
        allowed_prefixes = ("audio/",)
        if audio.content_type and not audio.content_type.startswith(allowed_prefixes):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported content-type: {audio.content_type}",
            )

        # Don’t read the whole file into memory if you don’t need to.
        # But reading once is fine for MVP; we’ll just compute the size for debugging.
        data = await audio.read()
        size_bytes = len(data)

        logger.info(
            "Received audio upload: filename=%s content_type=%s size=%d bytes",
            audio.filename,
            audio.content_type,
            size_bytes,
        )

        return {
            "status": "ok",
            "endpoint": "upload",
            "filename": audio.filename,
            "content_type": audio.content_type,
            "size_bytes": size_bytes,
        }

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
