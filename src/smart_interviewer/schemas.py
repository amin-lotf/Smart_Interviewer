from pydantic import BaseModel, Field


class VoiceTranscriptionResponse(BaseModel):
    status: str = "ok"
    text: str = Field(..., description="Transcribed text of the user audio")
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
