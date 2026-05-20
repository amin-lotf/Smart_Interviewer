from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from openai import OpenAI

from smart_interviewer.settings import Settings

logger = logging.getLogger(__name__)

FORMAT_TO_EXTENSION = {
    "mp3": ".mp3",
    "wav": ".wav",
    "aac": ".aac",
    "flac": ".flac",
    "opus": ".opus",
    "pcm": ".pcm",
}

FORMAT_TO_MEDIA_TYPE = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "pcm": "audio/L16",
}


class OpenAITTSService:
    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        model: str,
        voice: str,
        response_format: str,
        cache_dir: Path,
    ) -> None:
        self._configured_enabled = enabled
        self._api_key = (api_key or "").strip()
        self.model = (model or "").strip() or "gpt-4o-mini-tts"
        self.voice = (voice or "").strip() or "coral"
        self.response_format = (response_format or "").strip().lower() or "mp3"
        self.cache_dir = cache_dir
        self._client = OpenAI(api_key=self._api_key) if self._api_key else None

        if self._configured_enabled and not self._api_key:
            logger.warning("TTS is enabled but OPENAI_API_KEY is missing. Question playback will stay disabled.")

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAITTSService":
        cache_dir = Path(tempfile.gettempdir()) / "smart_interviewer_tts"
        return cls(
            enabled=bool(settings.TTS_ENABLED),
            api_key=settings.OPENAI_API_KEY,
            model=settings.TTS_MODEL,
            voice=settings.TTS_VOICE,
            response_format=settings.TTS_RESPONSE_FORMAT,
            cache_dir=cache_dir,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._configured_enabled and self._client)

    @property
    def media_type(self) -> str:
        return FORMAT_TO_MEDIA_TYPE.get(self.response_format, "application/octet-stream")

    def synthesize_to_path(self, text: str) -> Path | None:
        normalized = " ".join((text or "").split()).strip()
        if not normalized:
            return None

        if not self.enabled:
            return None

        extension = FORMAT_TO_EXTENSION.get(self.response_format, f".{self.response_format}")
        cache_key = hashlib.sha256(
            f"{self.model}:{self.voice}:{self.response_format}:{normalized}".encode("utf-8")
        ).hexdigest()
        target_path = self.cache_dir / f"{cache_key}{extension}"

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Failed to create TTS cache directory at %s", self.cache_dir)
            return None

        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        try:
            with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=normalized,
                response_format=self.response_format,
            ) as response:
                response.stream_to_file(temp_path)
            temp_path.replace(target_path)
            return target_path
        except Exception:
            logger.exception("OpenAI TTS generation failed for %s characters of question text.", len(normalized))
            temp_path.unlink(missing_ok=True)
            return None
