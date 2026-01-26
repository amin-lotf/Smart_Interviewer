import tempfile
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator

from faster_whisper import WhisperModel


@dataclass
class WhisperTranscriber:
    """
    Wrapper around faster-whisper for audio transcription.

    Converts audio bytes to text using OpenAI's Whisper model.

    Attributes:
        model_name: Whisper model size (tiny/base/small/medium/large)
        device: Compute device (cpu/cuda)
        compute_type: Quantization type (int8/float16/float32)
        language: Target language code (e.g., 'en', 'es')
    """
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    _model: WhisperModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Whisper model after dataclass initialization."""
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data
            suffix: File extension hint for audio format

        Returns:
            Transcribed text string
        """
        if not audio_bytes:
            return ""
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as f:
            f.write(audio_bytes)
            f.flush()

            segments, _info = self._model.transcribe(
                f.name,
                language=self.language,
                vad_filter=True,
            )

            parts: list[str] = []
            for seg in segments:
                t = (seg.text or "").strip()
                if t:
                    parts.append(t)
            return " ".join(parts).strip()

    async def transcribe_bytes_stream(self, audio_bytes: bytes, suffix: str = ".webm") -> AsyncIterator[str]:
        """
        Streams transcription segments as they are produced by Whisper.
        Yields individual text segments.
        """
        if not audio_bytes:
            return

        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as f:
            f.write(audio_bytes)
            f.flush()

            segments, _info = self._model.transcribe(
                f.name,
                language=self.language,
                vad_filter=True,
            )

            for seg in segments:
                t = (seg.text or "").strip()
                if t:
                    yield t