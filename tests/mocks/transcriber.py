"""Mock Whisper transcriber for testing."""
from typing import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class MockWhisperTranscriber:
    """
    Mock transcriber that returns predefined responses without calling real Whisper.

    Attributes:
        model_name: Mock model name (not used)
        device: Mock device (not used)
        compute_type: Mock compute type (not used)
        language: Mock language (not used)
        mock_response: Default response to return
    """
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    mock_response: str = "This is a mock transcription"
    _model: None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Mock initialization - does nothing."""
        pass

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
        """
        Mock transcription that returns predefined text.

        Args:
            audio_bytes: Audio data (ignored)
            suffix: File extension (ignored)

        Returns:
            Mock transcription text
        """
        if not audio_bytes:
            return ""
        return self.mock_response

    async def transcribe_bytes_stream(self, audio_bytes: bytes, suffix: str = ".webm") -> AsyncIterator[str]:
        """
        Mock streaming transcription.

        Args:
            audio_bytes: Audio data (ignored)
            suffix: File extension (ignored)

        Yields:
            Mock transcription segments
        """
        if not audio_bytes:
            return

        # Split mock response into words to simulate streaming
        words = self.mock_response.split()
        for word in words:
            yield word
