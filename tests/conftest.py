"""Pytest configuration and fixtures."""
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from tests.mocks.transcriber import MockWhisperTranscriber
from tests.mocks.llm import create_mock_llm_with_responses


@pytest.fixture
def mock_transcriber():
    """Create a mock Whisper transcriber."""
    return MockWhisperTranscriber(
        mock_response="This is my answer to the question"
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM with predefined responses."""
    return create_mock_llm_with_responses({
        "generate": "What is the time complexity of binary search?",
        "evaluate": "CORRECT: The answer demonstrates understanding.",
        "grade": "pass",
        "summary": "Interview completed successfully. Final level: 3",
    })


@pytest.fixture
def test_session_id():
    """Provide a consistent test session ID."""
    return "test-session-123"


@pytest.fixture
def audio_file_bytes():
    """Create mock audio file bytes for testing."""
    # Return some dummy bytes that represent an audio file
    return b"mock-audio-data-webm-format" * 100
