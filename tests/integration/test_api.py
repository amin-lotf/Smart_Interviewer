"""Integration tests for FastAPI endpoints."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import ASGITransport, AsyncClient
from io import BytesIO

from smart_interviewer.app import create_app
from smart_interviewer.core import initial_state, InterviewEngine
from tests.mocks.transcriber import MockWhisperTranscriber


@pytest.fixture
async def mock_transcriber():
    """Create a mock transcriber."""
    return MockWhisperTranscriber(mock_response="This is a mock transcription")


@pytest.fixture
def mock_interview_engine():
    """Mock the InterviewEngine to avoid real LLM calls."""
    from smart_interviewer.core.types import AgentPhase, ClientAction

    mock_engine = MagicMock()

    # Mock init method
    async def mock_init(thread_id):
        state = initial_state()
        return state

    # Mock resume method
    async def mock_resume(thread_id, resume_payload):
        action = resume_payload.get("action")
        state = initial_state()

        if action == ClientAction.START:
            state["phase"] = AgentPhase.AWAITING_ANSWER
            state["current_question"] = "What is binary search?"
            state["assistant_text"] = "Let's begin with the first question."
            state["can_proceed"] = True
            state["allowed_actions"] = [ClientAction.ANSWER, ClientAction.FINISH]
        elif action == ClientAction.ANSWER:
            state["phase"] = AgentPhase.EVALUATED
            state["text"] = "Mock transcription of audio"
            state["last_correct"] = True
            state["assistant_text"] = "Correct! Moving to next question."
            state["can_proceed"] = True
            state["allowed_actions"] = [ClientAction.NEXT, ClientAction.FINISH]
        elif action == ClientAction.NEXT:
            state["phase"] = AgentPhase.AWAITING_ANSWER
            state["current_question"] = "What is the time complexity?"
            state["assistant_text"] = "Next question."
            state["can_proceed"] = True
            state["allowed_actions"] = [ClientAction.ANSWER, ClientAction.FINISH]
        elif action == ClientAction.FINISH:
            state["phase"] = AgentPhase.DONE
            state["interview_done"] = True
            state["final_level"] = 3
            state["assistant_text"] = "Interview completed!"
            state["allowed_actions"] = []

        return state

    mock_engine.init = mock_init
    mock_engine.resume = mock_resume
    return mock_engine


@pytest.fixture
async def app(mock_transcriber, mock_interview_engine):
    """Create FastAPI app for testing with mocked dependencies."""
    # Patch InterviewEngine before creating the app
    with patch('smart_interviewer.core.engine.InterviewEngine.__init__', return_value=None):
        app = create_app()

        # Manually set app state since lifespan isn't triggered in tests
        app.state.transcriber = mock_transcriber
        app.state.engine = mock_interview_engine

        yield app


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestRootEndpoint:
    """Tests for root endpoint."""

    async def test_root(self, client):
        """Test root endpoint returns service info."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "SmartInterviewer"
        assert "version" in data


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    async def test_reset_session_without_session_id(self, client):
        """Test reset session fails without session ID."""
        response = await client.post("/v1/session/reset")
        assert response.status_code == 400
        assert "session id" in response.json()["detail"].lower()

    async def test_reset_session_with_session_id(self, client, test_session_id):
        """Test reset session creates new session."""
        response = await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["session_id"] == test_session_id

    async def test_get_session_state_without_session_id(self, client):
        """Test get state fails without session ID."""
        response = await client.get("/v1/session/state")
        assert response.status_code == 400

    async def test_get_session_state_with_session_id(self, client, test_session_id):
        """Test get session state returns current state."""
        # First reset to initialize
        await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )

        # Then get state
        response = await client.get(
            "/v1/session/state",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "phase" in data
        assert "assistant_text" in data


class TestInterviewEndpoints:
    """Tests for interview flow endpoints."""

    async def test_start_interview(self, client, test_session_id):
        """Test starting an interview."""
        # Initialize session
        await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )

        # Start interview
        response = await client.post(
            "/v1/interview/start",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["phase"] == "AWAITING_ANSWER"
        assert data["current_question"]
        assert data["can_proceed"] is True

    async def test_answer_interview_without_audio(self, client, test_session_id):
        """Test answering fails without audio file."""
        response = await client.post(
            "/v1/interview/answer",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 422  # Validation error

    async def test_answer_interview_with_audio(
        self, client, test_session_id, audio_file_bytes
    ):
        """Test answering with audio file."""
        # Initialize and start
        await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )
        await client.post(
            "/v1/interview/start",
            headers={"X-Session-Id": test_session_id}
        )

        # Submit answer
        files = {"audio": ("test.webm", BytesIO(audio_file_bytes), "audio/webm")}
        response = await client.post(
            "/v1/interview/answer",
            headers={"X-Session-Id": test_session_id},
            files=files
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "text" in data or "transcript" in data

    async def test_next_question(self, client, test_session_id):
        """Test moving to next question."""
        # Initialize session
        await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )

        # Move to next
        response = await client.post(
            "/v1/interview/next",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "current_question" in data

    async def test_finish_interview(self, client, test_session_id):
        """Test finishing an interview."""
        # Initialize session
        await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )

        # Finish interview
        response = await client.post(
            "/v1/interview/finish",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["interview_done"] is True
        assert data["final_level"] > 0


class TestInterviewFlow:
    """Test complete interview flow."""

    async def test_complete_interview_flow(
        self, client, test_session_id, audio_file_bytes
    ):
        """Test a complete interview workflow from start to finish."""
        # 1. Reset session
        response = await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200

        # 2. Start interview
        response = await client.post(
            "/v1/interview/start",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        assert response.json()["phase"] == "AWAITING_ANSWER"

        # 3. Submit answer
        files = {"audio": ("answer.webm", BytesIO(audio_file_bytes), "audio/webm")}
        response = await client.post(
            "/v1/interview/answer",
            headers={"X-Session-Id": test_session_id},
            files=files
        )
        assert response.status_code == 200

        # 4. Get next question
        response = await client.post(
            "/v1/interview/next",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200

        # 5. Finish interview
        response = await client.post(
            "/v1/interview/finish",
            headers={"X-Session-Id": test_session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["interview_done"] is True


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_missing_session_id_header(self, client):
        """Test endpoints fail gracefully without session ID."""
        endpoints = [
            "/v1/session/reset",
            "/v1/session/state",
            "/v1/interview/start",
            "/v1/interview/next",
            "/v1/interview/finish",
        ]

        for endpoint in endpoints:
            if endpoint == "/v1/session/state":
                response = await client.get(endpoint)
            else:
                response = await client.post(endpoint)
            assert response.status_code == 400

    async def test_empty_session_id(self, client):
        """Test endpoints reject empty session IDs."""
        response = await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": "   "}
        )
        assert response.status_code == 400

    async def test_invalid_audio_content_type(
        self, client, test_session_id
    ):
        """Test answer endpoint rejects non-audio files."""
        await client.post(
            "/v1/session/reset",
            headers={"X-Session-Id": test_session_id}
        )

        # Try to upload text file as audio
        files = {"audio": ("test.txt", BytesIO(b"not audio"), "text/plain")}
        response = await client.post(
            "/v1/interview/answer",
            headers={"X-Session-Id": test_session_id},
            files=files
        )
        assert response.status_code == 415  # Unsupported Media Type
