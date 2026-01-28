# Smart Interviewer Tests

This folder contains the test suite for Smart Interviewer.

These are **black-box integration tests** against the FastAPI app, using **mocked** speech-to-text and LLM components so tests are:
- fast
- deterministic
- cost-free (no OpenAI calls)
- CI-friendly (no GPU required)

## Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and shared fixtures
├── README.md                # This file
├── mocks/                   # Mock implementations for testing
│   ├── __init__.py
│   ├── llm.py              # Mock LLM (no real API calls)
│   └── transcriber.py      # Mock Whisper transcriber (no real model)
└── integration/             # Integration tests
    ├── __init__.py
    └── test_api.py         # FastAPI endpoint tests
```

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=smart_interviewer --cov-report=html
```

### Run specific test file

```bash
pytest tests/integration/test_api.py
```

### Run specific test class or function

```bash
pytest tests/integration/test_api.py::TestSessionEndpoints
pytest tests/integration/test_api.py::TestSessionEndpoints::test_reset_session_with_session_id
```

### Run with verbose output

```bash
pytest -v
```

### Run with print statements visible

```bash
pytest -s
```

## What’s covered

The integration tests cover:

- ✅ Root endpoint (`/`)
- ✅ Session management (`/v1/session/*`)
  - Reset session
  - Get session state
- ✅ Interview flow (`/v1/interview/*`)
  - Start interview
  - Answer questions (with audio upload)
  - Next question
  - Finish interview
- ✅ Complete interview workflow
- ✅ Error handling
  - Missing session IDs
  - Invalid audio files
  - Empty inputs

## Mock Components

### MockWhisperTranscriber

Replaces the real Whisper model to avoid:
- Loading large model files
- GPU/CPU intensive transcription
- Network calls

Returns predefined transcription text instantly.

### MockLLM

Replaces the real LangChain LLM to avoid:
- API calls to OpenAI or other providers
- Token costs
- Network latency

Returns predefined responses based on keyword matching.

## Adding New Tests

1. Create test file in appropriate directory (e.g., `tests/integration/test_new_feature.py`)
2. Import necessary fixtures from `conftest.py`
3. Use mock components to avoid real API/model calls
4. Follow naming convention: `test_*` for files and functions, `Test*` for classes
5. Use descriptive test names that explain what is being tested

Example:

```python
import pytest

async def test_new_endpoint(client, test_session_id):
    """Test description."""
    response = await client.get(
        "/v1/new/endpoint",
        headers={"X-Session-Id": test_session_id}
    )
    assert response.status_code == 200
```

## Notes

- All tests use mocked LLM and Whisper to ensure fast, deterministic, and cost-free execution
- Tests are async-aware using `pytest-asyncio`
- Session state is isolated per test using unique session IDs
- Audio uploads are tested with mock binary data
