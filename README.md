# Smart Interviewer

An AI-powered adaptive interviewing system that evaluates candidates through voice interactions with streaming LLM responses. The system automatically adjusts difficulty based on performance and provides real-time feedback with follow-up questions.

## Why Smart Interviewer?

This project demonstrates how to build:
- AI Interview agents with real-time evaluation
- Deterministic evaluation logic (not chatty demos)
- Testable, production-oriented LLM workflows

It is designed as a portfolio-grade MVP for:
- AI interviewing platforms
- HR automation tools
- Voice-based assessment systems

## Features

- **🎤 Voice-Based Interviews**: Record answers using your microphone
- **🤖 AI Evaluation**: GPT-4o-mini evaluates answers with detailed feedback
- **📊 Adaptive Difficulty**: Automatically progresses through levels based on performance
- **🔄 Real-Time Streaming**: LLM responses stream token-by-token for smooth UX
- **❓ Follow-Up Questions**: System asks clarifying questions for incomplete answers
- **📝 Interview Transcripts**: Complete interview logs with all attempts and evaluations
- **🎯 Customizable Question Banks**: Easy-to-edit markdown format for questions

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← User Interface
└────────┬────────┘
         │
┌────────▼────────┐
│  FastAPI Server │  ← REST API + Streaming
└────────┬────────┘
         │
┌────────▼────────┐
│  LangGraph Core │  ← Interview State Machine
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼─────┐
│Whisper│ │OpenAI  │  ← AI Services
│(STT)  │ │(LLM)   │
└───────┘ └────────┘
```

## Technology Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Frontend**: Streamlit
- **AI Models**:
  - OpenAI GPT-4o-mini (question generation & evaluation)
  - faster-whisper (speech-to-text)
- **State Management**: LangGraph with in-memory checkpointing

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key
- Microphone access (for voice input)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smart-interviewer.git
   cd smart-interviewer
   ```

2. **Install dependencies** (using uv)
   ```bash
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Configure environment variables**

   Copy the `.env.example` file (or create `.env`) and add your OpenAI API key:
   ```bash
   # API Keys
   OPENAI_API_KEY=your-openai-api-key-here

   # Whisper Configuration
   WHISPER_MODEL_NAME=small
   WHISPER_DEVICE=cpu
   WHISPER_COMPUTE_TYPE=int8
   WHISPER_LANGUAGE=en

   # Interview Configuration
   QUESTIONS_PER_LEVEL=3
   MIN_PASSED_FOR_LEVEL=2
   MAX_FOLLOWUP_QUESTIONS=2

   # LLM Configuration
   LLM_MODEL=gpt-4o-mini
   LLM_TEMPERATURE=0.3
   ```

4. **Customize question bank** (optional)

   Edit `data/question_bank.md` to add your own questions:
   ```markdown
   # Level 1

   ## Item: llm-basics
   **Context:**
   Large Language Models (LLMs) are AI systems trained on vast amounts of text...

   **Objective:**
   Verify the candidate understands what LLM stands for.
   ```

## Usage

### Running the Application

1. **Start the FastAPI backend**:
   ```bash
   # Using the installed script
   interview

   # Or using Python module
   python -m smart_interviewer.main
   ```
   Server runs on `http://localhost:8000`

2. **Start the Streamlit UI** (in another terminal):
   ```bash
   streamlit run src/smart_interviewer/ui/streamlit_app.py
   ```
   UI opens at `http://localhost:8501`

### Development Mode

For development with auto-reload:
```bash
dev-interview
```

### Interview Flow

1. **Start**: Click "🚀 Start" to begin the interview
2. **Question**: Watch as the question streams in word-by-word
3. **Answer**: Record your voice answer using the microphone
4. **Evaluation**: AI evaluates and may ask follow-up questions
5. **Progress**: Level-based adaptive progression
6. **Finish**: Download complete interview transcript at the end

## Configuration

All configuration is managed through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `WHISPER_MODEL_NAME` | Whisper model size | `small` |
| `WHISPER_DEVICE` | Compute device | `cpu` |
| `LLM_MODEL` | OpenAI model | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Response randomness | `0.3` |
| `QUESTIONS_PER_LEVEL` | Questions per difficulty level | `3` |
| `MIN_PASSED_FOR_LEVEL` | Required correct answers | `2` |
| `MAX_FOLLOWUP_QUESTIONS` | Max follow-ups per answer | `2` |
| `AUDIO_SAMPLE_RATE` | Recording sample rate (Hz) | `16000` |

## API Endpoints

### Standard Endpoints
- `GET /` - Health check
- `POST /v1/session/reset` - Reset interview session
- `GET /v1/session/state` - Get current state
- `POST /v1/interview/start` - Start interview
- `POST /v1/interview/answer` - Submit audio answer
- `POST /v1/interview/next` - Move to next question
- `POST /v1/interview/finish` - End interview



Streaming endpoints return NDJSON (newline-delimited JSON) format.

## Project Structure

```
smart-interviewer/
├── src/smart_interviewer/
│   ├── main.py             # Entry point (production)
│   ├── dev.py              # Entry point (development)
│   ├── app.py              # FastAPI application
│   ├── settings.py         # Configuration management
│   ├── consts.py           # Constants
│   ├── utils.py            # Utilities
│   ├── core/               # Core interview logic
│   │   ├── engine.py       # Interview engine
│   │   ├── graph.py        # LangGraph state machine
│   │   ├── nodes.py        # Graph nodes
│   │   ├── grading.py      # Answer evaluation
│   │   ├── progression.py  # Level progression logic
│   │   ├── question_bank.py # Question loader
│   │   ├── prompts.py      # LLM prompts
│   │   ├── llm.py          # LLM configuration
│   │   ├── transcriber.py  # Whisper STT wrapper
│   │   ├── history.py      # Conversation history
│   │   ├── summary.py      # Interview summary
│   │   └── types.py        # Type definitions
│   └── ui/
│       ├── api_client.py   # API client library
│       └── streamlit_app.py # Streamlit UI
├── tests/                  # Test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── mocks/              # Mock components
│   │   ├── llm.py          # Mock LLM
│   │   └── transcriber.py  # Mock transcriber
│   └── integration/        # Integration tests
│       └── test_api.py     # API tests
├── data/
│   └── question_bank.md    # Interview questions
├── .env                    # Environment configuration
├── pyproject.toml          # Project dependencies
└── README.md
```

## Development

### Running Tests

Install test dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=smart_interviewer --cov-report=html
```

Run specific test file:
```bash
pytest tests/integration/test_api.py
```

The test suite uses mocked LLM and Whisper components for fast, deterministic, and cost-free testing. See `tests/README.md` for more details.



### Adding New Questions

Edit `data/question_bank.md`:
```markdown
# Level 2

## Item: advanced-topic
Context:
[Background information for the question]

Objective:
[What specific knowledge you're testing]
```

The system will automatically:
- Generate contextual questions
- Evaluate answers against the objective
- Ask follow-up questions if needed



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **OpenAI** for GPT models
- **faster-whisper** for efficient speech recognition
- **LangChain/LangGraph** for LLM orchestration
- **Streamlit** for rapid UI development

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an MVP project. For production use, consider adding:
- User authentication
- Database persistence
- Rate limiting
- Error monitoring
- Comprehensive test coverage
