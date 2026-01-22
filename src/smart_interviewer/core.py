# smart_interviewer/core.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, TypedDict, List, Literal, Any, cast

import anyio
from faster_whisper import WhisperModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command  # LangGraph 1.0

from smart_interviewer.settings import settings


# -----------------------------
# Enums
# -----------------------------
class AgentPhase(StrEnum):
    IDLE = "IDLE"
    AWAITING_START = "AWAITING_START"
    AWAITING_ANSWER = "AWAITING_ANSWER"
    AWAITING_NEXT = "AWAITING_NEXT"
    EVALUATED = "EVALUATED"
    DONE = "DONE"


class ClientAction(StrEnum):
    START = "START"
    NEXT = "NEXT"
    ANSWER = "ANSWER"
    RETRY = "RETRY"


# -----------------------------
# State
# -----------------------------
class InterviewState(TypedDict, total=False):
    # audio in (only for ANSWER resume)
    audio_bytes: bytes
    filename: str
    content_type: str

    # transcript (latest)
    text: str

    # interview control
    phase: AgentPhase
    current_question: str
    score: int
    turn: int  # number of questions already asked (or next turn index)

    # policy / ui helpers
    can_proceed: bool
    allowed_actions: List[ClientAction]

    # output for UI
    assistant_text: str

    # internal: last evaluation
    last_correct: bool
    last_reason: str


# -----------------------------
# Scoring helper
# -----------------------------
def update_score(delta: int, reason: str) -> dict:
    d = int(delta)
    if d not in (-1, 1):
        d = 1 if d > 0 else -1
    return {"delta": d, "reason": str(reason)[:300]}


# -----------------------------
# Transcriber
# -----------------------------
@dataclass
class WhisperTranscriber:
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    _model: WhisperModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
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


# -----------------------------
# LLM setup (LangChain 1.0)
# -----------------------------
LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=settings.OPENAI_API_KEY,
)

ASK_SYS = """
You are an interviewer for LLM fundamentals.
Ask exactly ONE short spoken-friendly question about LLMs.
Start easy, then slowly increase difficulty based on turn number.
Output ONLY the question text. No preamble.
"""

EVAL_SYS = (
    "You are a strict but fair grader.\n"
    "Given a question and an answer, decide if the answer is correct.\n"
    'Return JSON ONLY with keys: {"correct": true/false, "reason": "..."}\n'
    "Accept short phrase answers if they match the meaning.\n"
    "Do not ask follow-up questions. Do not include extra keys."
)


# -----------------------------
# Initial state
# -----------------------------
def initial_state() -> InterviewState:
    return {
        "phase": AgentPhase.IDLE,
        "score": 0,
        "turn": 0,
        "current_question": "",
        "assistant_text": "Press Start to begin.",
        "can_proceed": False,
        "allowed_actions": [ClientAction.START],
        "text": "",
        "last_correct": False,
        "last_reason": "",
    }


# -----------------------------
# Helpers
# -----------------------------
def _best_effort_suffix(filename: str, content_type: str) -> str:
    suffix = os.path.splitext(filename)[1].lower() if filename else ""
    if suffix:
        return suffix
    ct = (content_type or "").lower()
    if ct.endswith("webm"):
        return ".webm"
    if ct.endswith("wav"):
        return ".wav"
    if ct.endswith("mpeg") or ct.endswith("mp3"):
        return ".mp3"
    return ".bin"


def _require_action(resume: Any, expected: ClientAction) -> None:
    """
    resume can be:
      {"action": "START"}  (from button)
      {"action": "ANSWER", "audio_bytes": ..., "filename": ..., "content_type": ...}
    """
    if not isinstance(resume, dict):
        raise ValueError("Resume payload must be a dict.")
    action = str(resume.get("action") or "")
    if action != str(expected):
        raise ValueError(f"Expected action {expected}, got {action!r}.")


# -----------------------------
# Graph nodes (LangGraph 1.0)
# -----------------------------
async def node_wait_start(state: InterviewState) -> InterviewState:
    # Put UI into a consistent "awaiting start" shape, then interrupt.
    s = {
        **state,
        "phase": AgentPhase.AWAITING_START,
        "assistant_text": state.get("assistant_text") or "Press Start to begin.",
        "allowed_actions": [ClientAction.START],
        "can_proceed": False,
    }
    resume = interrupt(
        {
            "type": "await_action",
            "allowed_actions": [ClientAction.START],
            "message": s["assistant_text"],
        }
    )
    _require_action(resume, ClientAction.START)
    return {
        **s,
        # After Start, we move on to ask_question
        "phase": AgentPhase.IDLE,
    }


async def node_ask_question(state: InterviewState) -> InterviewState:
    turn = int(state.get("turn") or 0)
    score = int(state.get("score") or 0)

    msg = [
        SystemMessage(content=ASK_SYS),
        HumanMessage(content=f"Turn number: {turn}. Current score: {score}."),
    ]
    resp = await LLM.ainvoke(msg)
    q = (resp.content or "").strip() or "What does LLM stand for?"

    # turn increments when we ASK the question
    new_turn = turn + 1

    return {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "current_question": q,
        "assistant_text": q,
        "turn": new_turn,
        "can_proceed": False,
        "allowed_actions": [ClientAction.ANSWER],
        "text": "",
    }


async def node_wait_answer(state: InterviewState) -> InterviewState:
    s = {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "allowed_actions": [ClientAction.ANSWER],
        "can_proceed": False,
    }
    resume = interrupt(
        {
            "type": "await_action",
            "allowed_actions": [ClientAction.ANSWER],
            "message": "Record and submit your answer.",
        }
    )
    _require_action(resume, ClientAction.ANSWER)

    audio_bytes = cast(bytes, resume.get("audio_bytes") or b"")
    filename = cast(str, resume.get("filename") or "")
    content_type = cast(str, resume.get("content_type") or "")

    return {
        **s,
        "audio_bytes": audio_bytes,
        "filename": filename,
        "content_type": content_type,
    }


async def node_transcribe(state: InterviewState, *, transcriber: WhisperTranscriber) -> InterviewState:
    audio_bytes = cast(bytes, state.get("audio_bytes") or b"")
    filename = cast(str, state.get("filename") or "")
    content_type = cast(str, state.get("content_type") or "")

    suffix = _best_effort_suffix(filename, content_type)
    text = await anyio.to_thread.run_sync(transcriber.transcribe_bytes, audio_bytes, suffix)
    a = (text or "").strip()

    if not a:
        # No transcript => stay awaiting answer; UI can re-submit ANSWER
        return {
            **state,
            "text": "",
            "assistant_text": "I didn't catch that. Please record again.",
            "allowed_actions": [ClientAction.ANSWER],
            "can_proceed": False,
            "phase": AgentPhase.AWAITING_ANSWER,
        }

    return {
        **state,
        "text": a,
    }


async def node_evaluate(state: InterviewState) -> InterviewState:
    q = (state.get("current_question") or "").strip()
    a = (state.get("text") or "").strip()

    if not q:
        return {
            **state,
            "assistant_text": "No question found. Press Start again.",
            "allowed_actions": [ClientAction.START],
            "can_proceed": False,
            "phase": AgentPhase.IDLE,
        }

    if not a:
        # transcription failed earlier and already set UI; keep it
        return state

    prompt = [
        SystemMessage(content=EVAL_SYS),
        HumanMessage(content=f"Question: {q}\nAnswer: {a}\nReturn JSON."),
    ]
    resp = await LLM.ainvoke(prompt)
    raw = (resp.content or "").strip()

    correct = False
    reason = "Could not parse grading."

    try:
        data = json.loads(raw)
        correct = bool(data.get("correct"))
        reason = (str(data.get("reason") or "").strip() or ("Correct." if correct else "Incorrect."))
    except Exception:
        # MVP fallback heuristic for common first question
        if q.lower().startswith("what does llm stand for") and "large language model" in a.lower():
            correct = True
            reason = "Correct."

    # scoring
    score = int(state.get("score") or 0)
    delta = 1 if correct else -1
    tool_out = update_score(delta=delta, reason=reason)
    score += int(tool_out["delta"])

    # MVP gating (as you requested): ALWAYS proceed
    can_proceed = True
    allowed_actions = [ClientAction.NEXT]

    feedback = f"{'✅ Correct' if correct else '❌ Not quite'} — {tool_out['reason']}\n\nScore: {score}"

    return {
        **state,
        "score": score,
        "phase": AgentPhase.EVALUATED,
        "assistant_text": feedback,
        "can_proceed": can_proceed,
        "allowed_actions": allowed_actions,
        "last_correct": correct,
        "last_reason": tool_out["reason"],
    }


def route_after_eval(state: InterviewState) -> Literal["wait_next", "end"]:
    # You asked: after evaluation check if it asked 10 questions then end else next
    # NOTE: turn increments when we ask a question, so after asking 10th question, turn==10.
    turn = int(state.get("turn") or 0)
    if turn >= 10:
        return "end"
    return "wait_next"


async def node_wait_next(state: InterviewState) -> InterviewState:
    s = {
        **state,
        "phase": AgentPhase.AWAITING_NEXT,
        "allowed_actions": [ClientAction.NEXT],
        "can_proceed": True,
    }
    resume = interrupt(
        {
            "type": "await_action",
            "allowed_actions": [ClientAction.NEXT],
            "message": "Click Next to continue.",
        }
    )
    _require_action(resume, ClientAction.NEXT)
    return {
        **s,
        "phase": AgentPhase.IDLE,  # will flow into ask_question next
    }


async def node_end(state: InterviewState) -> InterviewState:
    return {
        **state,
        "phase": AgentPhase.DONE,
        "assistant_text": f"Interview finished. Final score: {int(state.get('score') or 0)}",
        "allowed_actions": [],
        "can_proceed": False,
    }


# -----------------------------
# Build graph (LangGraph 1.0)
# -----------------------------
def build_interview_graph(*, transcriber: WhisperTranscriber):
    g = StateGraph(InterviewState)

    # Nodes
    g.add_node("wait_start", node_wait_start)
    g.add_node("ask_question", node_ask_question)
    g.add_node("wait_answer", node_wait_answer)

    async def transcribe_node(state: InterviewState) -> InterviewState:
        return await node_transcribe(state, transcriber=transcriber)

    g.add_node("transcribe", transcribe_node)
    g.add_node("evaluate", node_evaluate)
    g.add_node("wait_next", node_wait_next)
    g.add_node("end", node_end)

    # Flow
    g.add_edge(START, "wait_start")
    g.add_edge("wait_start", "ask_question")
    g.add_edge("ask_question", "wait_answer")
    g.add_edge("wait_answer", "transcribe")
    g.add_edge("transcribe", "evaluate")

    g.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {"wait_next": "wait_next", "end": "end"},
    )

    g.add_edge("wait_next", "ask_question")
    g.add_edge("end", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)


# -----------------------------
# Runtime helper for your API/UI
# -----------------------------
class InterviewEngine:
    """
    A small wrapper so your FastAPI/Streamlit can:
      - start a session (thread_id)
      - handle button clicks and voice uploads via Command(resume=...)
    """

    def __init__(self, *, transcriber: WhisperTranscriber):
        self._graph = build_interview_graph(transcriber=transcriber)

    async def init(self, *, thread_id: str) -> InterviewState:
        # Seed initial state into the checkpoint for this thread_id.
        # First call will immediately interrupt at wait_start.
        state = initial_state()
        out = await self._graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return cast(InterviewState, out)

    async def resume(self, *, thread_id: str, resume_payload: dict) -> InterviewState:
        """
        resume_payload examples:
          {"action": "START"}
          {"action": "NEXT"}
          {"action": "ANSWER", "audio_bytes": b"...", "filename": "x.webm", "content_type": "audio/webm"}
        """
        out = await self._graph.ainvoke(
            Command(resume=resume_payload),
            config={"configurable": {"thread_id": thread_id}},
        )
        return cast(InterviewState, out)
