# smart_interviewer/core.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, TypedDict, List

import anyio
from faster_whisper import WhisperModel
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from smart_interviewer.settings import settings
from langgraph.checkpoint.memory import InMemorySaver

class AgentPhase(StrEnum):
    IDLE = "IDLE"
    AWAITING_ANSWER = "AWAITING_ANSWER"
    EVALUATED = "EVALUATED"

class ClientAction(StrEnum):
    START = "START"
    NEXT = "NEXT"
    ANSWER = "ANSWER"
    RETRY = "RETRY"


# -----------------------------
# State
# -----------------------------
class InterviewState(TypedDict, total=False):
    # audio in (only for answer endpoint)
    audio_bytes: bytes
    filename: str
    content_type: str

    # transcript (latest)
    text: str

    # interview control
    phase: AgentPhase  # "IDLE" | "AWAITING_ANSWER" | "EVALUATED"
    current_question: str
    score: int
    turn: int

    # gating
    can_proceed: bool
    allowed_actions: List[ClientAction]

    # output for UI
    assistant_text: str


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
# LLM setup (no RAG yet)
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


# -----------------------------
# Core actions (button-driven)
# -----------------------------
async def start_interview(state: InterviewState) -> InterviewState:
    # reset-ish but keep score/turn if you want; MVP resets them
    score = 0
    turn = 0

    msg = [
        SystemMessage(content=ASK_SYS),
        HumanMessage(content=f"Turn number: {turn}. Current score: {score}."),
    ]
    resp = await LLM.ainvoke(msg)
    q = (resp.content or "").strip() or "What does LLM stand for?"

    return {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "current_question": q,
        "assistant_text": q,
        "turn": turn + 1,
        "score": score,
        "can_proceed": False,
        "allowed_actions": [ClientAction.ANSWER],
        "text": "",
    }


async def next_question(state: InterviewState) -> InterviewState:
    # only call if can_proceed is True and phase is EVALUATED
    turn = int(state.get("turn") or 0)
    score = int(state.get("score") or 0)

    msg = [
        SystemMessage(content=ASK_SYS),
        HumanMessage(content=f"Turn number: {turn}. Current score: {score}."),
    ]
    resp = await LLM.ainvoke(msg)
    q = (resp.content or "").strip() or "What does LLM stand for?"

    return {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "current_question": q,
        "assistant_text": q,
        "turn": turn + 1,
        "can_proceed": False,
        "allowed_actions": [ClientAction.ANSWER],
        "text": "",
    }


async def answer_and_evaluate(
    state: InterviewState,
    *,
    transcriber: WhisperTranscriber,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
) -> InterviewState:
    phase = (state.get("phase") or AgentPhase.IDLE)
    q = (state.get("current_question") or "").strip()

    if phase != AgentPhase.AWAITING_ANSWER or not q:
        return {
            **state,
            "assistant_text": "Press Start to get a question first.",
            "allowed_actions": [ClientAction.START],
            "can_proceed": False,
            "phase": AgentPhase.IDLE,
        }

    suffix = _best_effort_suffix(filename, content_type)
    text = await anyio.to_thread.run_sync(transcriber.transcribe_bytes, audio_bytes, suffix)
    a = (text or "").strip()

    if not a:
        return {
            **state,
            "text": "",
            "assistant_text": "I didn't catch that. Please record again.",
            "allowed_actions": [ClientAction.ANSWER],
            "can_proceed": False,
            "phase": AgentPhase.AWAITING_ANSWER,
        }

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

    # gating policy (MVP):
    # - correct => can proceed
    # - wrong => retry (no next)
    # can_proceed = bool(correct)
    # allowed_actions = ["next"] if can_proceed else ["answer"]  # "answer" acts as retry
    can_proceed = True
    allowed_actions = ["next"]

    feedback = f"{'✅ Correct' if correct else '❌ Not quite'} — {tool_out['reason']}\n\nScore: {score}"

    return {
        **state,
        "text": a,
        "score": score,
        "phase": "EVALUATED",
        "assistant_text": feedback,
        "can_proceed": can_proceed,
        "allowed_actions": allowed_actions,
    }
