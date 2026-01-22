# smart_interviewer/core.py
from __future__ import annotations

import json
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional, TypedDict, List, Literal, Any, cast, Annotated, Dict, Tuple

import anyio
from faster_whisper import WhisperModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

from langgraph.graph import StateGraph, START, END, add_messages
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
# Question bank model
# -----------------------------
@dataclass(frozen=True, slots=True)
class InterviewItem:
    level: int
    item_id: str
    context: str
    objective: str
    question: str


@dataclass(frozen=True, slots=True)
class QuestionBank:
    # level -> list[InterviewItem]
    items_by_level: Dict[int, List[InterviewItem]]

    @property
    def levels_sorted(self) -> List[int]:
        return sorted(self.items_by_level.keys())

    def has_level(self, level: int) -> bool:
        return level in self.items_by_level and bool(self.items_by_level[level])

    def max_level(self) -> int:
        return max(self.items_by_level.keys()) if self.items_by_level else 0


LEVEL_RE = re.compile(r"^\s*#\s*Level\s*(\d+)\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^\s*##\s*Item:\s*(.+?)\s*$", re.IGNORECASE)


def load_question_bank_from_md(path: str | Path) -> QuestionBank:
    """
    Parses a markdown file like:

    #Level 1 — ...
    ##Item: LLM-definition
    context:
    ...
    objective:
    ...
    question:
    ...

    Returns QuestionBank(level -> items).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Question file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    cur_level: Optional[int] = None
    cur_item_id: Optional[str] = None

    cur_context: List[str] = []
    cur_objective: List[str] = []
    cur_question: List[str] = []

    section: Optional[str] = None  # "context"|"objective"|"question"|None
    out: Dict[int, List[InterviewItem]] = {}

    def flush_item() -> None:
        nonlocal cur_level, cur_item_id, cur_context, cur_objective, cur_question, section
        if cur_level is None or cur_item_id is None:
            return
        ctx = "\n".join([x.rstrip() for x in cur_context]).strip()
        obj = "\n".join([x.rstrip() for x in cur_objective]).strip()
        q = "\n".join([x.rstrip() for x in cur_question]).strip()
        # Only keep valid items with a question
        if q:
            out.setdefault(cur_level, []).append(
                InterviewItem(
                    level=cur_level,
                    item_id=cur_item_id.strip(),
                    context=ctx,
                    objective=obj,
                    question=q,
                )
            )
        # reset item buffers
        cur_item_id = None
        cur_context = []
        cur_objective = []
        cur_question = []
        section = None

    for raw in lines:
        line = raw.rstrip("\n")

        m_level = LEVEL_RE.match(line)
        if m_level:
            # new level flush any pending item
            flush_item()
            cur_level = int(m_level.group(1))
            out.setdefault(cur_level, [])
            continue

        m_item = ITEM_RE.match(line)
        if m_item:
            # new item flush previous item
            flush_item()
            cur_item_id = m_item.group(1).strip()
            continue

        low = line.strip().lower()
        if low == "context:":
            section = "context"
            continue
        if low == "objective:":
            section = "objective"
            continue
        if low == "question:":
            section = "question"
            continue

        # accumulate
        if cur_level is None or cur_item_id is None:
            continue
        if section == "context":
            cur_context.append(line)
        elif section == "objective":
            cur_objective.append(line)
        elif section == "question":
            cur_question.append(line)
        else:
            # ignore unrelated lines inside item
            continue

    flush_item()
    return QuestionBank(items_by_level=out)


def _bank_path() -> Path:
    # You can add settings.QUESTION_BANK_PATH in your settings later if you want.
    # For now, default to project root file name if not set.
    candidate = getattr(settings, "QUESTION_BANK_PATH", None)
    if candidate:
        return Path(candidate)
    return Path("LLM_interview_questions.md")


# Load once at import time (simple and fast). For hot reload, you can move into InterviewEngine init.
QUESTION_BANK = load_question_bank_from_md(_bank_path())


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
    turn: int  # number of questions already asked overall

    # level-based progression
    current_level: int            # starts at 1 (testing Level 1)
    last_passed_level: int        # starts at 0
    batch_size: int               # usually 3
    batch_index: int              # 0..batch_size-1
    batch_correct: int            # number correct in current batch
    batch_item_ids: List[str]     # chosen items (ids) for the level, length=batch_size
    batch_level: int              # which level the batch belongs to

    # current item payload (for grading)
    current_item_id: str
    current_context: str
    current_objective: str

    # terminal
    interview_done: bool
    final_level: int

    # policy / ui helpers
    can_proceed: bool
    allowed_actions: List[ClientAction]

    # output for UI
    assistant_text: str

    # internal: last evaluation
    last_correct: bool
    last_reason: str

    # memory/messages
    messages: Annotated[List[BaseMessage], add_messages]


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
    temperature=0.2,
    api_key=settings.OPENAI_API_KEY,
)

EVAL_SYS = (
    "You are a strict-but-fair interview grader.\n"
    "You will be given:\n"
    "- Question\n"
    "- Reference context\n"
    "- Objective (what the question is trying to verify)\n"
    "- Candidate answer\n\n"
    "Decide if the candidate answer is correct.\n"
    "IMPORTANT:\n"
    "- Do NOT require exact wording.\n"
    "- If the answer is 'almost correct' and matches the meaning implied by the context/objective, mark it correct.\n"
    "- If it's partially correct but misses the key point, mark it incorrect.\n"
    "Return JSON ONLY with exactly these keys:\n"
    '{"correct": true/false, "reason": "..."}\n'
    "No extra keys. No markdown."
)


# -----------------------------
# Initial state
# -----------------------------
def initial_state() -> InterviewState:
    # user level assumed 0, so we TEST level 1 first
    return {
        "phase": AgentPhase.IDLE,
        "turn": 0,
        "current_question": "",
        "assistant_text": "Press Start to begin.",
        "can_proceed": False,
        "allowed_actions": [ClientAction.START],
        "text": "",
        "last_correct": False,
        "last_reason": "",
        "messages": [],

        "current_level": 1,
        "last_passed_level": 0,

        "batch_size": 3,
        "batch_index": 0,
        "batch_correct": 0,
        "batch_item_ids": [],
        "batch_level": 0,

        "current_item_id": "",
        "current_context": "",
        "current_objective": "",

        "interview_done": False,
        "final_level": 0,
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
    if not isinstance(resume, dict):
        raise ValueError("Resume payload must be a dict.")
    action = str(resume.get("action") or "")
    if action != str(expected):
        raise ValueError(f"Expected action {expected}, got {action!r}.")


def _pick_batch_for_level(*, level: int, batch_size: int) -> Tuple[List[str], int]:
    """
    Returns (item_ids, effective_batch_size).
    If level has fewer than requested items, uses all available.
    """
    items = QUESTION_BANK.items_by_level.get(level, [])
    if not items:
        return ([], 0)
    n = min(batch_size, len(items))
    chosen = random.sample(items, k=n)
    return ([it.item_id for it in chosen], n)


def _get_item(level: int, item_id: str) -> InterviewItem:
    items = QUESTION_BANK.items_by_level.get(level, [])
    for it in items:
        if it.item_id == item_id:
            return it
    raise KeyError(f"Item not found: level={level}, item_id={item_id!r}")


def _required_correct_for_batch(n: int) -> int:
    """
    For 3 questions => need 2.
    If fewer questions exist, be reasonable:
      n=2 => need 2? (too strict) => require 1 (n-1)
      n=1 => require 1
    """
    if n >= 3:
        return 2
    return max(1, n - 1)


# -----------------------------
# Graph nodes (LangGraph 1.0)
# -----------------------------
async def node_wait_start(state: InterviewState) -> InterviewState:
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
    return {**s, "phase": AgentPhase.IDLE}


async def node_ask_question(state: InterviewState) -> InterviewState:
    # If we're done, don't ask.
    if bool(state.get("interview_done")):
        return {
            **state,
            "phase": AgentPhase.DONE,
            "assistant_text": f"Interview finished. Final level: {int(state.get('final_level') or 0)}",
            "allowed_actions": [],
            "can_proceed": False,
        }

    level = int(state.get("current_level") or 1)
    batch_size = int(state.get("batch_size") or 3)

    # If current level doesn't exist => end at last passed
    if not QUESTION_BANK.has_level(level):
        final_level = int(state.get("last_passed_level") or 0)
        return {
            **state,
            "interview_done": True,
            "final_level": final_level,
            "phase": AgentPhase.DONE,
            "assistant_text": f"Reached the end of available levels. Your level: {final_level}",
            "allowed_actions": [],
            "can_proceed": False,
        }

    batch_level = int(state.get("batch_level") or 0)
    batch_item_ids = list(state.get("batch_item_ids") or [])
    batch_index = int(state.get("batch_index") or 0)

    # Need a fresh batch if:
    # - no batch yet
    # - batch belongs to different level
    # - batch finished (safety)
    if (not batch_item_ids) or (batch_level != level) or (batch_index >= len(batch_item_ids)):
        item_ids, effective_n = _pick_batch_for_level(level=level, batch_size=batch_size)
        if not item_ids:
            final_level = int(state.get("last_passed_level") or 0)
            return {
                **state,
                "interview_done": True,
                "final_level": final_level,
                "phase": AgentPhase.DONE,
                "assistant_text": f"No questions found for Level {level}. Your level: {final_level}",
                "allowed_actions": [],
                "can_proceed": False,
            }
        batch_item_ids = item_ids
        batch_index = 0
        # reset correct counter for new level attempt
        state = {
            **state,
            "batch_item_ids": batch_item_ids,
            "batch_index": 0,
            "batch_correct": 0,
            "batch_level": level,
            "batch_size": effective_n,  # track effective size actually used
        }

    # Ask the next question from the batch
    batch_item_ids = list(state.get("batch_item_ids") or [])
    batch_index = int(state.get("batch_index") or 0)
    item_id = batch_item_ids[batch_index]
    item = _get_item(level=level, item_id=item_id)

    turn = int(state.get("turn") or 0) + 1
    q = (item.question or "").strip() or "What does LLM stand for?"

    new_messages = [AIMessage(content=f"[L{level}][{item_id}] Q{turn}: {q}")]
    return {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "current_question": q,
        "assistant_text": f"(Level {level}, Q{batch_index+1}/{len(batch_item_ids)}) {q}",
        "turn": turn,
        "can_proceed": False,
        "allowed_actions": [ClientAction.ANSWER],
        "text": "",

        "current_item_id": item_id,
        "current_context": item.context,
        "current_objective": item.objective,

        "messages": new_messages,
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

    audio_bytes = resume.get("audio_bytes") or b""
    filename = resume.get("filename") or ""
    content_type = resume.get("content_type") or ""

    return {
        **s,
        "audio_bytes": audio_bytes,
        "filename": filename,
        "content_type": content_type,
    }


async def node_transcribe(state: InterviewState, *, transcriber: WhisperTranscriber) -> InterviewState:
    audio_bytes = state.get("audio_bytes") or b""
    filename = state.get("filename") or ""
    content_type = state.get("content_type") or ""

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

    new_messages = [HumanMessage(content=f"A{int(state.get('turn') or 0)}: {a}")]
    return {
        **state,
        "text": a,
        "messages": new_messages,
    }


async def node_evaluate(state: InterviewState) -> InterviewState:
    q = (state.get("current_question") or "").strip()
    a = (state.get("text") or "").strip()
    level = int(state.get("current_level") or 1)

    if not q:
        return {
            **state,
            "assistant_text": "No question found. Press Start again.",
            "allowed_actions": [ClientAction.START],
            "can_proceed": False,
            "phase": AgentPhase.IDLE,
        }
    if not a:
        return state

    ctx = (state.get("current_context") or "").strip()
    obj = (state.get("current_objective") or "").strip()

    prompt = [
        SystemMessage(content=EVAL_SYS),
        HumanMessage(
            content=(
                f"Level: {level}\n"
                f"Question: {q}\n\n"
                f"Reference context:\n{ctx}\n\n"
                f"Objective:\n{obj}\n\n"
                f"Candidate answer:\n{a}\n\n"
                "Return JSON."
            )
        ),
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
        # basic fallback for first classic question
        if q.lower().startswith("what does llm stand for") and "large language model" in a.lower():
            correct = True
            reason = "Correct."

    # Update batch counters
    batch_correct = int(state.get("batch_correct") or 0) + (1 if correct else 0)
    batch_index = int(state.get("batch_index") or 0) + 1
    batch_item_ids = list(state.get("batch_item_ids") or [])
    batch_n = len(batch_item_ids)

    # default: continue interview
    interview_done = False
    final_level = int(state.get("final_level") or 0)
    last_passed_level = int(state.get("last_passed_level") or 0)
    current_level = int(state.get("current_level") or 1)

    extra_line = ""
    # If finished this level's batch, decide pass/fail
    if batch_index >= batch_n:
        need = _required_correct_for_batch(batch_n)
        if batch_correct >= need:
            # PASS this level
            last_passed_level = current_level
            next_level = current_level + 1

            if QUESTION_BANK.has_level(next_level):
                extra_line = f"\n\n✅ Passed Level {current_level} ({batch_correct}/{batch_n}). Next: Level {next_level}."
                # advance level and reset batch state for next level
                current_level = next_level
                batch_item_ids = []
                batch_index = 0
                batch_correct = 0
            else:
                # no more levels => done
                interview_done = True
                final_level = last_passed_level
                extra_line = f"\n\n🏁 Passed Level {last_passed_level}. No more levels available. Final level: {final_level}."
        else:
            # FAIL this level => done, keep last_passed_level
            interview_done = True
            final_level = last_passed_level
            extra_line = (
                f"\n\n❌ Failed Level {current_level} ({batch_correct}/{batch_n}). "
                f"Final level: {final_level}."
            )

    feedback = f"{'✅ Correct' if correct else '❌ Not quite'} — {reason}{extra_line}"

    new_messages = [AIMessage(content=f"Feedback: {feedback}")]

    # UI: ALWAYS allow Next (even if done -> Next will route to end)
    return {
        **state,
        "phase": AgentPhase.EVALUATED,
        "assistant_text": feedback,
        "can_proceed": True,
        "allowed_actions": [ClientAction.NEXT],
        "last_correct": correct,
        "last_reason": reason,
        "messages": new_messages,

        "batch_correct": batch_correct,
        "batch_index": batch_index,
        "batch_item_ids": batch_item_ids,
        "current_level": current_level,
        "last_passed_level": last_passed_level,
        "interview_done": interview_done,
        "final_level": final_level,
    }


def route_after_eval(state: InterviewState) -> Literal["wait_next", "end"]:
    if bool(state.get("interview_done")):
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
    return {**s, "phase": AgentPhase.IDLE}


async def node_end(state: InterviewState) -> InterviewState:
    final_level = int(state.get("final_level") or state.get("last_passed_level") or 0)
    return {
        **state,
        "phase": AgentPhase.DONE,
        "assistant_text": f"Interview finished. Final level: {final_level}",
        "allowed_actions": [],
        "can_proceed": False,
    }


# -----------------------------
# Build graph (LangGraph 1.0)
# -----------------------------
def build_interview_graph(*, transcriber: WhisperTranscriber):
    g = StateGraph(InterviewState)

    g.add_node("wait_start", node_wait_start)
    g.add_node("ask_question", node_ask_question)
    g.add_node("wait_answer", node_wait_answer)

    async def transcribe_node(state: InterviewState) -> InterviewState:
        return await node_transcribe(state, transcriber=transcriber)

    g.add_node("transcribe", transcribe_node)
    g.add_node("evaluate", node_evaluate)
    g.add_node("wait_next", node_wait_next)
    g.add_node("end", node_end)

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
    def __init__(self, *, transcriber: WhisperTranscriber):
        self._graph = build_interview_graph(transcriber=transcriber)

    async def init(self, *, thread_id: str) -> InterviewState:
        state = initial_state()
        out = await self._graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return out

    async def resume(self, *, thread_id: str, resume_payload: dict) -> InterviewState:
        out = await self._graph.ainvoke(
            Command(resume=resume_payload),
            config={"configurable": {"thread_id": thread_id}},
        )
        return out
