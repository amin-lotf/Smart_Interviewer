# smart_interviewer/core.py
from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
from datetime import timezone, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, List, Literal, Any, Annotated,Tuple

import anyio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command  # LangGraph 1.0

from smart_interviewer.settings import settings
from smart_interviewer.transcriber import WhisperTranscriber
from smart_interviewer.utils import load_question_bank_from_md, InterviewItem

logger = logging.getLogger("smart_interviewer")
# -----------------------------
# Enums
# -----------------------------
class AgentPhase(StrEnum):
    IDLE = "IDLE"
    AWAITING_START = "AWAITING_START"
    AWAITING_ANSWER = "AWAITING_ANSWER"
    AWAITING_NEXT = "AWAITING_NEXT"
    ANSWERED = "ANSWERED"
    EVALUATED = "EVALUATED"
    AWAITING_FINISH = "AWAITING_FINISH"
    DONE = "DONE"


class ClientAction(StrEnum):
    START = "START"
    NEXT = "NEXT"
    ANSWER = "ANSWER"
    RETRY = "RETRY"
    FINISH = "FINISH"


class TurnLog(TypedDict):
    level: int
    turn: int
    item_id: str
    context: str
    objective: str
    question: str
    answer: str
    correct: bool
    evaluation: str



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
    turns_log: List[TurnLog]
    started_at: str  # optional
    finished_at: str  # optional
    summary_filename: str
    summary_content_type: str
    summary_data_base64: str





# -----------------------------
# LLM setup (LangChain 1.0)
# -----------------------------
LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=settings.OPENAI_API_KEY,
)


ASK_SYS = (
    "You are an interview question generator.\n"
    "You will be given:\n"
    "- Reference context\n"
    "- Objective (what we want to verify)\n"
    "- Previously asked questions (optional)\n\n"
    "Your job:\n"
    "- Ask ONE clear interview question that particularly tests the objective using the context.\n"
    "- Do NOT include the answer.\n"
    "- Do NOT quote the context verbatim unless absolutely necessary.\n"
    "- Keep it concise (one sentence preferred).\n"
    "- Avoid repeating previously asked questions.\n\n"
    "Return ONLY the question text. No JSON. No markdown."
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
    "- you can check the correctness only by looking at the context\n"
    "- If the answer is 'almost correct' for the question only according to the context.  mark it correct.\n"
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
    now = datetime.now(timezone.utc).isoformat()
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

        "batch_size": settings.QUESTIONS_PER_LEVEL,
        "batch_index": 0,
        "batch_correct": 0,
        "batch_item_ids": [],
        "batch_level": 0,

        "current_item_id": "",
        "current_context": "",
        "current_objective": "",

        "interview_done": False,
        "final_level": 0,
        "turns_log": [],
        "started_at": now,
        "finished_at": "",
        "summary_filename": "",
        "summary_content_type": "",
        "summary_data_base64": "",
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


def _required_correct_for_batch(cur_n_questions: int) -> int:
    if cur_n_questions >= settings.MIN_PASSED_FOR_LEVEL:
        if settings.MIN_PASSED_FOR_LEVEL <= settings.QUESTIONS_PER_LEVEL:
            return settings.MIN_PASSED_FOR_LEVEL
        else:
            return settings.QUESTIONS_PER_LEVEL
    return max(1, cur_n_questions)

def _extract_previous_questions(messages: List[BaseMessage], limit: int = 8) -> List[str]:
    """
    Pulls prior questions from AIMessage contents like: "[L2][item] Q5: <question>"
    Best-effort only.
    """
    out: List[str] = []
    if not messages:
        return out

    # scan from newest to oldest
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            txt = (m.content or "").strip()
            # match "Q<turn>: <question>"
            mm = re.search(r"\bQ\d+\s*:\s*(.+)$", txt)
            if mm:
                q = mm.group(1).strip()
                if q:
                    out.append(q)
        if len(out) >= limit:
            break

    # return in chronological-ish order
    return list(reversed(out))


async def _generate_question_for_item(
    *,
    level: int,
    turn: int,
    item_id: str,
    context: str,
    objective: str,
    prev_questions: List[str],
) -> str:
    prev_block = ""
    if prev_questions:
        prev_block = "\n".join(f"- {q}" for q in prev_questions[-8:])

    prompt = [
        SystemMessage(content=ASK_SYS),
        HumanMessage(
            content=(
                f"Level: {level}\n"
                f"Turn: {turn}\n"
                f"Item ID: {item_id}\n\n"
                f"Reference context:\n{context}\n\n"
                f"Objective:\n{objective}\n\n"
                + (f"Previously asked questions:\n{prev_block}\n\n" if prev_block else "")
                + "Generate the next question now."
            )
        ),
    ]

    resp = await LLM.ainvoke(prompt)
    q = (resp.content or "").strip()

    # Safety cleanup
    q = re.sub(r"^[-*\d.\s]+", "", q).strip()  # remove bullet/numbering
    q = q.strip('"“”')  # remove wrapping quotes

    # Hard fallback
    if not q or len(q) < 5:
        q = "Explain the key idea in your own words and why it matters."

    # Ensure it ends like a question (optional but nice)
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"

    return q

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
    prev_questions = _extract_previous_questions(list(state.get("messages") or []), limit=8)

    q = await _generate_question_for_item(
        level=level,
        turn=turn,
        item_id=item_id,
        context=(item.context or "").strip(),
        objective=(item.objective or "").strip(),
        prev_questions=prev_questions,
    )

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
        "assistant_text": "Answer recorded.",
        "messages": new_messages,
        "phase": AgentPhase.ANSWERED
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

    log_entry: TurnLog = {
        "level": int(state.get("batch_level") or level),  # batch_level is the tested level
        "turn": int(state.get("turn") or 0),
        "item_id": str(state.get("current_item_id") or ""),
        "context": ctx,
        "objective": obj,
        "question": q,
        "answer": a,
        "correct": bool(correct),
        "evaluation": reason,  # short evaluation
    }

    turns_log = list(state.get("turns_log") or [])
    turns_log.append(log_entry)
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
        "turns_log": turns_log,
    }


def route_after_eval(state: InterviewState) -> Literal["wait_next"]:
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


def _build_summary_payload(state: InterviewState) -> dict:
    final_level = int(state.get("final_level") or state.get("last_passed_level") or 0)
    return {
        "meta": {
            "app": "smart_interviewer",
            "started_at": state.get("started_at") or "",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "final_level": final_level,
            "turns": int(state.get("turn") or 0),
        },
        "items": list(state.get("turns_log") or []),
    }


async def node_prepare_finish(state: InterviewState) -> InterviewState:
    summary = _build_summary_payload(state)
    json_text = json.dumps(summary, ensure_ascii=False, indent=2)

    data_b64 = base64.b64encode(json_text.encode("utf-8")).decode("ascii")

    return {
        **state,
        "phase": AgentPhase.AWAITING_FINISH,
        "assistant_text": "Interview complete. Download the summary, then click Finish.",
        "allowed_actions": [ClientAction.FINISH],
        "can_proceed": True,

        "summary_filename": "interview_summary.json",
        "summary_content_type": "application/json",
        "summary_data_base64": data_b64,
    }


async def node_wait_finish(state: InterviewState) -> InterviewState:
    resume = interrupt(
        {
            "type": "await_action",
            "allowed_actions": [ClientAction.FINISH],
            "message": "Click Finish to close the interview.",
        }
    )
    _require_action(resume, ClientAction.FINISH)
    return {**state}  # move on


async def node_finalize(state: InterviewState) -> InterviewState:
    final_level = int(state.get("final_level") or state.get("last_passed_level") or 0)
    return {
        **state,
        "phase": AgentPhase.DONE,
        "assistant_text": f"Interview finished. Final level: {final_level}",
        "allowed_actions": [],
        "can_proceed": False,
    }


def route_after_next(state: InterviewState) -> Literal["ask", "finish"]:
    if bool(state.get("interview_done")):
        return "finish"
    return "ask"

def route_after_transcribe(state: InterviewState) -> Literal["evaluate","wait_again"]:
    if state.get('phase')==AgentPhase.AWAITING_ANSWER:
        return 'wait_again'
    return 'evaluate'

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

    # finish flow (3-step)
    g.add_node("finish_prepare", node_prepare_finish)
    g.add_node("finish_wait", node_wait_finish)
    g.add_node("finish_finalize", node_finalize)

    g.add_edge(START, "wait_start")
    g.add_edge("wait_start", "ask_question")
    g.add_edge("ask_question", "wait_answer")
    g.add_edge("wait_answer", "transcribe")
    # g.add_edge("transcribe", "evaluate")
    g.add_conditional_edges(
        'transcribe',
        route_after_transcribe,
        {"wait_again": "wait_answer","evaluate":"evaluate"}
    )
    # evaluate -> always go to wait_next (user clicks Next)
    g.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {"wait_next": "wait_next"},
    )

    # ✅ ONLY ONE conditional branch from wait_next
    g.add_conditional_edges(
        "wait_next",
        route_after_next,
        {"ask": "ask_question", "finish": "finish_prepare"},
    )

    g.add_edge("finish_prepare", "finish_wait")
    g.add_edge("finish_wait", "finish_finalize")
    g.add_edge("finish_finalize", END)

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


if __name__ == "__main__":
    transcriber = WhisperTranscriber(
        model_name='small',
        device='cpu',
        compute_type='int8',
        language=None,
    )
    engine = InterviewEngine(transcriber=transcriber)
    engine._graph.get_graph().draw_mermaid_png(output_file_path='flow.png')
