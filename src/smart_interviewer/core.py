# smart_interviewer/core.py
from __future__ import annotations

"""
Interview orchestration and grading logic using LangGraph.

This module implements an adaptive interview system that:
- Generates contextual questions based on difficulty level
- Evaluates answers using LLM with follow-up capability
- Automatically progresses through difficulty levels
- Manages interview state using LangGraph's state machine

Key Components:
    - InterviewEngine: Main interface for running interviews
    - InterviewState: TypedDict defining all state variables
    - Graph Nodes: Individual steps in the interview flow
    - Evaluation Functions: LLM-based answer assessment
    - Level Progression: Logic for advancing through difficulties

Example:
    ```python
    from smart_interviewer.core import InterviewEngine, WhisperTranscriber

    transcriber = WhisperTranscriber(model_name="small")
    engine = InterviewEngine(transcriber=transcriber)

    # Start interview
    state = await engine.init(thread_id="session-123")

    # Process user actions
    state = await engine.resume(
        thread_id="session-123",
        resume_payload={"action": "START"}
    )
    ```
"""

import base64
import json
import logging
import os
import random
import re
from datetime import timezone, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, List, Literal, Any, Annotated, Tuple, AsyncIterator

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

class TurnAttempt(TypedDict):
    question: str
    answer: str
    verdict: Literal["correct", "incorrect", "needs_more"]
    reason: str
    is_followup: bool

class TurnLog(TypedDict):
    level: int
    turn: int
    item_id: str
    context: str
    objective: str

    root_question: str
    attempts: List[TurnAttempt]

    correct: bool
    final_reason: str



def _bank_path() -> Path:
    # You can add settings.QUESTION_BANK_PATH in your settings later if you want.
    # For now, default to project root file name if not set.
    candidate = getattr(settings, "QUESTION_BANK_PATH", None)
    if candidate:
        return Path(candidate)
    return Path("data/question_bank.md")


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
    followups_used: int
    max_followups: int
    root_question: str  # first question for this item (kept)
    turn_attempts: List[TurnAttempt]  # attempts for current item (includes followups)




# -----------------------------
# LLM setup (LangChain 1.0)
# -----------------------------
LLM = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
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
    "- Current question\n"
    "- Reference context\n"
    "- Objective\n"
    "- Candidate answer\n\n"
    "Return ONE of these verdicts:\n"
    "- correct: answer sufficiently addresses the question.\n"
    "- incorrect: answer is wrong.\n"
    "- needs_more: answer is partially correct / incomplete / vague.\n\n"
    "If verdict is needs_more, generate a FOLLOW-UP QUESTION that asks ONLY for the missing part.\n"
    "Rules for follow-up question:\n"
    "- It must be narrower than the original question.\n"
    "- It must NOT repeat the whole original question.\n"
    "- It must NOT introduce a new topic.\n"
    
    "Return JSON ONLY with exactly these keys:\n"
    '{"verdict": "correct|incorrect|needs_more", "reason": "...", "next_question": "..."}\n'
    "If verdict != needs_more, set next_question to empty string.\n"
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
        "root_question": "",
        "turn_attempts": [],
        "followups_used": 0,
        "max_followups": settings.MAX_FOLLOWUP_QUESTIONS,
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
    q = q.strip('"""')  # remove wrapping quotes

    # Hard fallback
    if not q or len(q) < 5:
        q = "Explain the key idea in your own words and why it matters."

    # Ensure it ends like a question (optional but nice)
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"

    return q


async def _generate_question_for_item_stream(
    *,
    level: int,
    turn: int,
    item_id: str,
    context: str,
    objective: str,
    prev_questions: List[str],
) -> AsyncIterator[str]:
    """
    Streams question generation token by token.
    Yields cleaned tokens as they arrive.
    """
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

    accumulated = ""
    async for chunk in LLM.astream(prompt):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            accumulated += content
            yield content

    # Post-process accumulated text for safety
    q = accumulated.strip()
    q = re.sub(r"^[-*\d.\s]+", "", q).strip()
    q = q.strip('"""')

    if not q or len(q) < 5:
        yield "\nExplain the key idea in your own words and why it matters?"
    elif not q.endswith("?"):
        yield "?"

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
        "root_question": q,  # <-- keep original
        "turn_attempts": [],  # <-- new bundle starts here
        "followups_used": 0,  # <-- reset followups per item
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


async def _evaluate_answer_get_json(
    *,
    level: int,
    question: str,
    answer: str,
    context: str,
    objective: str,
) -> str:
    """Helper to get JSON evaluation response (non-streaming)."""
    prompt = [
        SystemMessage(content=EVAL_SYS),
        HumanMessage(
            content=(
                f"Level: {level}\n"
                f"Question: {question}\n\n"
                f"Reference context:\n{context}\n\n"
                f"Objective:\n{objective}\n\n"
                f"Candidate answer:\n{answer}\n\n"
                "Return JSON."
            )
        ),
    ]
    resp = await LLM.ainvoke(prompt)
    return (resp.content or "").strip()


async def _evaluate_answer_stream(
    *,
    level: int,
    question: str,
    answer: str,
    context: str,
    objective: str,
) -> AsyncIterator[str]:
    """
    Streams evaluation JSON response token by token.

    Args:
        level: Current difficulty level
        question: Question that was asked
        answer: Candidate's answer
        context: Reference context for the question
        objective: Learning objective being tested

    Yields:
        Tokens from the LLM evaluation response
    """
    prompt = [
        SystemMessage(content=EVAL_SYS),
        HumanMessage(
            content=(
                f"Level: {level}\n"
                f"Question: {question}\n\n"
                f"Reference context:\n{context}\n\n"
                f"Objective:\n{objective}\n\n"
                f"Candidate answer:\n{answer}\n\n"
                "Return JSON."
            )
        ),
    ]
    async for chunk in LLM.astream(prompt):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield content


def _parse_evaluation_result(raw: str, question: str, answer: str) -> Tuple[str, str, str]:
    """
    Parse the LLM's evaluation JSON response.

    Args:
        raw: Raw JSON string from LLM
        question: Original question (for fallback logic)
        answer: Candidate's answer (for fallback logic)

    Returns:
        Tuple of (verdict, reason, next_question)
    """
    verdict = "incorrect"
    reason = "Could not parse grading."
    next_q = ""

    try:
        data = json.loads(raw)
        verdict = str(data.get("verdict") or "").strip().lower()
        reason = (str(data.get("reason") or "").strip() or "No reason.")
        next_q = (str(data.get("next_question") or "")).strip()
        if verdict not in {"correct", "incorrect", "needs_more"}:
            verdict = "incorrect"
    except Exception:
        # Fallback for specific known questions
        if question.lower().startswith("what does llm stand for") and "large language model" in answer.lower():
            verdict = "correct"
            reason = "Correct."

    return verdict, reason, next_q


def _handle_followup_logic(
    verdict: str,
    next_q: str,
    state: InterviewState,
    turns_attempts: List[TurnAttempt],
) -> InterviewState | None:
    """
    Handle follow-up question logic when verdict is 'needs_more'.

    Args:
        verdict: Evaluation verdict
        next_q: Follow-up question text
        state: Current interview state
        turns_attempts: List of turn attempts

    Returns:
        New state dict if follow-up should be asked, None to continue normal flow
    """
    if verdict != "needs_more":
        return None

    used = int(state.get("followups_used") or 0)
    max_fu = int(state.get("max_followups") or settings.MAX_FOLLOWUP_QUESTIONS)

    if used < max_fu:
        if not next_q:
            next_q = "Can you add the missing detail?"

        return {
            **state,
            "phase": AgentPhase.AWAITING_ANSWER,
            "followups_used": used + 1,
            "current_question": next_q,
            "assistant_text": next_q,
            "allowed_actions": [ClientAction.ANSWER],
            "can_proceed": False,
            "turn_attempts": turns_attempts,
            "messages": [AIMessage(content=f"[follow-up] {next_q}")],
        }

    return None


def _calculate_level_progression(
    correct: bool,
    state: InterviewState,
) -> Tuple[bool, int, int, int, str]:
    """
    Calculate level progression after answering a question.

    Args:
        correct: Whether the answer was correct
        state: Current interview state

    Returns:
        Tuple of (interview_done, final_level, current_level, last_passed_level, extra_line)
    """
    batch_correct = int(state.get("batch_correct") or 0) + (1 if correct else 0)
    batch_index = int(state.get("batch_index") or 0) + 1
    batch_item_ids = list(state.get("batch_item_ids") or [])
    batch_n = len(batch_item_ids)

    interview_done = False
    final_level = int(state.get("final_level") or 0)
    last_passed_level = int(state.get("last_passed_level") or 0)
    current_level = int(state.get("current_level") or 1)
    extra_line = ""

    # Check if finished this level's batch
    if batch_index >= batch_n:
        need = _required_correct_for_batch(batch_n)

        if batch_correct >= need:
            # PASS this level
            last_passed_level = current_level
            next_level = current_level + 1

            if QUESTION_BANK.has_level(next_level):
                extra_line = f"\n\n✅ Passed Level {current_level} ({batch_correct}/{batch_n}). Next: Level {next_level}."
                current_level = next_level
            else:
                # No more levels => interview done
                interview_done = True
                final_level = last_passed_level
                extra_line = f"\n\n🏁 Passed Level {last_passed_level}. No more levels available. Final level: {final_level}."
        else:
            # FAIL this level => interview done
            interview_done = True
            final_level = last_passed_level
            extra_line = (
                f"\n\n❌ Failed Level {current_level} ({batch_correct}/{batch_n}). "
                f"Final level: {final_level}."
            )

    return interview_done, final_level, current_level, last_passed_level, extra_line


async def node_evaluate(state: InterviewState) -> InterviewState:
    """
    Evaluate the candidate's answer using LLM and determine next steps.

    This node handles:
    1. Getting LLM evaluation of the answer
    2. Determining if a follow-up question is needed
    3. Calculating level progression
    4. Building feedback message

    Args:
        state: Current interview state

    Returns:
        Updated state with evaluation results
    """
    # Extract current state
    q = (state.get("current_question") or "").strip()
    a = (state.get("text") or "").strip()
    level = int(state.get("current_level") or 1)
    turns_attempts = list(state.get("turn_attempts") or [])

    # Validation
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

    # Get evaluation from LLM
    ctx = (state.get("current_context") or "").strip()
    obj = (state.get("current_objective") or "").strip()

    raw = await _evaluate_answer_get_json(
        level=level,
        question=q,
        answer=a,
        context=ctx,
        objective=obj,
    )

    # Parse evaluation result
    verdict, reason, next_q = _parse_evaluation_result(raw, q, a)

    # Record this attempt
    turns_attempts.append(
        {
            "question": q,
            "answer": a,
            "verdict": verdict,  # type: ignore
            "reason": reason,
            "is_followup": bool(q != (state.get("root_question") or "")),
        }
    )

    # Handle follow-up questions if needed
    followup_state = _handle_followup_logic(verdict, next_q, state, turns_attempts)
    if followup_state is not None:
        return followup_state

    # If follow-up budget exhausted, mark as incorrect
    if verdict == "needs_more":
        verdict = "incorrect"
        turns_attempts[-1]["verdict"] = "incorrect"  # type: ignore
        turns_attempts[-1]["reason"] = "Follow-up limit reached. " + (turns_attempts[-1]["reason"] or "")

    # Determine correctness and calculate progression
    correct = (verdict == "correct")
    if not reason:
        reason = "Correct." if correct else "Incorrect."

    interview_done, final_level, current_level, last_passed_level, extra_line = _calculate_level_progression(correct, state)

    # Build feedback message
    feedback = f"{'✅ Correct' if correct else '❌ Not quite'} — {reason}{extra_line}"
    new_messages = [AIMessage(content=f"Feedback: {feedback}")]

    # Create log entry
    log_entry: TurnLog = {
        "level": int(state.get("batch_level") or level),
        "turn": int(state.get("turn") or 0),
        "item_id": str(state.get("current_item_id") or ""),
        "context": ctx,
        "objective": obj,
        "root_question": str(state.get("root_question") or q),
        "attempts": turns_attempts,
        "correct": bool(correct),
        "final_reason": reason,
    }

    turns_log = list(state.get("turns_log") or [])
    turns_log.append(log_entry)

    # Calculate updated batch counters
    batch_correct = int(state.get("batch_correct") or 0) + (1 if correct else 0)
    batch_index = int(state.get("batch_index") or 0) + 1
    batch_item_ids = list(state.get("batch_item_ids") or [])

    # Reset batch if advancing to next level
    if interview_done or (batch_index >= len(batch_item_ids) and current_level != int(state.get("current_level") or 1)):
        batch_item_ids = []
        batch_index = 0
        batch_correct = 0

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
        "turn_attempts": [],
        "root_question": "",
    }


def route_after_eval(state: InterviewState) -> Literal["wait_next","wait_again"]:
    if state.get('phase') == AgentPhase.AWAITING_ANSWER:
        return 'wait_again'
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
    """
    Build the interview state machine using LangGraph.

    Graph Flow:
        START → wait_start → ask_question → wait_answer → transcribe
                                  ↑              ↓
                                  |         evaluate
                                  |              ↓
                                  └──── wait_next ──→ finish_prepare → finish_wait → finish_finalize → END

    Key decision points:
    - After transcribe: Check if transcript is valid, retry if empty
    - After evaluate: Check if follow-up needed (verdict='needs_more')
    - After wait_next: Check if interview done or continue with next question

    Args:
        transcriber: WhisperTranscriber instance for audio processing

    Returns:
        Compiled LangGraph with in-memory checkpointing
    """
    g = StateGraph(InterviewState)

    # Main interview nodes
    g.add_node("wait_start", node_wait_start)          # Wait for START action
    g.add_node("ask_question", node_ask_question)      # Generate question using LLM
    g.add_node("wait_answer", node_wait_answer)        # Wait for ANSWER action with audio

    # Wrap transcribe node to inject transcriber dependency
    async def transcribe_node(state: InterviewState) -> InterviewState:
        return await node_transcribe(state, transcriber=transcriber)

    g.add_node("transcribe", transcribe_node)          # Convert audio to text
    g.add_node("evaluate", node_evaluate)              # Evaluate answer using LLM
    g.add_node("wait_next", node_wait_next)            # Wait for NEXT action

    # Finish flow (3-step process)
    g.add_node("finish_prepare", node_prepare_finish)  # Generate summary JSON
    g.add_node("finish_wait", node_wait_finish)        # Wait for FINISH action
    g.add_node("finish_finalize", node_finalize)       # Final cleanup

    # Define edges (flow)
    g.add_edge(START, "wait_start")
    g.add_edge("wait_start", "ask_question")
    g.add_edge("ask_question", "wait_answer")
    g.add_edge("wait_answer", "transcribe")

    # After transcribe: retry if empty, otherwise evaluate
    g.add_conditional_edges(
        'transcribe',
        route_after_transcribe,
        {"wait_again": "wait_answer", "evaluate": "evaluate"}
    )

    # After evaluate: may loop back to wait_answer for follow-up questions
    g.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {"wait_next": "wait_next", "wait_again": "wait_answer"},
    )

    # After wait_next: either ask next question or finish
    g.add_conditional_edges(
        "wait_next",
        route_after_next,
        {"ask": "ask_question", "finish": "finish_prepare"},
    )

    # Finish flow is linear
    g.add_edge("finish_prepare", "finish_wait")
    g.add_edge("finish_wait", "finish_finalize")
    g.add_edge("finish_finalize", END)

    # Compile with checkpointer to persist state between API calls
    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)


# -----------------------------
# Runtime helper for your API/UI
# -----------------------------
class InterviewEngine:
    """
    Main interface for running adaptive interviews using LangGraph.

    The engine manages the interview state machine, which includes:
    - Question generation based on difficulty level
    - Answer evaluation with follow-up questions
    - Automatic level progression
    - Interview completion and summary generation

    Example:
        ```python
        transcriber = WhisperTranscriber(model_name="small")
        engine = InterviewEngine(transcriber=transcriber)

        # Initialize new interview session
        state = await engine.init(thread_id="user-123")

        # Resume with user action
        state = await engine.resume(
            thread_id="user-123",
            resume_payload={"action": "START"}
        )
        ```
    """

    def __init__(self, *, transcriber: WhisperTranscriber):
        """
        Initialize the interview engine.

        Args:
            transcriber: WhisperTranscriber instance for speech-to-text
        """
        self._graph = build_interview_graph(transcriber=transcriber)

    async def init(self, *, thread_id: str) -> InterviewState:
        """
        Initialize a new interview session.

        Creates initial state and runs graph until first interrupt (AWAITING_START).

        Args:
            thread_id: Unique identifier for this interview session

        Returns:
            Initial interview state
        """
        state = initial_state()
        out = await self._graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return out

    async def resume(self, *, thread_id: str, resume_payload: dict) -> InterviewState:
        """
        Resume interview execution with user action.

        Args:
            thread_id: Interview session identifier
            resume_payload: Action payload, e.g., {"action": "START"} or
                          {"action": "ANSWER", "audio_bytes": b"...", ...}

        Returns:
            Updated interview state after processing the action
        """
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
        language='English',
    )
    engine = InterviewEngine(transcriber=transcriber)
    engine._graph.get_graph().draw_mermaid_png(output_file_path='flow.png')
