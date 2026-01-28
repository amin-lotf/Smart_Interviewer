from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Literal, List

import anyio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt
from langgraph.config import get_stream_writer

from smart_interviewer.settings import settings
from smart_interviewer.core.transcriber import WhisperTranscriber

from smart_interviewer.core.types import InterviewState, AgentPhase, ClientAction, TurnLog, TurnAttempt
from smart_interviewer.core.prompts import ASK_SYS
from smart_interviewer.core.llm import LLM
from smart_interviewer.core.question_bank import pick_batch_for_level, get_item, has_level
from smart_interviewer.core.history import previous_questions_from_turns_log
from smart_interviewer.core.grading import evaluate_answer_streaming, parse_evaluation
from smart_interviewer.core.progression import calculate_level_progression
from smart_interviewer.core.summary import make_summary_base64


def best_effort_suffix(filename: str, content_type: str) -> str:
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


def require_action(resume: Any, expected: ClientAction) -> None:
    if not isinstance(resume, dict):
        raise ValueError("Resume payload must be a dict.")
    action = str(resume.get("action") or "")
    if action != str(expected):
        raise ValueError(f"Expected action {expected}, got {action!r}.")


async def generate_question_for_item(
    *,
    level: int,
    turn: int,
    item_id: str,
    context: str,
    objective: str,
    prev_questions: List[str],
    writer: StreamWriter | None = None,
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

    # Stream the response if writer is provided
    import logging
    logger = logging.getLogger(__name__)

    q = ""
    if writer:
        logger.warning(f"✨ STREAMING question for turn {turn}, level {level}")
        async for chunk in LLM.astream(prompt):
            token = chunk.content or ""
            if token:
                q += token
                writer(("question_token", token))
        logger.warning(f"✨ Finished streaming {len(q)} chars")
    else:
        logger.warning(f"⚡ NON-STREAMING for turn {turn}, level {level} (likely session init)")
        resp = await LLM.ainvoke(prompt)
        q = (resp.content or "").strip()

    # hardening
    q = q.strip()
    q = q.splitlines()[0].strip()
    q = re.sub(r"^[-*\d.\s]+", "", q).strip()
    q = q.strip('"""').strip()
    q = q[:240].strip()

    if not q or len(q) < 5:
        q = "Explain the key idea in your own words and why it matters."

    if not q.endswith("?"):
        q = q.rstrip(".") + "?"

    return q


async def node_wait_start(state: InterviewState) -> InterviewState:
    s = {
        **state,
        "phase": AgentPhase.AWAITING_START,
        "assistant_text": state.get("assistant_text") or "Press Start to begin.",
        "allowed_actions": [ClientAction.START],
        "can_proceed": False,
    }
    resume = interrupt(
        {"type": "await_action", "allowed_actions": [ClientAction.START], "message": s["assistant_text"]}
    )
    require_action(resume, ClientAction.START)
    return {**s, "phase": AgentPhase.IDLE}


async def node_ask_question(state: InterviewState) -> InterviewState:
    if bool(state.get("interview_done")):
        final_level = int(state.get("final_level") or state.get("last_passed_level") or 0)
        return {
            **state,
            "phase": AgentPhase.DONE,
            "assistant_text": f"Interview finished. Final level: {final_level}",
            "allowed_actions": [],
            "can_proceed": False,
        }

    level = int(state.get("current_level") or 1)
    batch_size = int(state.get("batch_size") or settings.QUESTIONS_PER_LEVEL)

    if not has_level(level):
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

    if (not batch_item_ids) or (batch_level != level) or (batch_index >= len(batch_item_ids)):
        item_ids, effective_n = pick_batch_for_level(level=level, batch_size=batch_size)
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
        state = {
            **state,
            "batch_item_ids": item_ids,
            "batch_index": 0,
            "batch_correct": 0,
            "batch_level": level,
            "batch_size": effective_n,
        }
        batch_item_ids = item_ids
        batch_index = 0

    item_id = batch_item_ids[batch_index]
    item = get_item(level=level, item_id=item_id)

    turn = int(state.get("turn") or 0) + 1
    prev_questions = previous_questions_from_turns_log(list(state.get("turns_log") or []), limit=8)

    # Get stream writer from context
    writer = get_stream_writer()

    q = await generate_question_for_item(
        level=level,
        turn=turn,
        item_id=item_id,
        context=(item.context or "").strip(),
        objective=(item.objective or "").strip(),
        prev_questions=prev_questions,
        writer=writer,
    )

    new_messages = [AIMessage(content=f"[L{level}][{item_id}] Q{turn}: {q}")]

    return {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "current_question": q,
        "root_question": q,
        "turn_attempts": [],
        "followups_used": 0,
        "assistant_text": f"(Level {level}, Q{batch_index+1}/{len(batch_item_ids)}) {q}",
        "turn": turn,
        "can_proceed": False,
        "allowed_actions": [ClientAction.ANSWER],
        "text": "",
        "current_item_id": item_id,
        "current_context": item.context,
        "current_objective": item.objective,
        "messages": list(state.get("messages") or []) + new_messages,
    }


async def node_wait_answer(state: InterviewState) -> InterviewState:
    s = {
        **state,
        "phase": AgentPhase.AWAITING_ANSWER,
        "allowed_actions": [ClientAction.ANSWER],
        "can_proceed": False,
    }
    resume = interrupt(
        {"type": "await_action", "allowed_actions": [ClientAction.ANSWER], "message": "Record and submit your answer."}
    )
    require_action(resume, ClientAction.ANSWER)

    return {
        **s,
        "audio_bytes": resume.get("audio_bytes") or b"",
        "filename": resume.get("filename") or "",
        "content_type": resume.get("content_type") or "",
    }


async def node_transcribe(state: InterviewState, *, transcriber: WhisperTranscriber) -> InterviewState:
    audio_bytes = state.get("audio_bytes") or b""
    filename = state.get("filename") or ""
    content_type = state.get("content_type") or ""

    suffix = best_effort_suffix(filename, content_type)
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
        "messages": list(state.get("messages") or []) + new_messages,
        "phase": AgentPhase.ANSWERED,
    }


def handle_followup_logic(
    *,
    verdict: str,
    next_q: str,
    reason:str,
    state: InterviewState,
    turns_attempts: List[TurnAttempt],
) -> InterviewState | None:
    if verdict != "needs_more":
        return None

    used = int(state.get("followups_used") or 0)
    max_fu = int(state.get("max_followups") or settings.MAX_FOLLOWUP_QUESTIONS)

    if used < max_fu:
        fq = (next_q or "Can you add the missing detail?").strip()
        fq = fq.splitlines()[0].strip()
        if fq and not fq.endswith("?"):
            fq = fq.rstrip(".") + "?"

        return {
            **state,
            "phase": AgentPhase.AWAITING_ANSWER,
            "followups_used": used + 1,
            "current_question": fq,
            "assistant_text": reason,
            "allowed_actions": [ClientAction.ANSWER],
            "can_proceed": False,
            "turn_attempts": turns_attempts,
            "messages": list(state.get("messages") or []) + [AIMessage(content=f"[follow-up] {fq}")],
        }

    return None


async def node_evaluate(state: InterviewState) -> InterviewState:
    q = (state.get("current_question") or "").strip()
    a = (state.get("text") or "").strip()
    level = int(state.get("current_level") or 1)
    turns_attempts = list(state.get("turn_attempts") or [])

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

    # Get stream writer from context
    writer = get_stream_writer()

    raw = await evaluate_answer_streaming(level=level, question=q, answer=a, context=ctx, objective=obj, writer=writer)
    verdict, reason, next_q = parse_evaluation(raw, question=q, answer=a)

    turns_attempts.append(
        {
            "question": q,
            "answer": a,
            "verdict": verdict,  # type: ignore
            "reason": reason,
            "is_followup": bool(q != (state.get("root_question") or "")),
        }
    )
    followup_state = handle_followup_logic(verdict=verdict, next_q=next_q, reason=reason,state=state, turns_attempts=turns_attempts)
    if followup_state is not None:
        return followup_state

    # Follow-up budget exhausted => treat as incorrect
    if verdict == "needs_more":
        verdict = "incorrect"
        turns_attempts[-1]["verdict"] = "incorrect"  # type: ignore
        turns_attempts[-1]["reason"] = "Follow-up limit reached. " + (turns_attempts[-1]["reason"] or "")
        reason = turns_attempts[-1]["reason"]

    correct = (verdict == "correct")
    if not reason:
        reason = "Correct." if correct else "Incorrect."

    interview_done, final_level, next_level, last_passed_level, extra_line = calculate_level_progression(
        correct=correct, state=state
    )

    # if they passed and next level doesn't exist -> end
    if not interview_done and extra_line.startswith("\n\n✅") and not has_level(next_level):
        interview_done = True
        final_level = next_level - 1
        extra_line = f"\n\n✅ Passed Level {final_level}. No more levels available. Final level: {final_level}."

    feedback = f"{'✅ Correct' if correct else '❌ Not quite'} — {reason}{extra_line}"

    log_entry: TurnLog = {
        "level": int(state.get("batch_level") or level),
        "turn": int(state.get("turn") or 0),
        "item_id": str(state.get("current_item_id") or ""),
        "context": ctx,
        "objective": obj,
        "root_question": str(state.get("root_question") or q),
        "attempts": turns_attempts,
        "correct": bool(correct),
        "final_reason": ("Failed: follow-up limit reached" if "Follow-up limit reached" in reason else reason),
    }

    turns_log = list(state.get("turns_log") or [])
    turns_log.append(log_entry)

    # update batch counters
    batch_correct = int(state.get("batch_correct") or 0) + (1 if correct else 0)
    batch_index = int(state.get("batch_index") or 0) + 1
    batch_item_ids = list(state.get("batch_item_ids") or [])

    # if advancing to next level (or done), reset batch
    current_level = int(state.get("current_level") or 1)
    if interview_done or (batch_index >= len(batch_item_ids) and next_level != current_level):
        batch_item_ids = []
        batch_index = 0
        batch_correct = 0
        current_level = next_level

    return {
        **state,
        "phase": AgentPhase.EVALUATED,
        "assistant_text": feedback,
        "can_proceed": True,
        "allowed_actions": [ClientAction.NEXT],
        "last_correct": correct,
        "last_reason": reason,
        "messages": list(state.get("messages") or []) + [AIMessage(content=f"Feedback: {feedback}")],
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


def route_after_eval(state: InterviewState) -> Literal["wait_next", "wait_again"]:
    if state.get("phase") == AgentPhase.AWAITING_ANSWER:
        return "wait_again"
    return "wait_next"


def route_after_transcribe(state: InterviewState) -> Literal["evaluate", "wait_again"]:
    if state.get("phase") == AgentPhase.AWAITING_ANSWER:
        return "wait_again"
    return "evaluate"


async def node_wait_next(state: InterviewState) -> InterviewState:
    s = {
        **state,
        "phase": AgentPhase.AWAITING_NEXT,
        "allowed_actions": [ClientAction.NEXT],
        "can_proceed": True,
    }
    resume = interrupt(
        {"type": "await_action", "allowed_actions": [ClientAction.NEXT], "message": "Click Next to continue."}
    )
    require_action(resume, ClientAction.NEXT)
    return {**s, "phase": AgentPhase.IDLE}


def route_after_next(state: InterviewState) -> Literal["ask", "finish"]:
    return "finish" if bool(state.get("interview_done")) else "ask"


async def node_prepare_finish(state: InterviewState) -> InterviewState:
    data_b64 = make_summary_base64(state)
    return {
        **state,
        "phase": AgentPhase.AWAITING_FINISH,
        "assistant_text": "Interview complete. Download the summary, then click Finish.",
        "allowed_actions": [ClientAction.FINISH],
        "can_proceed": True,
        "summary_filename": "interview_summary.json",
        "summary_content_type": "application/json",
        "summary_data_base64": data_b64,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


async def node_wait_finish(state: InterviewState) -> InterviewState:
    resume = interrupt(
        {"type": "await_action", "allowed_actions": [ClientAction.FINISH], "message": "Click Finish to close the interview."}
    )
    require_action(resume, ClientAction.FINISH)
    return {**state}


async def node_finalize(state: InterviewState) -> InterviewState:
    final_level = int(state.get("final_level") or state.get("last_passed_level") or 0)
    return {
        **state,
        "phase": AgentPhase.DONE,
        "assistant_text": f"Interview finished. Final level: {final_level}",
        "allowed_actions": [],
        "can_proceed": False,
    }
