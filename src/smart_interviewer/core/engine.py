from __future__ import annotations

from datetime import datetime, timezone

from langgraph.types import Command

from smart_interviewer.settings import settings
from smart_interviewer.core.transcriber import WhisperTranscriber

from smart_interviewer.core.types import InterviewState, AgentPhase, ClientAction
from smart_interviewer.core.graph import build_interview_graph
from smart_interviewer.core.question_bank import seed_rng_if_configured


def initial_state() -> InterviewState:
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


class InterviewEngine:
    def __init__(self, *, transcriber: WhisperTranscriber):
        seed_rng_if_configured()
        self._graph = build_interview_graph(transcriber=transcriber)

    async def init(self, *, thread_id: str) -> InterviewState:
        state = initial_state()
        out = await self._graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
        return out

    async def resume(self, *, thread_id: str, resume_payload: dict) -> InterviewState:
        out = await self._graph.ainvoke(
            Command(resume=resume_payload),
            config={"configurable": {"thread_id": thread_id}},
        )
        return out



