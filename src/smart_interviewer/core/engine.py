from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Tuple, Any

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

    async def resume_stream(
        self, *, thread_id: str, resume_payload: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Resume execution with streaming enabled.
        Yields tuples of (event_type, data) where event_type is:
        - "question_token": streaming question generation
        - "evaluation_token": streaming evaluation feedback
        - "values": final state update
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"🔥 resume_stream called with stream_mode=['custom', 'values']")

        chunk_count = 0
        async for chunk in self._graph.astream(
            Command(resume=resume_payload),
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["custom", "values"],
        ):
            chunk_count += 1
            logger.warning(f"🔥 Chunk {chunk_count}: type={type(chunk)}, value={str(chunk)[:150]}")

            # Check 2-tuple format first (what we're actually getting)
            if isinstance(chunk, tuple) and len(chunk) == 2:
                mode, data = chunk

                if mode == "custom":
                    # custom event: ("custom", (event_type, token))
                    if isinstance(data, tuple) and len(data) == 2:
                        event_type, token = data
                        logger.warning(f"🔥 Yielding custom event: {event_type}, token={str(token)[:30]}")
                        yield (event_type, token)
                    else:
                        logger.warning(f"🔥 Unexpected custom data format: {data}")
                elif mode == "values":
                    # values event: ("values", state_dict)
                    logger.warning(f"🔥 Yielding values update")
                    yield ("values", data)
                else:
                    # Unknown mode
                    logger.warning(f"🔥 Unknown mode: {mode}")
                    yield (mode, data)

            # Check 3-tuple format (alternative format)
            elif isinstance(chunk, tuple) and len(chunk) == 3:
                namespace, mode, data = chunk
                if mode == "custom" and isinstance(data, tuple) and len(data) == 2:
                    event_type, token = data
                    logger.warning(f"🔥 Yielding custom event (3-tuple): {event_type}")
                    yield (event_type, token)
                elif mode == "values":
                    logger.warning(f"🔥 Yielding values (3-tuple)")
                    yield ("values", data)

            elif isinstance(chunk, dict):
                logger.warning(f"🔥 Yielding dict as values")
                yield ("values", chunk)
            else:
                logger.warning(f"🔥 Unknown chunk type: {type(chunk)}")
                yield ("update", chunk)



