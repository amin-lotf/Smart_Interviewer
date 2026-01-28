#smart_interviewer/types.py
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from enum import StrEnum
from typing import TypedDict, List, Literal, Any, Annotated, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class VoiceTranscriptionResponse(BaseModel):
    status: str = "ok"
    text: str = Field(..., description="Transcribed text of the user audio")
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None


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
    root_question: str
    turn: int  # number of questions asked overall

    # level-based progression
    current_level: int
    last_passed_level: int
    batch_size: int
    batch_index: int
    batch_correct: int
    batch_item_ids: List[str]
    batch_level: int

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
    turn_attempts: List[TurnAttempt]

    started_at: str
    finished_at: str

    summary_filename: str
    summary_content_type: str
    summary_data_base64: str

    followups_used: int
    max_followups: int


@dataclass(frozen=True, slots=True)
class InterviewItem:
    level: int
    item_id: str
    context: str
    objective: str


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
