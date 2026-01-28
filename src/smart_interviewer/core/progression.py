from __future__ import annotations

from typing import Tuple
from smart_interviewer.settings import settings
from smart_interviewer.core.types import InterviewState


def required_correct_for_batch(cur_n_questions: int) -> int:
    if cur_n_questions >= settings.MIN_PASSED_FOR_LEVEL:
        if settings.MIN_PASSED_FOR_LEVEL <= settings.QUESTIONS_PER_LEVEL:
            return settings.MIN_PASSED_FOR_LEVEL
        return settings.QUESTIONS_PER_LEVEL
    return max(1, cur_n_questions)


def calculate_level_progression(*, correct: bool, state: InterviewState) -> Tuple[bool, int, int, int, str]:
    batch_correct = int(state.get("batch_correct") or 0) + (1 if correct else 0)
    batch_index = int(state.get("batch_index") or 0) + 1
    batch_item_ids = list(state.get("batch_item_ids") or [])
    batch_n = len(batch_item_ids)

    interview_done = False
    final_level = int(state.get("final_level") or 0)
    last_passed_level = int(state.get("last_passed_level") or 0)
    current_level = int(state.get("current_level") or 1)
    extra_line = ""

    if batch_index >= batch_n:
        need = required_correct_for_batch(batch_n)

        if batch_correct >= need:
            last_passed_level = current_level
            next_level = current_level + 1
            extra_line = f"\n\n✅ Passed Level {current_level} ({batch_correct}/{batch_n})."

            current_level = next_level  # the caller will check if level exists
        else:
            interview_done = True
            final_level = last_passed_level
            extra_line = f"\n\n❌ Failed Level {current_level} ({batch_correct}/{batch_n}). Final level: {final_level}."

    return interview_done, final_level, current_level, last_passed_level, extra_line
