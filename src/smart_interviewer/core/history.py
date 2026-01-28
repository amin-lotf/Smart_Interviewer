from __future__ import annotations

from typing import List

from smart_interviewer.core.types import TurnLog


def previous_questions_from_turns_log(turns_log: List[TurnLog], limit: int = 8) -> List[str]:
    """
    Structured history: collect root questions + follow-ups from past items.
    Returns newest-last (chronological).
    """
    out: List[str] = []
    for item in turns_log:
        rq = (item.get("root_question") or "").strip()
        if rq:
            out.append(rq)
        for att in (item.get("attempts") or []):
            q = (att.get("question") or "").strip()
            if q and q != rq:
                out.append(q)

    # de-dupe while keeping order
    seen = set()
    uniq: List[str] = []
    for q in out:
        if q not in seen:
            uniq.append(q)
            seen.add(q)

    return uniq[-limit:]
