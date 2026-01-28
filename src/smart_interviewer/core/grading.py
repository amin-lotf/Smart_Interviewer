from __future__ import annotations

import json
import re
from typing import Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import StreamWriter

from smart_interviewer.core.llm import LLM
from smart_interviewer.core.prompts import EVAL_SYS


def _extract_json_object(text: str) -> str:
    """
    Best-effort: extract the first {...} JSON object from the model output.
    """
    if not text:
        return ""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else text.strip()


def parse_evaluation(raw: str, *, question: str, answer: str) -> Tuple[str, str, str]:
    verdict = "incorrect"
    reason = "Could not parse grading."
    next_q = ""

    blob = _extract_json_object(raw)
    try:
        data = json.loads(blob)
        verdict = str(data.get("verdict") or "").strip().lower()
        reason = (str(data.get("reason") or "").strip() or "No reason.")
        next_q = (str(data.get("next_question") or "")).strip()

        if verdict not in {"correct", "incorrect", "needs_more"}:
            verdict = "incorrect"

        # clamp follow-up to a single line
        if next_q:
            next_q = next_q.splitlines()[0].strip()

    except Exception:
        # tiny fallback (keeps your old special-case idea, but less hacky)
        if question.lower().startswith("what does llm stand for") and "large language model" in answer.lower():
            verdict = "correct"
            reason = "Correct."
            next_q = ""

    return verdict, reason, next_q


async def evaluate_answer_json(
    *,
    level: int,
    question: str,
    answer: str,
    context: str,
    objective: str,
) -> str:
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


async def evaluate_answer_streaming(
    *,
    level: int,
    question: str,
    answer: str,
    context: str,
    objective: str,
    writer: StreamWriter | None = None,
) -> str:
    """
    Evaluate answer with optional streaming support.
    If writer is provided, streams evaluation tokens.
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

    raw = ""
    if writer:
        async for chunk in LLM.astream(prompt):
            token = chunk.content or ""
            if token:
                raw += token
                writer(("evaluation_token", token))
    else:
        resp = await LLM.ainvoke(prompt)
        raw = (resp.content or "").strip()

    return raw.strip()
