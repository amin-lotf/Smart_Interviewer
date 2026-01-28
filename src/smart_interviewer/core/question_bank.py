from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict

from smart_interviewer.settings import settings
from smart_interviewer.core.types import InterviewItem, QuestionBank

LEVEL_RE = re.compile(r"^\s*#\s*Level\s*(\d+)\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^\s*##\s*Item:\s*(.+?)\s*$", re.IGNORECASE)

def bank_path() -> Path:
    candidate = getattr(settings, "QUESTION_BANK_PATH", None)
    if candidate:
        return Path(candidate)
    return Path("data/question_bank.md")


def load_question_bank_from_md(path: str | Path) -> QuestionBank:
    """
    Parses a markdown file like:

    #Level 1 — ...
    ##Item: LLM-definition
    context:
    ...
    objective:
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

    section: Optional[str] = None  # "context"|"objective"|"question"|None
    out: Dict[int, List[InterviewItem]] = {}

    def flush_item() -> None:
        nonlocal cur_level, cur_item_id, cur_context, cur_objective, section
        if cur_level is None or cur_item_id is None:
            return
        ctx = "\n".join([x.rstrip() for x in cur_context]).strip()
        obj = "\n".join([x.rstrip() for x in cur_objective]).strip()
        # Only keep valid items with context.
        if ctx:
            out.setdefault(cur_level, []).append(
                InterviewItem(
                    level=cur_level,
                    item_id=cur_item_id.strip(),
                    context=ctx,
                    objective=obj,
                )
            )
        # reset item buffers
        cur_item_id = None
        cur_context = []
        cur_objective = []
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

        # accumulate
        if cur_level is None or cur_item_id is None:
            continue
        if section == "context":
            cur_context.append(line)
        elif section == "objective":
            cur_objective.append(line)
        else:
            # ignore unrelated lines inside item
            continue

    flush_item()
    return QuestionBank(items_by_level=out)


QUESTION_BANK = load_question_bank_from_md(bank_path())


def seed_rng_if_configured() -> None:
    seed = getattr(settings, "RANDOM_SEED", None)
    if seed is None:
        return
    try:
        random.seed(int(seed))
    except Exception:
        # ignore bad seed values
        pass


def pick_batch_for_level(*, level: int, batch_size: int) -> Tuple[List[str], int]:
    items = QUESTION_BANK.items_by_level.get(level, [])
    if not items:
        return ([], 0)
    n = min(batch_size, len(items))
    chosen = random.sample(items, k=n)
    return ([it.item_id for it in chosen], n)


def get_item(*, level: int, item_id: str) -> InterviewItem:
    items = QUESTION_BANK.items_by_level.get(level, [])
    for it in items:
        if it.item_id == item_id:
            return it
    raise KeyError(f"Item not found: level={level}, item_id={item_id!r}")


def has_level(level: int) -> bool:
    return QUESTION_BANK.has_level(level)



