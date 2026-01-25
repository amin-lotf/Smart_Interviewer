import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List,  Optional


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


LEVEL_RE = re.compile(r"^\s*#\s*Level\s*(\d+)\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^\s*##\s*Item:\s*(.+?)\s*$", re.IGNORECASE)

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
        # Only keep valid items with a question
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