from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Dict, Any

from smart_interviewer.core.types import InterviewState


def build_summary_payload(state: InterviewState) -> Dict[str, Any]:
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


def make_summary_base64(state: InterviewState) -> str:
    summary = build_summary_payload(state)
    json_text = json.dumps(summary, ensure_ascii=False, indent=2)
    return base64.b64encode(json_text.encode("utf-8")).decode("ascii")
