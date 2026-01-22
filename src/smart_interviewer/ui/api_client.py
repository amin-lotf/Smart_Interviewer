# smart_interviewer/ui/api_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import requests


@dataclass(frozen=True)
class SessionView:
    status: str
    phase: str
    score: int
    turn: int
    current_question: str
    assistant_text: str
    transcript: str
    can_proceed: bool
    allowed_actions: List[str]


class ApiClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def reset_session(self, *, session_id: str) -> dict:
        r = requests.post(
            f"{self.base_url}/v1/session/reset",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
        )
        r.raise_for_status()
        return r.json()

    def get_state(self, *, session_id: str) -> SessionView:
        r = requests.get(
            f"{self.base_url}/v1/session/state",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
        )
        r.raise_for_status()
        return self._parse(r.json())

    def start(self, *, session_id: str) -> SessionView:
        r = requests.post(
            f"{self.base_url}/v1/interview/start",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
        )
        r.raise_for_status()
        return self._parse(r.json())

    def next(self, *, session_id: str) -> SessionView:
        r = requests.post(
            f"{self.base_url}/v1/interview/next",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
        )
        r.raise_for_status()
        return self._parse(r.json())

    def answer(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        session_id: str,
    ) -> SessionView:
        files = {"audio": (filename, audio_bytes, content_type)}
        r = requests.post(
            f"{self.base_url}/v1/interview/answer",
            files=files,
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
        )
        r.raise_for_status()
        return self._parse(r.json())

    @staticmethod
    def _parse(j: dict) -> SessionView:
        return SessionView(
            status=j.get("status", "ok"),
            phase=j.get("phase", ""),
            score=int(j.get("score", 0)),
            turn=int(j.get("turn", 0)),
            current_question=j.get("current_question", ""),
            assistant_text=j.get("assistant_text", ""),
            transcript=j.get("transcript", ""),
            can_proceed=bool(j.get("can_proceed", False)),
            allowed_actions=list(j.get("allowed_actions", [])),
        )
