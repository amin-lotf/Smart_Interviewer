# smart_interviewer/ui/api_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterator
import requests
import json


@dataclass(frozen=True)
class SessionView:
    status: str
    phase: str
    turn: int

    current_level: int
    last_passed_level: int
    batch_level: int
    batch_index: int
    batch_size: int
    batch_correct: int
    interview_done: bool
    final_level: int

    current_question: str
    assistant_text: str
    transcript: str
    can_proceed: bool
    allowed_actions: List[str]

    # ✅ new
    interrupt: Optional[Dict[str, Any]] = None
    download: Optional[Dict[str, str]] = None
    summary: Optional[Dict[str, str]] = None


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

    def start_stream(self, *, session_id: str) -> Iterator[Dict[str, Any]]:
        """
        Streams question generation as it's generated.
        Yields dicts with either {"token": "..."} or {"final_state": SessionView}
        """
        r = requests.post(
            f"{self.base_url}/v1/interview/start/stream",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
            stream=True,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "token" in data:
                    yield {"token": data["token"]}
                elif "final_state" in data:
                    yield {"final_state": self._parse(data["final_state"])}

    def next(self, *, session_id: str) -> SessionView:
        r = requests.post(
            f"{self.base_url}/v1/interview/next",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
        )
        r.raise_for_status()
        return self._parse(r.json())

    def next_stream(self, *, session_id: str) -> Iterator[Dict[str, Any]]:
        """
        Streams next question generation as it's generated.
        Yields dicts with either {"token": "..."} or {"final_state": SessionView}
        """
        r = requests.post(
            f"{self.base_url}/v1/interview/next/stream",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
            stream=True,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "token" in data:
                    yield {"token": data["token"]}
                elif "final_state" in data:
                    yield {"final_state": self._parse(data["final_state"])}

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

    def answer_stream(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        session_id: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams full answer flow: transcript -> evaluation -> follow-up (if needed).
        Yields dicts with:
        - {"type": "transcript", "text": "..."}
        - {"type": "eval_token", "token": "..."}
        - {"type": "evaluation", "verdict": "...", "reason": "..."}
        - {"type": "followup_question", "question": "..."}
        - {"final_state": SessionView}
        """
        files = {"audio": (filename, audio_bytes, content_type)}
        r = requests.post(
            f"{self.base_url}/v1/interview/answer/stream",
            files=files,
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
            stream=True,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "final_state" in data:
                    yield {"final_state": self._parse(data["final_state"])}
                else:
                    yield data

    def evaluate_stream(self, *, session_id: str) -> Iterator[Dict[str, Any]]:
        """
        Streams evaluation feedback as it's generated.
        Yields dicts with {"token": "..."} or {"evaluation": {"verdict": "...", "reason": "..."}}
        """
        r = requests.post(
            f"{self.base_url}/v1/interview/evaluate/stream",
            timeout=self.timeout_s,
            headers={"X-Session-Id": session_id},
            stream=True,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "token" in data:
                    yield {"token": data["token"]}
                elif "evaluation" in data:
                    yield {"evaluation": data["evaluation"]}
                elif "error" in data:
                    yield {"error": data["error"]}

    def finish(self, *, session_id: str) -> SessionView:
        r = requests.post(
            f"{self.base_url}/v1/interview/finish",
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
            turn=int(j.get("turn", 0)),

            current_level=int(j.get("current_level", 1)),
            last_passed_level=int(j.get("last_passed_level", 0)),
            batch_level=int(j.get("batch_level", 0)),
            batch_index=int(j.get("batch_index", 0)),
            batch_size=int(j.get("batch_size", 3)),
            batch_correct=int(j.get("batch_correct", 0)),
            interview_done=bool(j.get("interview_done", False)),
            final_level=int(j.get("final_level", 0)),

            current_question=j.get("current_question", ""),
            assistant_text=j.get("assistant_text", ""),
            transcript=j.get("transcript", ""),
            can_proceed=bool(j.get("can_proceed", False)),
            allowed_actions=list(j.get("allowed_actions", [])),

            # ✅ new
            interrupt=j.get("interrupt"),
            download=j.get("download"),
            summary=j.get("summary"),
        )
