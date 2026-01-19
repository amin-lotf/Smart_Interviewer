from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterator
import json
import requests


# ----------------------------
# DTOs
# ----------------------------
@dataclass(frozen=True)
class VoiceUploadResult:
    status: str
    text: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class StreamEvent:
    type: str
    data: dict


# ----------------------------
# API Client
# ----------------------------
class ApiClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # ---------- basic ----------
    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    # ---------- non-streaming ----------
    def upload_voice(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> VoiceUploadResult:
        files = {
            "audio": (filename, audio_bytes, content_type),
        }
        r = requests.post(
            f"{self.base_url}/v1/voice/upload",
            files=files,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        j = r.json()
        return VoiceUploadResult(
            status=j.get("status", "ok"),
            text=j.get("text", ""),
            filename=j.get("filename"),
            content_type=j.get("content_type"),
            size_bytes=j.get("size_bytes"),
        )

    # ---------- STREAMING (SSE) ----------
    def upload_voice_stream(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        read_timeout_s: float = 300.0,
    ) -> Iterator[StreamEvent]:
        """
        Calls /v1/voice/upload/stream and yields StreamEvent objects.
        """

        files = {
            "audio": (filename, audio_bytes, content_type),
        }

        # (connect timeout, read timeout)
        timeout = (10.0, read_timeout_s)

        with requests.post(
            f"{self.base_url}/v1/voice/upload/stream",
            files=files,
            stream=True,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        ) as r:
            r.raise_for_status()

            event_type: str | None = None
            data_buf: list[str] = []

            for raw_line in r.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.strip()

                # blank line = end of one SSE event
                if line == "":
                    if event_type is not None:
                        raw_data = "\n".join(data_buf).strip()
                        if raw_data:
                            try:
                                payload = json.loads(raw_data)
                            except json.JSONDecodeError:
                                payload = {"raw": raw_data}
                        else:
                            payload = {}

                        yield StreamEvent(type=event_type, data=payload)

                    event_type = None
                    data_buf = []
                    continue

                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_buf.append(line[len("data:") :].strip())

            # flush last event if stream ends without blank line
            if event_type is not None:
                raw_data = "\n".join(data_buf).strip()
                try:
                    payload = json.loads(raw_data) if raw_data else {}
                except json.JSONDecodeError:
                    payload = {"raw": raw_data}
                yield StreamEvent(type=event_type, data=payload)
