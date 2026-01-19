# smart_interviewer_ui/services/api_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import requests


@dataclass(frozen=True)
class VoiceUploadResult:
    status: str
    text: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


class ApiClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def upload_voice(self, *, audio_bytes: bytes, filename: str, content_type: str) -> VoiceUploadResult:
        files = {
            "audio": (filename, audio_bytes, content_type),
        }
        r = requests.post(f"{self.base_url}/v1/voice/upload", files=files, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        return VoiceUploadResult(
            status=j.get("status", "ok"),
            text=j.get("text", ""),
            filename=j.get("filename"),
            content_type=j.get("content_type"),
            size_bytes=j.get("size_bytes"),
        )
