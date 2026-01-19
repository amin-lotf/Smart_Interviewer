from __future__ import annotations

import operator
import os
import tempfile
from dataclasses import dataclass, field
from typing import TypedDict, Optional, Annotated

import anyio
from langgraph.graph import StateGraph, START, END
# Local Whisper runner (recommended for a portfolio demo)
from faster_whisper import WhisperModel


# -----------------------------
# State (LangGraph 1.0)
# -----------------------------
class VoiceState(TypedDict, total=False):
    audio_bytes: bytes
    filename: str
    content_type: str
    text: str
    assistant_text: str


# -----------------------------
# Transcriber
# -----------------------------
@dataclass
class WhisperTranscriber:
    """
    Local transcription using faster-whisper.
    - Uses a temp file because many Whisper pipelines assume a file path.
    """
    model_name: str = "small"
    device: str = "cpu"          # "cuda" if you have an NVIDIA GPU configured
    compute_type: str = "int8"   # e.g. "float16" on GPU
    language: Optional[str] = None  # e.g. "en", "zh", None = auto
    _model: WhisperModel = field(init=False, repr=False)
    def __post_init__(self) -> None:
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
        if not audio_bytes:
            raise ValueError("Empty audio bytes")

        # Write to a temp file (simple, robust)
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as f:
            f.write(audio_bytes)
            f.flush()

            segments, info = self._model.transcribe(
                f.name,
                language=self.language,
                vad_filter=True,
            )

            # Join segments
            parts = []
            for seg in segments:
                t = (seg.text or "").strip()
                if t:
                    parts.append(t)

            return " ".join(parts).strip()


# -----------------------------
# Graph nodes
# -----------------------------
async def node_transcribe(state: VoiceState, *, transcriber: WhisperTranscriber) -> VoiceState:
    audio_bytes = state.get("audio_bytes", b"")
    filename = state.get("filename", "")
    content_type = state.get("content_type", "")

    # best-effort suffix for temp file (helps ffmpeg sniff format)
    suffix = os.path.splitext(filename)[1].lower() if filename else ""
    if not suffix:
        # fallback guesses
        if content_type.endswith("webm"):
            suffix = ".webm"
        elif content_type.endswith("wav"):
            suffix = ".wav"
        elif content_type.endswith("mpeg") or content_type.endswith("mp3"):
            suffix = ".mp3"
        else:
            suffix = ".bin"

    # faster-whisper is sync -> run in a worker thread
    text = await anyio.to_thread.run_sync(transcriber.transcribe_bytes, audio_bytes, suffix)
    return {
        **state,
        "text": text or "",
    }



# -----------------------------
# Build compiled LangGraph app
# -----------------------------

async def node_fake_llm(state: VoiceState) -> VoiceState:
    # simulate thinking
    await anyio.sleep(1.2)

    t = (state.get("text") or "").strip()
    if not t:
        return {"assistant_text": "I didn't catch that. Can you repeat?"}

    return {"assistant_text": f"(fake llm) I heard you say: {t[:80]}..."}



def build_voice_graph(*, transcriber):
    graph = StateGraph(VoiceState)

    async def transcribe_node(state: VoiceState) -> VoiceState:
        return await node_transcribe(state, transcriber=transcriber)

    graph.add_node("transcribe", transcribe_node)
    graph.add_node("fake_llm", node_fake_llm)

    graph.add_edge(START, "transcribe")

    # simplest: sequential (works and you will still stream transcript early)
    graph.add_edge("transcribe", "fake_llm")
    graph.add_edge("fake_llm", END)

    return graph.compile()