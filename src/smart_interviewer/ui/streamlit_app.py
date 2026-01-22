# smart_interviewer/ui/streamlit_app.py
from __future__ import annotations

import time
import uuid
import streamlit as st

from smart_interviewer.core import ClientAction
from smart_interviewer.ui.api_client import ApiClient, SessionView

st.set_page_config(page_title="Smart Interviewer", page_icon="🎤", layout="centered")

st.markdown("## 🎤 Smart Interviewer")
st.caption("Button-driven flow: Start → Answer (voice) → Evaluate → Next (if allowed)")

# ----------------------------
# Session state
# ----------------------------
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "server" not in st.session_state:
    st.session_state.server = None  # type: ignore

api = ApiClient(st.session_state.api_base_url)

def refresh_state() -> SessionView:
    s = api.get_state(session_id=st.session_state.session_id)
    st.session_state.server = s
    return s

def ensure_state() -> SessionView:
    s = st.session_state.server
    if s is None:
        return refresh_state()
    return s

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Settings")
    st.session_state.api_base_url = st.text_input("API Base URL", st.session_state.api_base_url)
    api = ApiClient(st.session_state.api_base_url)

    st.caption(f"Session: `{st.session_state.session_id}`")

    col0, col1 = st.columns(2)
    with col0:
        if st.button("Health", use_container_width=True):
            try:
                st.success(api.health())
            except Exception as e:
                st.error(str(e))
    with col1:
        if st.button("Reload state", use_container_width=True):
            try:
                refresh_state()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.button("Reset session (server)", use_container_width=True):
        try:
            api.reset_session(session_id=st.session_state.session_id)
            st.session_state.messages = []
            refresh_state()
            st.success("Reset done.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("New session id", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.server = None
        st.rerun()

# ----------------------------
# Main
# ----------------------------
s = ensure_state()

st.metric("Score", int(s.score))
st.caption(f"Phase: `{s.phase}` | Allowed: `{', '.join(s.allowed_actions) or '—'}`")

# Chat window
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# # Current question box
# if s.current_question:
#     st.info(f"**Question:** {s.current_question}")

# Controls row
colA, colB, colC = st.columns(3)

with colA:
    start_disabled = ClientAction.START not in s.allowed_actions
    if st.button("▶️ Start", use_container_width=True, disabled=start_disabled):
        try:
            s2 = api.start(session_id=st.session_state.session_id)
            st.session_state.server = s2
            st.session_state.messages.append({"role": "assistant", "content": f"**Question:** {s2.current_question}"})
            st.rerun()
        except Exception as e:
            st.error(str(e))

with colB:
    next_disabled = not s.can_proceed or (ClientAction.NEXT not in s.allowed_actions)
    if st.button("➡️ Next", use_container_width=True, disabled=next_disabled):
        try:
            s2 = api.next(session_id=st.session_state.session_id)
            st.session_state.server = s2
            st.session_state.messages.append({"role": "assistant", "content": f"**Question:** {s2.current_question}"})
            st.rerun()
        except Exception as e:
            st.error(str(e))

with colC:
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# Voice input (only meaningful in AWAITING_ANSWER)
audio = st.audio_input("Record your answer", sample_rate=16000)

send_disabled = (audio is None) or (ClientAction.ANSWER not in s.allowed_actions)
send_clicked = st.button("📨 Send Answer", use_container_width=True, disabled=send_disabled)

if audio is not None:
    st.audio(audio, format="audio/wav")

if send_clicked and audio is not None:
    st.session_state.messages.append({"role": "user", "content": "🎙️ (voice answer)"})

    try:
        audio_bytes = audio.getvalue()
        s2 = api.answer(
            audio_bytes=audio_bytes,
            filename=f"voice_{int(time.time())}.wav",
            content_type="audio/wav",
            session_id=st.session_state.session_id,
        )
        st.session_state.server = s2
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Answer failed: `{e}`"})
        st.rerun()

    # show transcript + feedback
    st.session_state.messages.append(
        {"role": "assistant", "content": f"**Transcript:** {s2.transcript or '_(empty)_'}"}
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": f"**Feedback:**\n\n{s2.assistant_text}"}
    )
    st.rerun()
