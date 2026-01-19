# smart_interviewer_ui/app.py
from __future__ import annotations

import time
import streamlit as st
from st_audiorec import st_audiorec  # pip install streamlit-audiorec

from smart_interviewer.ui.api_client import ApiClient

st.set_page_config(page_title="Smart Interviewer", page_icon="🎙️", layout="centered")

# ----------------------------
# Styling (simple + clean)
# ----------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 1.5rem; padding-bottom: 6rem; }
      .si-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.25rem; }
      .si-subtitle { color: rgba(255,255,255,0.7); margin-bottom: 1.25rem; }
      .si-footer {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        background: rgba(20,20,20,0.85);
        backdrop-filter: blur(8px);
        border-top: 1px solid rgba(255,255,255,0.08);
        padding: 0.85rem 0;
        z-index: 999;
      }
      .si-footer-inner { max-width: 900px; margin: 0 auto; padding: 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="si-title">🎙️ Smart Interviewer</div>', unsafe_allow_html=True)
st.markdown('<div class="si-subtitle">Record → Send → See your transcript in the chat</div>', unsafe_allow_html=True)

# ----------------------------
# State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict]: {role: "user"|"assistant", content: str}

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None  # bytes

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

# ----------------------------
# Sidebar settings
# ----------------------------
with st.sidebar:
    st.header("Settings")
    st.session_state.api_base_url = st.text_input("API Base URL", st.session_state.api_base_url)
    st.caption("Example: http://localhost:8000")

    api = ApiClient(st.session_state.api_base_url)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Health check", use_container_width=True):
            try:
                st.success(api.health())
            except Exception as e:
                st.error(str(e))
    with col_b:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_audio = None
            st.rerun()

# ----------------------------
# Chat window
# ----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----------------------------
# Fixed bottom “composer”
# ----------------------------
st.markdown('<div class="si-footer"><div class="si-footer-inner">', unsafe_allow_html=True)

# Recorder
audio_bytes = st_audiorec()  # returns wav bytes after recording
if audio_bytes:
    st.session_state.last_audio = audio_bytes

cols = st.columns([1, 1, 2])
with cols[0]:
    send_disabled = st.session_state.last_audio is None
    send_clicked = st.button("📨 Send", use_container_width=True, disabled=send_disabled)

with cols[1]:
    if st.button("🗑️ Discard", use_container_width=True, disabled=st.session_state.last_audio is None):
        st.session_state.last_audio = None
        st.rerun()

with cols[2]:
    if st.session_state.last_audio:
        st.caption("Audio ready to send ✅")
        st.audio(st.session_state.last_audio, format="audio/wav")
    else:
        st.caption("Record something using the widget above 👆")

st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------------------
# Send action
# ----------------------------
if send_clicked:
    api = ApiClient(st.session_state.api_base_url)

    # Add a “user said (voice)” placeholder message
    st.session_state.messages.append({"role": "user", "content": "🎤 (voice message)"})

    with st.spinner("Uploading audio…"):
        try:
            result = api.upload_voice(
                audio_bytes=st.session_state.last_audio,
                filename=f"voice_{int(time.time())}.wav",
                content_type="audio/wav",
            )
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Upload failed: `{e}`"})
            st.session_state.last_audio = None
            st.rerun()

    # Show transcript in chat
    st.session_state.messages.append({"role": "assistant", "content": f"**Transcript:** {result.text}"})

    # Reset buffer
    st.session_state.last_audio = None
    st.rerun()
