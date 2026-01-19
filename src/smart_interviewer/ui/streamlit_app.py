# smart_interviewer/ui/streamlit_app.py
from __future__ import annotations

import time

import streamlit as st
from st_audiorec import st_audiorec  # pip install streamlit-audiorec

from smart_interviewer.ui.api_client import ApiClient

st.set_page_config(page_title="Smart Interviewer", page_icon="🎤", layout="centered")

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

st.markdown('<div class="si-title">🎤 Smart Interviewer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="si-subtitle">Record → Send → See transcript immediately (streaming)</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# State
# ----------------------------
if "messages" not in st.session_state:
    # list[dict]: {role: "user"|"assistant", content: str}
    st.session_state.messages = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None  # bytes

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if "pending_audio_send" not in st.session_state:
    st.session_state.pending_audio_send = False

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
            st.session_state.pending_audio_send = False
            st.rerun()

# ----------------------------
# Chat window
# ----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----------------------------
# If we are in "streaming" mode, do it here (so it appears in the chat area)
# ----------------------------
if st.session_state.pending_audio_send and st.session_state.last_audio:
    api = ApiClient(st.session_state.api_base_url)

    # One assistant "live" message containing two live-updated blocks
    with st.chat_message("assistant"):
        transcript_box = st.empty()
        assistant_box = st.empty()

    transcript_box.markdown("**Transcript:** _(listening...)_")
    assistant_box.markdown("**Assistant:** _(thinking...)_")

    transcript_last = ""
    assistant_last = ""

    try:
        for ev in api.upload_voice_stream(
            audio_bytes=st.session_state.last_audio,
            filename=f"voice_{int(time.time())}.wav",
            content_type="audio/wav",
        ):
            if ev.type == "transcript":
                transcript_last = (ev.data.get("text") or "").strip()
                transcript_box.markdown(f"**Transcript:** {transcript_last or '...'}")

            elif ev.type == "assistant":
                assistant_last = (ev.data.get("text") or "").strip()
                assistant_box.markdown(f"**Assistant:** {assistant_last or '...'}")

            elif ev.type == "done":
                break

    except Exception as e:
        # Persist error
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Streaming failed: `{e}`"})
    else:
        # Persist final transcript + assistant response as separate chat messages
        # (so they remain after rerun)
        if transcript_last:
            st.session_state.messages.append({"role": "assistant", "content": f"**Transcript:** {transcript_last}"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "**Transcript:** _(empty)_"})
        if assistant_last:
            st.session_state.messages.append({"role": "assistant", "content": f"**Assistant:** {assistant_last}"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "**Assistant:** _(no response)_"})
    finally:
        # Reset state + rerun to show persisted messages
        st.session_state.last_audio = None
        st.session_state.pending_audio_send = False
        st.rerun()

# ----------------------------
# Fixed bottom "composer"
# ----------------------------
st.markdown('<div class="si-footer"><div class="si-footer-inner">', unsafe_allow_html=True)

# Recorder
audio_bytes = st_audiorec()  # returns wav bytes after recording
if audio_bytes:
    st.session_state.last_audio = audio_bytes

cols = st.columns([1, 1, 2])

with cols[0]:
    send_disabled = st.session_state.last_audio is None or st.session_state.pending_audio_send
    send_clicked = st.button("📨 Send", use_container_width=True, disabled=send_disabled)

with cols[1]:
    discard_disabled = st.session_state.last_audio is None or st.session_state.pending_audio_send
    if st.button("🗑️ Discard", use_container_width=True, disabled=discard_disabled):
        st.session_state.last_audio = None
        st.rerun()

with cols[2]:
    if st.session_state.last_audio:
        st.caption("Audio ready to send ✅")
        st.audio(st.session_state.last_audio, format="audio/wav")
    else:
        st.caption("Record something using the widget above 🎙️")

st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------------------
# Send action (sets pending flag and reruns to stream in chat area)
# ----------------------------
if send_clicked:
    # Add a user "voice" placeholder message
    st.session_state.messages.append({"role": "user", "content": "🎙️ (voice message)"})

    # Set pending streaming state
    st.session_state.pending_audio_send = True
    st.rerun()
