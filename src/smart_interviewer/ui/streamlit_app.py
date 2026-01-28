# smart_interviewer/ui/streamlit_app.py
from __future__ import annotations

import base64
import time
import uuid
import streamlit as st

from smart_interviewer.core import ClientAction
from smart_interviewer.ui.api_client import ApiClient, SessionView

st.set_page_config(page_title="Smart Interviewer", page_icon="🎤", layout="centered")

st.markdown("## 🎤 Smart Interviewer")
st.caption("Flow: Start → Answer (voice) → Evaluate → Next (until fail or end of levels)")

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
if getattr(s, "summary", None):
    st.divider()
    st.subheader("📄 Summary")

    data_b64 = (s.summary.get("data_base64") or "").strip()
    filename = s.summary.get("filename") or "interview_summary.json"
    ctype = s.summary.get("content_type") or "application/json"

    if data_b64:
        file_bytes = base64.b64decode(data_b64.encode("ascii"))
        st.download_button(
            label="⬇️ Download summary (JSON)",
            data=file_bytes,
            file_name=filename,
            mime=ctype,
            use_container_width=True,
        )

    finish_disabled = (ClientAction.FINISH not in s.allowed_actions)
    if st.button("✅ Finish", use_container_width=True, disabled=finish_disabled):
        s2 = api.finish(session_id=st.session_state.session_id)
        st.session_state.server = s2
        st.rerun()

# Top status panel
c1, c2, c3, c4 = st.columns(4)

# batch_index in state is "how many have been answered in this batch so far"
# but the current question being shown is (batch_index+1)th in that batch during AWAITING_ANSWER.
progress_q = min(s.batch_index + 1, max(1, s.batch_size)) if not s.interview_done else s.batch_size

with c1:
    st.metric("Testing Level", int(s.current_level))
with c2:
    st.metric("Progress", f"Q {progress_q}/{int(s.batch_size)}")
with c3:
    st.metric("Correct (this level)", int(s.batch_correct))
with c4:
    st.metric("Last Passed", int(s.last_passed_level))

st.caption(f"Phase: `{s.phase}` | Allowed: `{', '.join(s.allowed_actions) or '—'}`")

if s.interview_done:
    st.success(f"✅ Interview finished. Final level: **{int(s.final_level)}**")




st.divider()

# Chat window
if not getattr(s, "summary", None):
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Controls row
    colA, colB, colC = st.columns(3)

    with colA:
        start_disabled = (ClientAction.START not in s.allowed_actions) or s.interview_done
        if st.button("🚀 Start", use_container_width=True, disabled=start_disabled):
            try:
                # Stream the question generation
                placeholder = st.empty()
                question_text = ""

                for event in api.start_stream(session_id=st.session_state.session_id):
                    event_type = event.get("type")

                    if event_type == "question_token":
                        question_text += event.get("token", "")
                        placeholder.markdown(f"**Question (streaming):** {question_text}")

                    elif event_type == "final_state":
                        s2 = event.get("final_state")
                        st.session_state.server = s2
                        placeholder.empty()  # Clear streaming placeholder

                        if not s2.interview_done and s2.current_question:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"**Question:** {s2.current_question}"}
                            )

                st.rerun()
            except Exception as e:
                st.error(str(e))

    with colB:
        next_disabled = (not s.can_proceed) or (ClientAction.NEXT not in s.allowed_actions)
        if st.button("⏭️ Next", use_container_width=True, disabled=next_disabled):
            try:
                # Stream the next question
                placeholder = st.empty()
                question_text = ""

                for event in api.next_stream(session_id=st.session_state.session_id):
                    event_type = event.get("type")

                    if event_type == "question_token":
                        question_text += event.get("token", "")
                        placeholder.markdown(f"**Question (streaming):** {question_text}")

                    elif event_type == "final_state":
                        s2 = event.get("final_state")
                        st.session_state.server = s2
                        placeholder.empty()  # Clear streaming placeholder

                        if s2.interview_done:
                            st.session_state.messages.append(
                                {"role": "assistant",
                                 "content": f"✅ Interview finished. Final level: **{int(s2.final_level)}**"}
                            )
                        elif s2.current_question:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"**Question:** {s2.current_question}"}
                            )

                st.rerun()
            except Exception as e:
                st.error(str(e))

    with colC:
        if st.button("🧹 Trim chat", use_container_width=True):
            st.session_state.messages = st.session_state.messages[-1:]
            st.rerun()

    st.divider()

    # Voice input (only meaningful in AWAITING_ANSWER)
    from smart_interviewer.settings import settings
    audio = st.audio_input("Record your answer", sample_rate=settings.AUDIO_SAMPLE_RATE)

    send_disabled = s.interview_done or (audio is None) or (ClientAction.ANSWER not in s.allowed_actions)
    send_clicked = st.button("📨 Send Answer", use_container_width=True, disabled=send_disabled)

    if audio is not None:
        st.audio(audio, format="audio/wav")

    if send_clicked and audio is not None:
        st.session_state.messages.append({"role": "user", "content": "🎙️ (voice answer)"})

        try:
            audio_bytes = audio.getvalue()

            # Stream the evaluation feedback
            placeholder = st.empty()
            evaluation_text = ""
            followup_text = ""

            for event in api.answer_stream(
                audio_bytes=audio_bytes,
                filename=f"voice_{int(time.time())}.wav",
                content_type="audio/wav",
                session_id=st.session_state.session_id,
            ):
                event_type = event.get("type")

                if event_type == "evaluation_token":
                    evaluation_text += event.get("token", "")
                    # Display in a nice info box without the raw JSON
                    with placeholder.container():
                        st.info("🔄 Evaluating your answer...")

                elif event_type == "followup_token":
                    followup_text += event.get("token", "")
                    with placeholder.container():
                        st.success("✅ Evaluation complete!")
                        st.markdown("**Follow-up question:**")
                        st.info(followup_text)

                elif event_type == "final_state":
                    s2 = event.get("final_state")
                    st.session_state.server = s2
                    placeholder.empty()  # Clear streaming placeholder

                    # show transcript + feedback in chat
                    if s2.transcript.strip():
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"**Transcript:** {s2.transcript.strip()}"}
                        )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"**Feedback:**\n\n{s2.assistant_text}"}
                    )

                    # If follow-up is needed, your engine will be back in AWAITING_ANSWER
                    if (ClientAction.ANSWER in s2.allowed_actions) and s2.current_question:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"**Follow-up:** {s2.current_question}"}
                        )

                    if s2.interview_done:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"✅ Interview finished. Final level: **{int(s2.final_level)}**"}
                        )

            st.rerun()

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Answer failed: `{e}`"})
            st.rerun()
