"""
🍷 Wine AI Sommelier Chatbot
================================
Agentic Vectorless RAG + Voice AI (Groq Whisper + gTTS)
Inspired by PageIndex (https://github.com/VectifyAI/PageIndex)

Run: streamlit run wine_chatbot.py
"""

import os, io, time, json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine AI Sommelier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* Global */
  .main { background: #fdf8f5; }

  /* Title */
  .app-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #7c0a02, #c0392b, #8e44ad);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 2px;
  }
  .app-sub {
    text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 20px;
  }

  /* Chat bubbles */
  .bubble-user {
    background: #7c0a02; color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; margin: 8px 0 8px 60px;
    font-size: 0.97rem; line-height: 1.5;
  }
  .bubble-ai {
    background: white; color: #222;
    border: 1.5px solid #e8b4b8;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px; margin: 8px 60px 8px 0;
    font-size: 0.97rem; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .bubble-system {
    background: #f0fff4; border-left: 3px solid #38a169;
    border-radius: 8px; padding: 8px 14px;
    margin: 4px 0; font-size: 0.88rem; color: #2d7f4f;
  }

  /* Step indicator */
  .step-badge {
    display: inline-block; background: #7c0a02; color: white;
    border-radius: 20px; padding: 3px 12px; font-size: 0.8rem;
    font-weight: 600; margin: 3px 0;
  }

  /* Source citation */
  .citation {
    background: #fff8e7; border: 1px solid #f0c040;
    border-radius: 8px; padding: 8px 14px; margin-top: 10px;
    font-size: 0.85rem; color: #5a3e00;
  }

  /* Book card */
  .book-card {
    background: white; border: 1px solid #e0d0d0;
    border-radius: 12px; padding: 14px;
    margin: 6px 0; transition: border-color 0.2s;
  }
  .book-card:hover { border-color: #7c0a02; }
  .book-card h4 { margin: 0 0 4px; color: #7c0a02; font-size: 0.95rem; }
  .book-card p  { margin: 0; color: #666; font-size: 0.82rem; }

  /* Sidebar */
  .sidebar-section { margin-bottom: 20px; }
  .sidebar-title {
    font-weight: 700; color: #7c0a02;
    font-size: 0.9rem; margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }

  /* Input area */
  .stTextArea textarea { border-radius: 12px !important; }
  .stButton button {
    border-radius: 10px !important;
    font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Imports (lazy to speed startup)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_agent_module():
    try:
        import wine_rag_agent as agent
        return agent
    except Exception as e:
        st.error(f"Failed to import wine_rag_agent: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_indexer_module():
    try:
        import wine_rag_indexer as indexer
        return indexer
    except Exception as e:
        st.error(f"Failed to import wine_rag_indexer: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "agent_history" not in st.session_state: st.session_state.agent_history = []
if "step_log"      not in st.session_state: st.session_state.step_log      = []
if "last_audio"    not in st.session_state: st.session_state.last_audio    = None
if "tts_enabled"   not in st.session_state: st.session_state.tts_enabled   = True

# ─────────────────────────────────────────────────────────────────────────────
# Voice helpers
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Groq Whisper STT (free)."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return ""
    try:
        import tempfile
        from groq import Groq
        client = Groq(api_key=api_key)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=(filename, f.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en",
            )
        os.unlink(tmp_path)
        return str(transcript).strip()
    except Exception as e:
        st.warning(f"Voice transcription failed: {e}")
        return ""


def text_to_speech(text: str) -> bytes | None:
    """gTTS Text-to-Speech (free, no API key)."""
    try:
        from gtts import gTTS
        # Trim to first 400 chars for spoken response
        spoken = text[:400] + ("…" if len(text) > 400 else "")
        # Remove citation block for speech
        if "📚 Sources:" in spoken:
            spoken = spoken[:spoken.index("📚 Sources:")].strip()
        tts = gTTS(text=spoken, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine AI Sommelier")
    st.markdown("*Agentic Vectorless RAG*")
    st.markdown("---")

    # ── Book status ────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-title">📚 Wine Books</p>', unsafe_allow_html=True)
    indexer = get_indexer_module()
    if indexer:
        status = indexer.get_index_status()
        for book_id, info in status.items():
            icon = "✅" if info["indexed"] else ("📄" if info["pdf_exists"] else "❌")
            st.markdown(
                f'<div class="book-card">'
                f'<h4>{info["emoji"]} {info["title"][:45]}…</h4>'
                f'<p>{icon} {"Indexed & ready" if info["indexed"] else ("PDF found, not indexed" if info["pdf_exists"] else "PDF missing")}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Index controls
        with st.expander("⚙️ Index Management"):
            book_choice = st.selectbox(
                "Select book to index",
                options=list(status.keys()),
                format_func=lambda k: status[k]["title"][:40]
            )
            col_a, col_b = st.columns(2)
            if col_a.button("📋 Index Book", use_container_width=True):
                with st.spinner(f"Indexing '{book_choice}'… (may take 3-5 min for large books)"):
                    try:
                        indexer.index_book(book_choice)
                        st.success("Indexed! ✅")
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            if col_b.button("🔄 Re-index", use_container_width=True):
                with st.spinner("Re-indexing…"):
                    try:
                        indexer.index_book(book_choice, force=True)
                        st.success("Re-indexed! ✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")

    # ── Voice settings ─────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-title">🎤 Voice Settings</p>', unsafe_allow_html=True)
    st.session_state.tts_enabled = st.toggle("🔊 Voice responses", value=st.session_state.tts_enabled)

    audio_file = st.file_uploader(
        "🎤 Upload voice question (WAV/MP3)",
        type=["wav", "mp3", "m4a", "webm"],
        label_visibility="visible"
    )

    if audio_file:
        st.audio(audio_file)
        if st.button("📝 Transcribe Voice", use_container_width=True):
            with st.spinner("Transcribing with Groq Whisper…"):
                transcript = transcribe_audio(audio_file.read(), audio_file.name)
                if transcript:
                    st.success(f"Heard: *{transcript}*")
                    st.session_state["voice_input"] = transcript
                else:
                    st.warning("Could not transcribe. Check your GROQ_API_KEY.")

    st.markdown("---")

    # ── Controls ───────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-title">⚙️ Controls</p>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.agent_history = []
        st.session_state.step_log      = []
        st.session_state.last_audio    = None
        st.rerun()

    if st.button("📋 View Last Steps", use_container_width=True):
        if st.session_state.step_log:
            st.markdown("**Agent steps (last query):**")
            for step in st.session_state.step_log:
                st.markdown(f'<div class="bubble-system">{step}</div>',
                            unsafe_allow_html=True)
        else:
            st.info("No steps recorded yet.")

    st.markdown("---")
    st.markdown(
        "**How it works:**\n\n"
        "This chatbot uses **Agentic Vectorless RAG** (inspired by [PageIndex](https://github.com/VectifyAI/PageIndex)):\n\n"
        "- 🌲 Builds a **tree index** from each PDF book\n"
        "- 🤖 LLM **reasons** over the tree to find relevant pages\n"
        "- 📄 Fetches **exact pages** — no vector similarity\n"
        "- 📚 Cites **book + page number** in every answer\n\n"
        "*No vector DB. No embeddings. Pure LLM reasoning.*",
        unsafe_allow_html=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">🍷 Wine AI Sommelier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Powered by Agentic Vectorless RAG · '
    'Groq LLaMA 3.1 · gTTS Voice · '
    '<a href="https://github.com/VectifyAI/PageIndex" target="_blank">PageIndex-inspired</a></p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Suggested questions
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💬 Try asking:")
    suggestions = [
        "What role does volatile acidity play in wine quality?",
        "How does alcohol content affect the perception of wine quality?",
        "What is malolactic fermentation and why is it used?",
        "How do tasters evaluate wine quality professionally?",
        "What chemical factors most influence red wine quality scores?",
        "Explain how sulphur dioxide protects wine during aging.",
        "What is the relationship between pH and wine preservation?",
        "How does residual sugar affect wine taste perception?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(f"💬 {s}", key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_question"] = s
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        elif msg["role"] == "assistant":
            # Split citation from main text
            content = msg["content"]
            if "📚 Sources:" in content:
                main, citation = content.split("📚 Sources:", 1)
                st.markdown(
                    f'<div class="bubble-ai">🍷 {main.strip()}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="citation">📚 Sources: {citation.strip()}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bubble-ai">🍷 {content}</div>',
                    unsafe_allow_html=True
                )

# Last audio
if st.session_state.last_audio and st.session_state.tts_enabled:
    st.audio(st.session_state.last_audio, format="audio/mp3")


# ─────────────────────────────────────────────────────────────────────────────
# Input area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

# Pre-fill from voice or suggestion
default_val = ""
if "voice_input" in st.session_state:
    default_val = st.session_state.pop("voice_input")
elif "pending_question" in st.session_state:
    default_val = st.session_state.pop("pending_question")

input_col, btn_col = st.columns([5, 1])
with input_col:
    user_input = st.text_area(
        "Ask the Wine AI Sommelier anything about wine chemistry, tasting, or quality…",
        value=default_val,
        height=80,
        placeholder="e.g. What is the role of tannins in wine aging?",
        label_visibility="collapsed",
        key="chat_input"
    )
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    send_btn = st.button("🚀 Ask", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Process question
# ─────────────────────────────────────────────────────────────────────────────
question = (user_input.strip()
            if (send_btn or (default_val and not user_input == ""))
            else "")

if question:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": question})

    # Reset step log for this query
    st.session_state.step_log = []

    # Progress placeholder
    progress_ph = st.empty()
    answer_ph   = st.empty()

    # Stream callback for live step display
    def stream_callback(step_text: str):
        st.session_state.step_log.append(step_text)
        progress_ph.markdown(
            f'<div class="bubble-system">{step_text}</div>',
            unsafe_allow_html=True
        )

    # Run agent
    with st.spinner(""):
        agent = get_agent_module()
        if agent is None:
            final = "Error: Could not load agent module. Check wine_rag_agent.py."
            updated_history = []
        else:
            try:
                final, updated_history = agent.answer(
                    question,
                    chat_history=st.session_state.agent_history[-6:],  # keep last 3 turns
                    stream_callback=stream_callback,
                )
                # Update agent history (keep system msg separate)
                st.session_state.agent_history = [
                    m for m in updated_history
                    if m["role"] != "system"
                ]
            except Exception as e:
                final = f"⚠️ Error during retrieval: {e}"
                updated_history = []

    # Clear progress
    progress_ph.empty()

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": final})

    # TTS
    if st.session_state.tts_enabled:
        audio_bytes = text_to_speech(final)
        st.session_state.last_audio = audio_bytes
    else:
        st.session_state.last_audio = None

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Bottom info bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**🧠 LLM**\nGroq LLaMA 3.1 70B")
with c2:
    st.markdown("**🌲 RAG Type**\nAgentic Vectorless")
with c3:
    st.markdown("**🎤 STT**\nGroq Whisper Large v3")
with c4:
    st.markdown("**🔊 TTS**\ngTTS (Google, free)")