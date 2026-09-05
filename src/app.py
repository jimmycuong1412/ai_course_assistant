"""
app.py - Main Streamlit UI for the AI Course Assistant.
Combines Pinecone Two-Stage Vector Store (Retrieve & Re-rank), LangGraph ReAct Agent,
Tavily Search, Multimodal Vision Analysis, and Text-to-Speech synthesis with Document Citations.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.agent.agent_runner import CourseAgentRunner
    from src.engines.tts_engine import TTSEngine
    from src.engines.vision_engine import VisionEngine
    from src.rag.vector_store import CourseVectorStore

# ==============================================================================
# Step 0: Page Config & Resource Paths
# ==============================================================================
# NOTE: keep this module free of heavy imports (langchain/langgraph/pinecone/etc.)
# above this point. Streamlit can't paint anything until the script starts
# emitting output, so any import-time cost here shows up as a blank page before
# the title even appears. Those libraries are imported lazily inside the
# cache_resource factories below instead.
load_dotenv()
RESOURCES_DIR = Path(__file__).parent.parent / "resources"
# Chat history survives page refreshes / app restarts via this local JSON file
# (in .cache/, already gitignored). Only role/content/sources are persisted —
# screenshots and generated audio stay in-memory only, to keep the file small.
CHAT_STORE_PATH = Path(__file__).parent.parent / ".cache" / "chat_sessions.json"


def load_chat_sessions() -> List[Dict[str, Any]]:
    if not CHAT_STORE_PATH.exists():
        return []
    try:
        return json.loads(CHAT_STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_chat_sessions(sessions: List[Dict[str, Any]]) -> None:
    CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAT_STORE_PATH.write_text(json.dumps(sessions, ensure_ascii=False, indent=2))


def persist_current_session(messages: List[Dict[str, Any]]) -> None:
    """Upserts the active conversation into the on-disk session list, auto-naming
    it from the first user message the first time it's saved."""
    sessions = load_chat_sessions()
    session_id = st.session_state.get("current_session_id")
    if session_id is None:
        session_id = uuid.uuid4().hex[:8]
        st.session_state.current_session_id = session_id

    existing = next((s for s in sessions if s["id"] == session_id), None)
    if existing:
        title = existing["title"]
    else:
        first_user_message = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
        title = first_user_message[:40] + "…" if len(first_user_message) > 40 else first_user_message

    serializable_messages = [
        {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])} for m in messages
    ]
    sessions = [s for s in sessions if s["id"] != session_id]
    sessions.append(
        {"id": session_id, "title": title, "updated_at": time.time(), "messages": serializable_messages}
    )
    save_chat_sessions(sessions)


def delete_chat_session(session_id: str) -> None:
    sessions = [s for s in load_chat_sessions() if s["id"] != session_id]
    save_chat_sessions(sessions)
    if st.session_state.get("current_session_id") == session_id:
        st.session_state.messages = []
        st.session_state.current_session_id = None


st.set_page_config(page_title="AI Course Assistant - WS4", page_icon="🎓", layout="wide")
st.title("🎓 AI Application Engineer - Smart Assistant")

# Native <audio> controls can't be restyled from Python; this is the one CSS
# escape hatch to make the per-message voice player feel less like a bare
# browser widget (rounded pill, app accent color for the play button/seek bar).
st.markdown(
    """
    <style>
    audio {
        width: 100% !important;
        max-width: 360px;
        height: 42px;
        border-radius: 999px;
        accent-color: #FF4B4B;
    }
    audio::-webkit-media-controls-panel {
        background-color: #ffffff;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Step 1: Resource Caching & Initialization
# ==============================================================================
@st.cache_resource(show_spinner="Connecting to Pinecone Vector Store...")
def get_vector_store() -> "CourseVectorStore":
    from src.rag.vector_store import CourseVectorStore

    return CourseVectorStore(RESOURCES_DIR)


@st.cache_resource(show_spinner="Initializing LangGraph ReAct Agent...")
def get_agent_runner(_vector_store: "CourseVectorStore") -> "CourseAgentRunner":
    from src.agent.agent_runner import CourseAgentRunner

    return CourseAgentRunner(_vector_store)


@st.cache_resource(show_spinner="Initializing Multimodal Vision Engine...")
def get_vision_engine() -> "VisionEngine":
    from src.engines.vision_engine import VisionEngine

    return VisionEngine()


@st.cache_resource(show_spinner="Initializing TTS Voice Engine...")
def get_tts_engine() -> "TTSEngine":
    from src.engines.tts_engine import TTSEngine

    return TTSEngine()


vector_store = get_vector_store()
agent_runner = get_agent_runner(vector_store)
vision_engine = get_vision_engine()
tts_engine = get_tts_engine()


# ==============================================================================
# Step 2: Sidebar
# ==============================================================================
CONNECTORS = [
    {
        "name": "Pinecone Vector Store",
        "type": "RAG retrieval",
        "connected": True,
        "detail": (
            f"Index: `{vector_store.index_name}`\n\n"
            "Two-stage retrieval: Pinecone similarity search (k=15) "
            "➔ Pinecone Inference rerank, `bge-reranker-v2-m3` (top=5)."
        ),
    },
    {
        "name": "OpenAI Chat Model",
        "type": "LLM",
        "connected": True,
        "detail": f"Model: `{os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}`\n\nDrives the LangGraph ReAct agent loop.",
    },
    {
        "name": "OpenAI Embeddings",
        "type": "Embedding model",
        "connected": True,
        "detail": f"Model: `{os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}`\n\nUsed to embed course documents and queries.",
    },
    {
        "name": "Tavily Web Search",
        "type": "Tool",
        "connected": bool(os.getenv("TAVILY_API_KEY")),
        "detail": "Fallback tool for external/live info the course knowledge base doesn't cover."
        if os.getenv("TAVILY_API_KEY")
        else "Not configured — set `TAVILY_API_KEY` in `.env` to enable web search fallback.",
    },
    {
        "name": "Vision Engine",
        "type": "Multimodal",
        "connected": True,
        "detail": "gpt-4o-mini with structured output. Transcribes uploaded screenshots (errors, code, diagrams).",
    },
    {
        "name": "Text-to-Speech",
        "type": "Voice",
        "connected": True,
        "detail": "gTTS with automatic English/Vietnamese detection, generated on demand via each reply's Listen button.",
    },
]


@st.dialog("Settings", width="large")
def show_settings_dialog() -> None:
    st.caption("Architecture: LangGraph ReAct Agent + Pinecone Two-Stage RAG + Tavily")
    st.subheader("Connectors")
    for connector in CONNECTORS:
        status_icon = ":material/check_circle:" if connector["connected"] else ":material/cancel:"
        status_text = "Connected" if connector["connected"] else "Not connected"
        with st.expander(f"{connector['name']}  ·  {connector['type']}  ·  {status_text}", icon=status_icon):
            st.markdown(connector["detail"])


with st.sidebar:
    st.subheader("Multimodal Input")
    uploaded_image = st.file_uploader(
        "Upload screenshot or diagram (optional)",
        type=["png", "jpg", "jpeg"],
        help="Upload error traceback screenshots, code images, or architecture diagrams.",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_container_width=True)

    if st.button("New chat", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_session_id = None
        st.rerun()

    header_col, menu_col = st.columns([5, 1], vertical_alignment="center")
    with header_col:
        st.subheader("Chat History")
    with menu_col:
        with st.popover(" ", icon=":material/more_vert:"):
            if st.button("Delete all history", icon=":material/delete_forever:", use_container_width=True):
                save_chat_sessions([])
                st.session_state.messages = []
                st.session_state.current_session_id = None
                st.rerun()

    saved_sessions = sorted(load_chat_sessions(), key=lambda s: s["updated_at"], reverse=True)
    with st.expander(f"{len(saved_sessions)} saved" if saved_sessions else "No saved chats yet", icon=":material/history:"):
        for session in saved_sessions:
            title_col, delete_col = st.columns([5, 1], vertical_alignment="center")
            with title_col:
                if st.button(session["title"], key=f"load_{session['id']}", use_container_width=True):
                    st.session_state.messages = session["messages"]
                    st.session_state.current_session_id = session["id"]
                    st.rerun()
            with delete_col:
                if st.button(" ", icon=":material/delete:", key=f"delete_{session['id']}", help="Delete this chat"):
                    delete_chat_session(session["id"])
                    st.rerun()

    st.divider()
    if st.button("Settings", icon=":material/settings:", use_container_width=True):
        show_settings_dialog()


# ==============================================================================
# Step 3: Chat History Rendering
# ==============================================================================
if "messages" not in st.session_state:
    # Fresh browser session (e.g. after a page refresh) — reopen whatever
    # conversation was most recently active instead of starting blank.
    sessions = load_chat_sessions()
    latest = max(sessions, key=lambda s: s["updated_at"], default=None)
    st.session_state.messages = latest["messages"] if latest else []
    st.session_state.current_session_id = latest["id"] if latest else None
messages = st.session_state.messages

# Message indices whose audio should autoplay on this render only — set right
# when "Listen" is clicked, consumed (and cleared) the moment it's rendered,
# so a later, unrelated rerun doesn't replay old audio on its own.
if "autoplay_once" not in st.session_state:
    st.session_state.autoplay_once = set()


def render_assistant_extras(idx: int, message: Dict[str, Any]) -> None:
    """Renders Listen, source citations, and the audio player (once generated) in one row."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if st.button("Listen", icon=":material/volume_up:", key=f"listen_{idx}"):
            if not message.get("audio"):
                with st.spinner("Generating audio..."):
                    try:
                        message["audio"] = tts_engine.synthesize(message["content"])
                    except Exception as tts_err:
                        st.warning(f"Couldn't generate audio: {tts_err}")
            if message.get("audio"):
                st.session_state.autoplay_once.add(idx)
        if message.get("sources"):
            with st.popover(f"{len(message['sources'])} sources", icon=":material/menu_book:"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['file']}** *(Pages: {src['pages']})*")
        if message.get("audio"):
            should_autoplay = idx in st.session_state.autoplay_once
            st.session_state.autoplay_once.discard(idx)
            st.audio(message["audio"], format="audio/mp3", autoplay=should_autoplay, width=360)


# Display previous conversation messages
for idx, message in enumerate(messages):
    if message.get("role") in ["user", "assistant"] and message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image_bytes"):
                st.image(message["image_bytes"], width=300)
            if message["role"] == "assistant":
                render_assistant_extras(idx, message)

# Starter suggestions — only useful before the conversation has content;
# once there's history, real context beats generic prompts.
SUGGESTED_QUESTIONS = [
    ("What's due in the current assignment?", ":material/assignment:"),
    ("Walk me through the latest workshop steps.", ":material/school:"),
    ("I'm getting an error in my code — how do I debug it?", ":material/bug_report:"),
]

if not messages:
    st.caption("Try asking:")
    suggestion_cols = st.columns(3)
    for i, (col, (question, icon)) in enumerate(zip(suggestion_cols, SUGGESTED_QUESTIONS)):
        with col:
            if st.button(question, icon=icon, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()


# ==============================================================================
# Step 4: User Query Processing & Execution Loop
# ==============================================================================
user_input = st.chat_input("Ask about assignments, workshops, code errors, or external technical topics...")
pending_prompt = st.session_state.pop("pending_prompt", None)

if user_input or uploaded_image or pending_prompt:
    current_prompt = user_input or pending_prompt or "Please inspect this uploaded image and provide guidance."
    augmented_prompt = current_prompt
    image_bytes_to_store = None

    # Process uploaded image if available
    if uploaded_image:
        image_bytes_to_store = uploaded_image.read()
        with st.spinner("Analyzing uploaded image with Multimodal Vision..."):
            vision_context = vision_engine.analyze_image_bytes(image_bytes_to_store, user_note=current_prompt)
            augmented_prompt = f"{current_prompt}\n\n{vision_context}"

    # Record User Message
    user_entry = {"role": "user", "content": current_prompt}
    if image_bytes_to_store:
        user_entry["image_bytes"] = image_bytes_to_store
    messages.append(user_entry)

    # Render it now — Step 3's loop already ran earlier this script pass, so
    # without this the user's bubble would stay invisible until the rerun
    # below finishes, leaving only a bare spinner in the meantime.
    with st.chat_message("user"):
        st.markdown(current_prompt)
        if image_bytes_to_store:
            st.image(image_bytes_to_store, width=300)

    with st.spinner("Agent is reasoning and executing tools..."):
        try:
            # Invoke LangGraph ReAct Agent
            full_response, extracted_sources = agent_runner.process_query(
                user_input=augmented_prompt,
                chat_history=messages,
                max_history_turns=10,
            )
        except Exception as err:
            full_response = f"An error occurred while executing the agent: {err}"
            extracted_sources = []
            print(f"[X] Agent Runner Error: {err}")

    # Append Assistant Message to History; the rerun re-enters Step 3's loop,
    # which is the single place that renders the Listen button + citations.
    messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": extracted_sources,
        }
    )
    persist_current_session(messages)
    st.rerun()