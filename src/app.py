"""
app.py - Main Streamlit UI for the AI Course Assistant.
Combines Pinecone Two-Stage Vector Store (Retrieve & Re-rank), LangGraph ReAct Agent,
Tavily Search, Multimodal Vision Analysis, and Text-to-Speech synthesis with Document Citations.
Equipped with real-time execution step streaming and session-state persistence.
"""

import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.agent.agent_runner import CourseAgentRunner
    from src.agent.followup_generator import FollowUpGenerator
    from src.engines.stt_engine import STTEngine
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

# Category → phrasing/icon for resource-derived starter suggestions. Mirrors
# CourseDocumentProcessor._determine_category's folder-name heuristic, kept
# separate rather than importing that class (which pulls in fitz + the vision
# engine at module scope) just to categorize filenames for the UI.
CATEGORY_SUGGESTION_STYLE = {
    "assignments": (":material/assignment:", "What are the requirements for {title}?"),
    "workshops": (":material/school:", "Walk me through {title}."),
    "guidelines": (":material/menu_book:", "What does {title} cover?"),
    "general": (":material/lightbulb:", "Tell me about {title}."),
}


@st.cache_data(show_spinner=False)
def get_resource_documents() -> List[Dict[str, str]]:
    """One-time disk scan of resources/ for starter-suggestion source material."""
    documents = []
    if not RESOURCES_DIR.exists():
        return documents
    for pdf_path in sorted(RESOURCES_DIR.rglob("*.pdf")):
        path_str = str(pdf_path.relative_to(RESOURCES_DIR)).lower()
        if "guideline" in path_str or "guide" in path_str:
            category = "guidelines"
        elif "workshop" in path_str:
            category = "workshops"
        elif "assignment" in path_str:
            category = "assignments"
        else:
            category = "general"
        documents.append({"title": pdf_path.stem.replace("_", " ").strip(), "category": category})
    return documents


def get_starter_suggestions() -> List[Dict[str, str]]:
    """Random sample of real course documents, phrased as questions. Re-rolled
    each time a fresh/empty chat starts, cached for the rest of that chat."""
    if "starter_suggestions" not in st.session_state:
        pool = get_resource_documents()
        sample = random.sample(pool, k=min(3, len(pool))) if pool else []
        st.session_state.starter_suggestions = [
            {"icon": CATEGORY_SUGGESTION_STYLE[doc["category"]][0], "question": CATEGORY_SUGGESTION_STYLE[doc["category"]][1].format(title=doc["title"])}
            for doc in sample
        ]
    return st.session_state.starter_suggestions


def load_chat_sessions() -> List[Dict[str, Any]]:
    if not CHAT_STORE_PATH.exists():
        return []
    try:
        return json.loads(CHAT_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_chat_sessions(sessions: List[Dict[str, Any]]) -> None:
    CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAT_STORE_PATH.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
        {
            "role": m["role"],
            "content": m["content"],
            "sources": m.get("sources", []),
            "thought_log": m.get("thought_log", []),
            "follow_ups": m.get("follow_ups", []),
            "follow_ups_on_topic": m.get("follow_ups_on_topic", True),
        }
        for m in messages
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
        reset_to_new_chat()


def reset_to_new_chat() -> None:
    """Clears the live view back to a blank chat, including the starter
    suggestions so a fresh conversation gets a newly rolled sample."""
    st.session_state.messages = []
    st.session_state.current_session_id = None
    st.session_state.pop("starter_suggestions", None)


st.set_page_config(page_title="AI Course Assistant - Hackathon", page_icon="🎓", layout="wide")
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


@st.cache_resource(show_spinner="Loading Speech-to-Text Models...")
def get_stt_engine() -> "STTEngine":
    from src.engines.stt_engine import STTEngine

    return STTEngine()


@st.cache_resource(show_spinner="Initializing Follow-up Suggestion Engine...")
def get_followup_generator() -> "FollowUpGenerator":
    from src.agent.followup_generator import FollowUpGenerator

    return FollowUpGenerator(RESOURCES_DIR)


vector_store = get_vector_store()
agent_runner = get_agent_runner(vector_store)
vision_engine = get_vision_engine()
tts_engine = get_tts_engine()
followup_generator = get_followup_generator()
stt_engine = get_stt_engine()


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
    {
        "name": "Speech-to-Text",
        "type": "Voice",
        "connected": True,
        "detail": "PhoWhisper (Vietnamese) / Whisper (English) via the mic button in the chat input; language picked in the sidebar.",
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

    st.subheader("🎤 Voice Input Language")
    stt_language_label = st.radio(
        "Speech-to-text language",
        options=["Vietnamese", "English"],
        horizontal=True,
        label_visibility="collapsed",
        help="Language of the voice message recorded via the mic button in the chat input.",
    )
    stt_language = "vi" if stt_language_label == "Vietnamese" else "en"

    if st.button("New chat", icon=":material/add_comment:", use_container_width=True):
        reset_to_new_chat()
        st.rerun()

    header_col, menu_col = st.columns([5, 1], vertical_alignment="center")
    with header_col:
        st.subheader("Chat History")
    with menu_col:
        with st.popover(" ", icon=":material/more_vert:"):
            if st.button("Delete all history", icon=":material/delete_forever:", use_container_width=True):
                save_chat_sessions([])
                reset_to_new_chat()
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
# Step 3: Chat History Rendering (with persisted execution steps)
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
            if message.get("image_bytes"):
                st.image(message["image_bytes"], width=300)

            # Render persisted agent execution steps (tool calls) above the answer
            if message["role"] == "assistant" and message.get("thought_log"):
                with st.expander("🔍 View Execution Steps & Tool Calls", expanded=False):
                    for step in message["thought_log"]:
                        step_type = step.get("type")
                        if step_type == "action":
                            st.markdown(f"🛠️ **Invoked Tool:** `{step.get('tool')}`")
                            st.caption(f"Arguments: `{step.get('args')}`")
                        elif step_type == "observation":
                            st.markdown(f"📥 **Output from:** `{step.get('tool')}`")
                            st.caption(step.get("preview", ""))

            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_assistant_extras(idx, message)

# Suggestion chips above the input: context-aware follow-ups grounded in the
# last answer once there's a conversation, or a random resource-derived
# sample for the very first, empty-state message (no prior answer to build
# follow-ups from yet).
last_message = messages[-1] if messages else None
if last_message and last_message["role"] == "assistant" and last_message.get("follow_ups"):
    st.caption(
        "Continue with:"
        if last_message.get("follow_ups_on_topic", True)
        else "That is outside this course — here is what I can help you with:"
    )
    suggestions = [
        {"icon": ":material/arrow_forward:", "question": q} for q in last_message["follow_ups"]
    ]
elif not messages:
    st.caption("Try asking:")
    suggestions = get_starter_suggestions()
else:
    suggestions = []

if suggestions:
    suggestion_cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(suggestion_cols, suggestions)):
        with col:
            if st.button(suggestion["question"], icon=suggestion["icon"], key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion["question"]
                st.rerun()


# ==============================================================================
# Step 4: User Query Processing & Real-Time Streaming Loop
# ==============================================================================
chat_value = st.chat_input(
    "Ask about assignments, workshops, code errors, or external technical topics... (or use the mic)",
    accept_audio=True,
    audio_sample_rate=16000,
)
pending_prompt = st.session_state.pop("pending_prompt", None)

user_input = chat_value.text.strip() if chat_value and chat_value.text else None
voice_transcript = None
if chat_value and chat_value.audio is not None:
    with st.spinner(f"Transcribing {stt_language_label} speech..."):
        voice_transcript = stt_engine.transcribe(chat_value.audio.getvalue(), language=stt_language)

# A file_uploader keeps returning the same file on every rerun, so an image alone can
# only open a turn once. Without this the end-of-turn st.rerun() below re-enters here
# with the uploader still populated and drives the agent in a loop.
image_is_new = bool(uploaded_image) and (
    uploaded_image.file_id != st.session_state.get("processed_image_id")
)

if user_input or image_is_new or pending_prompt or voice_transcript:
    current_prompt = user_input or voice_transcript or pending_prompt or "Please inspect this uploaded image and provide guidance."
    augmented_prompt = current_prompt
    image_bytes_to_store = None

    # Process uploaded image if available
    if uploaded_image:
        st.session_state.processed_image_id = uploaded_image.file_id
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

    # Stream the agent's reasoning/tool calls live, instead of a bare spinner.
    with st.chat_message("assistant"):
        final_response_text = ""
        extracted_sources: List[Dict[str, str]] = []
        thought_log: List[Dict[str, Any]] = []
        agent_error = False

        with st.status("🧠 Agent is coordinating tools and reasoning...", expanded=True) as status_box:
            try:
                event_stream = agent_runner.stream_query(
                    user_input=augmented_prompt,
                    chat_history=messages,
                    max_history_turns=10,
                )

                for event in event_stream:
                    event_type = event.get("type")

                    if event_type == "action":
                        tool_name = event.get("tool", "tool")
                        tool_args = event.get("args", {})
                        status_box.write(f"🛠️ **Executing Tool:** `{tool_name}`")
                        st.caption(f"Arguments: `{tool_args}`")
                        thought_log.append(event)

                    elif event_type == "observation":
                        tool_name = event.get("tool", "tool")
                        preview = event.get("preview", "")
                        status_box.write(f"📥 **Received Result from:** `{tool_name}`")
                        st.caption(preview)
                        thought_log.append(event)

                    elif event_type == "final_answer":
                        final_response_text = event.get("content", "")
                        extracted_sources = event.get("sources", [])

                if thought_log:
                    status_box.update(
                        label="✔ Tool execution and retrieval completed!",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_box.update(
                        label="✔ Direct answer generated (No tool calls required)",
                        state="complete",
                        expanded=False,
                    )

            except Exception as err:
                status_box.update(label="❌ An error occurred during agent execution", state="error")
                final_response_text = f"An error occurred while executing the agent: {err}"
                agent_error = True
                print(f"[X] Agent Runner Streaming Error: {err}")

    # Context-aware follow-up suggestions, grounded in this specific answer —
    # skipped on error, since there's nothing useful to follow up on. An off-topic
    # question comes back flagged, with course-oriented redirect questions instead.
    follow_ups: List[str] = []
    follow_ups_on_topic = True
    if final_response_text and not agent_error:
        followup_result = followup_generator.generate(
            user_question=current_prompt,
            assistant_answer=final_response_text,
            sources=extracted_sources,
            used_web_search=any(step.get("tool") == "tavily_search" for step in thought_log),
        )
        follow_ups = followup_result.questions
        follow_ups_on_topic = followup_result.is_on_topic

    # Append Assistant Message to History; the rerun re-enters Step 3's loop,
    # which is the single place that renders the execution log, Listen
    # button, sources, and audio player — no duplicate rendering here.
    messages.append(
        {
            "role": "assistant",
            "content": final_response_text,
            "sources": extracted_sources,
            "thought_log": thought_log,
            "follow_ups": follow_ups,
            "follow_ups_on_topic": follow_ups_on_topic,
        }
    )
    persist_current_session(messages)
    st.rerun()
