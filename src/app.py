"""
app.py - Main Streamlit UI for the AI Course Assistant.
Combines Pinecone Two-Stage Vector Store (Retrieve & Re-rank), LangGraph ReAct Agent,
Tavily Search, Multimodal Vision Analysis, and Text-to-Speech synthesis with Document Citations.
Equipped with real-time execution step streaming and session-state persistence.
"""

from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.agent.agent_runner import CourseAgentRunner
from src.agent.followup_generator import STARTER_QUESTIONS, FollowUpGenerator
from src.engines.tts_engine import TTSEngine
from src.rag.vector_store import CourseVectorStore
from src.engines.vision_engine import VisionEngine

# ==============================================================================
# Step 0: Page Config & Resource Paths
# ==============================================================================
load_dotenv()
RESOURCES_DIR = Path(__file__).parent.parent / "resources"

st.set_page_config(page_title="AI Course Assistant - Hackathon", page_icon="🎓", layout="wide")
st.title("🎓 AI Application Engineer - Smart Assistant")


# ==============================================================================
# Step 1: Resource Caching & Initialization
# ==============================================================================
@st.cache_resource(show_spinner="Connecting to Pinecone Vector Store...")
def get_vector_store() -> CourseVectorStore:
    return CourseVectorStore(RESOURCES_DIR)


@st.cache_resource(show_spinner="Initializing LangGraph ReAct Agent...")
def get_agent_runner(_vector_store: CourseVectorStore) -> CourseAgentRunner:
    return CourseAgentRunner(_vector_store)


@st.cache_resource(show_spinner="Initializing Multimodal Vision Engine...")
def get_vision_engine() -> VisionEngine:
    return VisionEngine()


@st.cache_resource(show_spinner="Initializing TTS Voice Engine...")
def get_tts_engine() -> TTSEngine:
    return TTSEngine()


@st.cache_resource(show_spinner="Initializing Follow-up Suggestion Engine...")
def get_followup_generator() -> FollowUpGenerator:
    return FollowUpGenerator(RESOURCES_DIR)


vector_store = get_vector_store()
agent_runner = get_agent_runner(vector_store)
vision_engine = get_vision_engine()
tts_engine = get_tts_engine()
followup_generator = get_followup_generator()


def suggestion_caption(is_on_topic: bool) -> str:
    """Labels the chips as a natural next step, or as a nudge back to the course."""
    if is_on_topic:
        return "💡 Suggested follow-up questions:"
    return "🎓 That is outside this course - here is what I can help you with:"


def render_suggestion_chips(questions: list, key_prefix: str) -> None:
    """
    Renders clickable question chips. A click stores the question in session state and
    reruns, so it enters the pipeline exactly like a manually typed chat message
    (st.chat_input cannot be populated programmatically).
    """
    if not questions:
        return

    columns = st.columns(len(questions))
    for idx, question in enumerate(questions):
        if columns[idx].button(question, key=f"{key_prefix}_{idx}", use_container_width=True):
            st.session_state.pending_input = question
            st.rerun()


# ==============================================================================
# Step 2: Sidebar Settings & Knowledge Base Status
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Architecture: **LangGraph ReAct Agent + Pinecone Two-Stage RAG + Tavily**")

    st.subheader("🎙️ Voice Output (TTS)")
    enable_tts = st.toggle("Enable Voice Output", value=True)
    autoplay_audio = st.toggle("Auto-play Audio", value=True, disabled=not enable_tts)

    st.subheader("🖼️ Multimodal Input")
    uploaded_image = st.file_uploader(
        "Upload screenshot or diagram (Optional)",
        type=["png", "jpg", "jpeg"],
        help="Upload error traceback screenshots, code images, or architecture diagrams.",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image Preview", use_container_width=True)

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📊 System Status")
    st.success(f"Connected to Pinecone Index: `{vector_store.index_name}`")
    st.info("⚡ Two-Stage Retrieval Active: **Pinecone (k=15) ➔ bge-reranker-v2-m3 (top=5)**")


# ==============================================================================
# Step 3: Session State & Chat History Rendering (with Execution Persistence)
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Capture chat input before rendering history so stale suggestion chips can be hidden while
# a new turn is being generated (st.chat_input always renders pinned to the bottom of the page).
typed_input = st.chat_input("Ask about assignments, workshops, code errors, or external technical topics...")
user_input = typed_input or st.session_state.pop("pending_input", None)

# A file_uploader keeps returning the same file on every rerun, so only an image that has
# not been answered yet counts as new input. It is still attached to a typed question.
image_is_new = bool(uploaded_image) and (
    uploaded_image.file_id != st.session_state.get("processed_image_id")
)
is_new_turn = bool(user_input or image_is_new)

# Display previous conversation messages
last_message_index = len(st.session_state.messages) - 1
for message_index, message in enumerate(st.session_state.messages):
    role = message.get("role")
    content = message.get("content")

    if role in ["user", "assistant"] and content:
        with st.chat_message(role):
            # Render uploaded images if present in user turn
            if message.get("image_bytes"):
                st.image(message["image_bytes"], width=300)

            # Render persistent agent execution steps from history
            if role == "assistant" and message.get("thought_log"):
                with st.expander("🔍 View Execution Steps & Tool Calls", expanded=False):
                    for step in message["thought_log"]:
                        step_type = step.get("type")
                        if step_type == "action":
                            st.markdown(f"🛠️ **Invoked Tool:** `{step.get('tool')}`")
                            st.caption(f"Arguments: `{step.get('args')}`")
                        elif step_type == "observation":
                            st.markdown(f"📥 **Output from:** `{step.get('tool')}`")
                            st.caption(step.get("preview", ""))

            # Render assistant text response
            st.markdown(content)

            # Render citations
            if message.get("sources"):
                with st.expander("📚 Referenced Course Materials", expanded=False):
                    for src in message["sources"]:
                        st.markdown(f"* 📄 **{src['file']}** *(Pages: {src['pages']})*")

            # Render audio player. The one-shot "autoplay" flag is consumed on the first
            # render after synthesis so the clip does not replay on later reruns.
            if message.get("audio"):
                st.audio(
                    message["audio"],
                    format="audio/mp3",
                    autoplay=message.pop("autoplay", False),
                )

            # Offer follow-up suggestions only for the newest answer, and only while no
            # new turn is already being generated below.
            if (
                role == "assistant"
                and message_index == last_message_index
                and not is_new_turn
                and message.get("followups")
            ):
                st.caption(suggestion_caption(message.get("followups_on_topic", True)))
                render_suggestion_chips(message["followups"], key_prefix=f"followup_hist_{message_index}")


# Cold-start prompts shown before the first question of a session
if not st.session_state.messages and not is_new_turn:
    st.caption("👋 New chat - try one of these to get started:")
    render_suggestion_chips(STARTER_QUESTIONS, key_prefix="starter")


# ==============================================================================
# Step 4: User Query Processing & Real-Time Streaming Loop
# ==============================================================================
if is_new_turn:
    current_prompt = user_input or "Please inspect this uploaded image and provide guidance."
    image_bytes_to_store = None
    augmented_prompt = current_prompt

    # Process uploaded image if available
    if uploaded_image:
        st.session_state.processed_image_id = uploaded_image.file_id
        image_bytes = uploaded_image.read()
        image_bytes_to_store = image_bytes
        with st.spinner("Analyzing uploaded image with Multimodal Vision..."):
            vision_context = vision_engine.analyze_image_bytes(image_bytes, user_note=current_prompt)
            augmented_prompt = f"{current_prompt}\n\n{vision_context}"

    # Record User Message
    user_entry = {"role": "user", "content": current_prompt}
    if image_bytes_to_store:
        user_entry["image_bytes"] = image_bytes_to_store
    st.session_state.messages.append(user_entry)

    with st.chat_message("user"):
        st.markdown(current_prompt)
        if image_bytes_to_store:
            st.image(image_bytes_to_store, width=300)

    # Generate Assistant Response with Real-Time Tool Execution Streaming
    with st.chat_message("assistant"):
        final_response_text = ""
        extracted_sources = []
        thought_log = []
        audio_bytes = None
        followup_questions = []
        followups_on_topic = True
        agent_failed = False

        # Container for Real-Time Execution Updates
        with st.status("🧠 Agent is coordinating tools and reasoning...", expanded=True) as status_box:
            try:
                # Stream events from LangGraph
                event_stream = agent_runner.stream_query(
                    user_input=augmented_prompt,
                    chat_history=st.session_state.messages,
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

                # Update status box to complete and collapse
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
                agent_failed = True
                final_response_text = f"An error occurred while executing the agent: {err}"
                st.error(final_response_text)
                print(f"[X] Agent Runner Streaming Error: {err}")

        # Render Final Markdown Response
        if final_response_text:
            st.markdown(final_response_text)

            # Render Citations in an Expander if sources are present
            if extracted_sources:
                with st.expander("📚 Referenced Course Materials", expanded=False):
                    for src in extracted_sources:
                        st.markdown(f"* 📄 **{src['file']}** *(Pages: {src['pages']})*")

            # Generate contextual follow-up suggestions. They are persisted with the turn
            # and rendered by the history loop, not here.
            if not agent_failed:
                used_web_search = any(
                    step.get("tool") == "tavily_search" for step in thought_log
                )
                with st.spinner("Preparing follow-up suggestions..."):
                    followup_result = followup_generator.generate(
                        user_question=current_prompt,
                        assistant_answer=final_response_text,
                        sources=extracted_sources,
                        used_web_search=used_web_search,
                    )
                followup_questions = followup_result.questions
                followups_on_topic = followup_result.is_on_topic

            # Synthesize TTS Audio strictly for the main textual answer
            if enable_tts:
                try:
                    audio_bytes = tts_engine.synthesize(final_response_text)
                except Exception as tts_err:
                    print(f"[LOG] TTS Warning: {tts_err}")

    # Append Assistant Message to History (Persisting Execution Log)
    if final_response_text:
        assistant_entry = {
            "role": "assistant",
            "content": final_response_text,
            "sources": extracted_sources,
            "thought_log": thought_log,
            "followups": followup_questions,
            "followups_on_topic": followups_on_topic,
        }
        if audio_bytes:
            assistant_entry["audio"] = audio_bytes
            assistant_entry["autoplay"] = autoplay_audio
        st.session_state.messages.append(assistant_entry)

        # Re-render the completed turn through the history loop above. Keeping a single
        # render path is what prevents duplicated audio players and stale chips: Streamlit
        # replaces elements by position, so two differently ordered paths cannot align.
        st.rerun()