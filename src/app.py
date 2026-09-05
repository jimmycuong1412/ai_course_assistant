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


vector_store = get_vector_store()
agent_runner = get_agent_runner(vector_store)
vision_engine = get_vision_engine()
tts_engine = get_tts_engine()


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

# Display previous conversation messages
for message in st.session_state.messages:
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

            # Render audio player
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")


# ==============================================================================
# Step 4: User Query Processing & Real-Time Streaming Loop
# ==============================================================================
user_input = st.chat_input("Ask about assignments, workshops, code errors, or external technical topics...")

if user_input or uploaded_image:
    current_prompt = user_input or "Please inspect this uploaded image and provide guidance."
    image_bytes_to_store = None
    augmented_prompt = current_prompt

    # Process uploaded image if available
    if uploaded_image:
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

            # Synthesize TTS Audio strictly for the main textual answer
            if enable_tts:
                try:
                    audio_bytes = tts_engine.synthesize(final_response_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay_audio)
                except Exception as tts_err:
                    print(f"[LOG] TTS Warning: {tts_err}")

    # Append Assistant Message to History (Persisting Execution Log)
    if final_response_text:
        assistant_entry = {
            "role": "assistant",
            "content": final_response_text,
            "sources": extracted_sources,
            "thought_log": thought_log,
        }
        if audio_bytes:
            assistant_entry["audio"] = audio_bytes
        st.session_state.messages.append(assistant_entry)