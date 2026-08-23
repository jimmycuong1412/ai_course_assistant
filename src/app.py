"""
app.py - Main Streamlit UI for the AI Course Assistant.
Combines Pinecone Vector Store, LangGraph ReAct Agent, Tavily Search,
Multimodal Vision Analysis, and Text-to-Speech synthesis with Document Citations.
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

st.set_page_config(page_title="AI Course Assistant - WS4", page_icon="🎓", layout="wide")
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
    st.caption("Architecture: **LangGraph ReAct Agent + Pinecone + Tavily**")

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


# ==============================================================================
# Step 3: Session State & Chat History Rendering
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
    if message.get("role") in ["user", "assistant"] and message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image_bytes"):
                st.image(message["image_bytes"], width=300)
            if message.get("sources"):
                with st.expander("📚 Referenced Course Materials", expanded=False):
                    for src in message["sources"]:
                        st.markdown(f"* 📄 **{src['file']}** *(Pages: {src['pages']})*")
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")


# ==============================================================================
# Step 4: User Query Processing & Execution Loop
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

    # Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        audio_bytes = None
        extracted_sources = []

        with st.spinner("Agent is reasoning and executing tools..."):
            try:
                # Invoke LangGraph ReAct Agent and obtain sources from metadata
                full_response, extracted_sources = agent_runner.process_query(
                    user_input=augmented_prompt,
                    chat_history=st.session_state.messages,
                    max_history_turns=10,
                )
                response_placeholder.markdown(full_response)

                # Render Citations in an Expander if sources are present
                if extracted_sources:
                    with st.expander("📚 Referenced Course Materials", expanded=False):
                        for src in extracted_sources:
                            st.markdown(f"* 📄 **{src['file']}** *(Pages: {src['pages']})*")

                # Synthesize TTS Audio strictly for the main textual answer
                if enable_tts and full_response:
                    try:
                        audio_bytes = tts_engine.synthesize(full_response)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay_audio)
                    except Exception as tts_err:
                        print(f"[LOG] TTS Warning: {tts_err}")

            except Exception as err:
                full_response = f"An error occurred while executing the agent: {err}"
                response_placeholder.error(full_response)
                print(f"[X] Agent Runner Error: {err}")

    # Append Assistant Message to History
    if full_response:
        assistant_entry = {
            "role": "assistant",
            "content": full_response,
            "sources": extracted_sources,
        }
        if audio_bytes:
            assistant_entry["audio"] = audio_bytes
        st.session_state.messages.append(assistant_entry)