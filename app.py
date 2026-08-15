import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, RateLimitError

from api_client import APIClient
from prompts import SYSTEM_PROMPT
from search_engine import CourseSearchEngine
from tools import TOOLS_SCHEMA, execute_tool_call
from tts import text_to_speech

load_dotenv()

RESOURCES_DIR = Path(__file__).parent / "resources"

st.set_page_config(page_title="AI Course Assistant", page_icon="🎓")
st.title("🎓 AI Application Engineer Course Assistant")


@st.cache_resource(show_spinner="Initializing API Client...")
def get_api_client() -> APIClient:
    return APIClient()


@st.cache_resource(show_spinner="Indexing course materials into ChromaDB...")
def get_search_engine(resources_dir: Path) -> CourseSearchEngine:
    return CourseSearchEngine(resources_dir)


api_client = get_api_client()
search_engine = get_search_engine(RESOURCES_DIR)

# Sidebar setup
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input(
        "Model name", value=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
    )

    st.header("Voice Settings")
    enable_tts = st.toggle("Enable Voice Output (TTS)", value=True)
    autoplay_audio = st.toggle(
        "Auto-play audio", value=True, disabled=not enable_tts
    )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.header("Knowledge Base Status")
    st.caption(
        f"Indexed **{len(search_engine.chunks)}** document chunks in ChromaDB."
    )

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    if message.get("role") in ["user", "assistant"] and message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

# User Input Handling
user_input = st.chat_input("Ask a question about assignments, workshops, or guidelines...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    print(f"\n[LOG] User Input: '{user_input}'")

    # Sanitize message history
    clean_history = [
        {"role": msg["role"], "content": str(msg["content"])}
        for msg in st.session_state.messages
        if msg.get("role") in ["user", "assistant"] and msg.get("content")
    ]

    request_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + clean_history

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        audio_bytes = None

        try:
            # Step 1: Initial completion call to determine tool usage
            response = api_client.make_api_call(
                model_name=model_name,
                messages=request_messages,
                tools=TOOLS_SCHEMA,
                temperature=0.0,
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                request_messages.append(response_message)

                # Execute requested tools
                for tool_call in response_message.tool_calls:
                    try:
                        tool_responses = execute_tool_call(tool_call, search_engine)
                        for tool_resp in tool_responses:
                            request_messages.append(tool_resp)
                    except Exception as tool_exc:
                        print(f"[LOG] Tool Execution Error: {tool_exc}")
                        request_messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": f"Error executing tool: {str(tool_exc)}",
                        })

                # Step 2: Stream final synthesized response
                stream = api_client.make_api_call(
                    model_name=model_name,
                    messages=request_messages,
                    stream=True,
                    temperature=0.3,
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response)
            else:
                print("[LOG] Tool Call: None (Direct Answer)")
                full_response = response_message.content or ""
                placeholder.markdown(full_response)

            # Step 3: Synthesize voice output if enabled (auto-detects English/Vietnamese)
            if enable_tts and full_response:
                with st.spinner("Generating audio..."):
                    try:
                        audio_bytes = text_to_speech(full_response)
                        st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay_audio)
                    except Exception as tts_exc:
                        print(f"[LOG] TTS Error: {tts_exc}")
                        st.warning(f"Could not generate audio: {tts_exc}")

        except (RateLimitError, APIConnectionError, APIError) as api_err:
            full_response = f"API Service Error (failed after 5 retries): {api_err}"
            placeholder.error(full_response)
            print(f"[LOG] API Error: {api_err}")
        except Exception as exc:
            full_response = f"An unexpected error occurred: {exc}"
            placeholder.error(full_response)
            print(f"[LOG] Unexpected Error: {exc}")

    if full_response:
        assistant_entry = {"role": "assistant", "content": full_response}
        if audio_bytes:
            assistant_entry["audio"] = audio_bytes
        st.session_state.messages.append(assistant_entry)