import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, RateLimitError

from api_client import make_api_call
from search_engine import CourseSearchEngine
from tools import TOOLS_SCHEMA, execute_tool_call
from prompts import SYSTEM_PROMPT

load_dotenv()

RESOURCES_DIR = Path(__file__).parent / "resources"

st.set_page_config(page_title="AI Course Assistant", page_icon="🎓")
st.title("🎓 AI Application Engineer Course Assistant")


@st.cache_resource(show_spinner="Indexing course materials...")
def get_search_engine(resources_dir: Path) -> CourseSearchEngine:
    return CourseSearchEngine(resources_dir)


search_engine = get_search_engine(RESOURCES_DIR)

# Sidebar setup
with st.sidebar:
    st.header("Azure OpenAI Settings")
    azure_endpoint = st.text_input(
        "Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", "")
    )
    api_key = st.text_input(
        "API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password"
    )
    model_name = st.text_input(
        "Model name", value=os.getenv("AZURE_OPENAI_MODEL", "")
    )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.header("Knowledge Base Status")
    st.caption(
        f"Indexed **{len(search_engine.chunks)}** document chunks from "
        f"`{RESOURCES_DIR.name}`"
    )

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    if message.get("role") in ["user", "assistant"] and message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Input Handling
user_input = st.chat_input("Ask a question about assignments, workshops, or guidelines...")

if user_input:
    if not (azure_endpoint and api_key and model_name):
        st.error("Please fill in Endpoint, API Key, and Model name in the sidebar.")
        st.stop()

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

        try:
            # Step 1: Initial completion call to determine tool usage
            response = make_api_call(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
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
                stream = make_api_call(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
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

        except (RateLimitError, APIConnectionError, APIError) as api_err:
            full_response = f"API Service Error (failed after 5 retries): {api_err}"
            placeholder.error(full_response)
            print(f"[LOG] API Error: {api_err}")
        except Exception as exc:
            full_response = f"An unexpected error occurred: {exc}"
            placeholder.error(full_response)
            print(f"[LOG] Unexpected Error: {exc}")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})