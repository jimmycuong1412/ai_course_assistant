import os
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

COURSE_DIR = Path(__file__).parent / "AI Application Engineer Level 1"

st.set_page_config(page_title="AI Course Assistant", page_icon="🎓")
st.title("🎓 AI Application Engineer Course Assistant")


def extract_pdf_text(file) -> str:
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


@st.cache_data(show_spinner="Reading course materials...")
def load_course_materials(course_dir: str) -> dict:
    sections = []
    for pdf_path in sorted(Path(course_dir).rglob("*.pdf")):
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        sections.append(f"=== {pdf_path.relative_to(course_dir)} ===\n{text}")
    return {"text": "\n\n".join(sections), "count": len(sections)}


course_materials = (
    load_course_materials(str(COURSE_DIR))
    if COURSE_DIR.exists()
    else {"text": "", "count": 0}
)

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

    st.header("Course Materials")
    if course_materials["count"]:
        st.caption(
            f"Loaded {course_materials['count']} PDFs from "
            f"'{COURSE_DIR.name}' ({len(course_materials['text'])} chars)"
        )
    else:
        st.caption(f"No PDFs found in '{COURSE_DIR.name}'")

    st.header("Extra PDF Context")
    pdf_file = st.file_uploader("Upload another PDF", type="pdf")
    if pdf_file is not None:
        if st.session_state.get("pdf_name") != pdf_file.name:
            st.session_state.pdf_text = extract_pdf_text(pdf_file)
            st.session_state.pdf_name = pdf_file.name
        st.caption(f"Loaded: {st.session_state.pdf_name} "
                   f"({len(st.session_state.pdf_text)} chars)")
    elif st.session_state.get("pdf_name"):
        if st.button("Remove PDF"):
            st.session_state.pdf_text = None
            st.session_state.pdf_name = None
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    if not (azure_endpoint and api_key and model_name):
        st.error("Please fill in Endpoint, API Key, and Model name in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Connection: close forces a fresh TCP connection per request. The STU AI
    # Portal gateway mismatches request/response pairs on reused/pipelined
    # connections (especially under concurrent load), which otherwise makes
    # the assistant echo back a previous, unrelated reply.
    client = OpenAI(
        base_url=azure_endpoint,
        api_key=api_key,
        default_headers={"Connection": "close"},
    )

    context_text = course_materials["text"]
    if st.session_state.get("pdf_text"):
        context_text += "\n\n" + st.session_state.pdf_text

    system_prompt = f"""You are a helpful assistant for an AI course.

--- COURSE MATERIALS ---
{context_text if context_text else "No course materials provided."}
--- END COURSE MATERIALS ---

STRICT INSTRUCTIONS:
1. Answer questions based ONLY on the information present in the course materials above.
2. If the user's question is unrelated to the course materials or cannot be answered using them, respond politely with:
   "I'm sorry, but that question is not related to the course materials. Please ask a question related to the course."
3. Do not make up answers, use external knowledge, or provide information outside of the course materials.
"""

    request_messages = [
        {"role": "system", "content": system_prompt}
    ] + st.session_state.messages

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=request_messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response)
        except Exception as exc:
            full_response = f"Error calling the API: {exc}"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
