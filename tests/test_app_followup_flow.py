"""
test_app_followup_flow.py - End-to-end Streamlit UI tests for the follow-up
suggestion flow, driven headlessly with streamlit.testing AppTest.
"""

import sys
import types
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.agent.followup_generator import FollowUpResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(PROJECT_ROOT / "src" / "app.py")
CHAT_STORE_PATH = PROJECT_ROOT / ".cache" / "chat_sessions.json"

FIRST_ANSWER = "Assignment 10 yêu cầu bạn dựng Pinecone serverless index."
FIRST_SOURCES = [
    {"file": "Assignment_10.pdf", "pages": "1", "code": "assignment_10", "category": "assignments"}
]
FIRST_FOLLOWUPS = ["Cấu hình Pinecone index thế nào?", "Assignment 10 cần nộp file gì?"]


def suggestion_labels(app_test: AppTest) -> list:
    """Labels of the suggestion chips rendered above the chat input."""
    return [b.label for b in app_test.button if b.key and b.key.startswith("suggest_")]


def tiny_png() -> bytes:
    """A real PNG - Streamlit decodes uploaded images before rendering the preview."""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "black").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def app() -> AppTest:
    """
    Boots app.py with every external engine mocked, so the tests exercise only the
    Streamlit rendering and click-dispatch logic.
    """
    # app.py memoizes its engines and its resource scan, and both outlive a single
    # AppTest instance - without clearing, mocks leak between tests.
    st.cache_resource.clear()
    st.cache_data.clear()

    # The app reopens the most recent on-disk session at startup, so a leftover store
    # would seed the next test with the previous test's conversation.
    if CHAT_STORE_PATH.exists():
        CHAT_STORE_PATH.unlink()

    # STTEngine imports torch/transformers/soundfile. Stub the whole module so the
    # suite does not need the speech stack installed just to render the page.
    stt_stub = types.ModuleType("src.engines.stt_engine")
    stt_stub.STTEngine = MagicMock
    sys.modules["src.engines.stt_engine"] = stt_stub

    vector_store = MagicMock()
    vector_store.index_name = "course-knowledge-index"

    agent_runner = MagicMock()
    agent_runner.stream_query.return_value = iter([
        {"type": "action", "tool": "search_course_knowledge", "args": {"query": "assignment 10"}},
        {"type": "observation", "tool": "search_course_knowledge", "preview": "DOCUMENT 1 ..."},
        {"type": "final_answer", "content": FIRST_ANSWER, "sources": FIRST_SOURCES},
    ])

    followup_generator = MagicMock()
    followup_generator.generate.return_value = FollowUpResult(questions=list(FIRST_FOLLOWUPS))

    with patch("src.rag.vector_store.CourseVectorStore", return_value=vector_store), \
         patch("src.agent.agent_runner.CourseAgentRunner", return_value=agent_runner), \
         patch("src.engines.vision_engine.VisionEngine", return_value=MagicMock()), \
         patch("src.engines.tts_engine.TTSEngine", return_value=MagicMock()), \
         patch("src.agent.followup_generator.FollowUpGenerator", return_value=followup_generator):

        app_test = AppTest.from_file(APP_PATH, default_timeout=60)
        app_test.agent_runner = agent_runner
        app_test.followup_generator = followup_generator
        yield app_test

    sys.modules.pop("src.engines.stt_engine", None)
    if CHAT_STORE_PATH.exists():
        CHAT_STORE_PATH.unlink()


class TestFollowUpFlow:

    def test_starter_chips_render_on_an_empty_chat(self, app: AppTest):
        """Cold start is owned by app.py's resource-derived starter suggestions."""
        result = app.run()

        assert not result.exception
        assert suggestion_labels(result)
        assert any("Try asking:" in c.value for c in result.caption)

    def test_clicking_a_starter_chip_runs_a_full_turn(self, app: AppTest):
        result = app.run()
        first_chip = next(b for b in result.button if b.key == "suggest_0")
        label = first_chip.label
        result = first_chip.click().run()

        assert not result.exception
        messages = result.session_state["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == label
        assert messages[1]["content"] == FIRST_ANSWER

    def test_suggestions_are_generated_and_persisted_with_the_turn(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        message = result.session_state["messages"][1]
        assert message["follow_ups"] == FIRST_FOLLOWUPS
        assert message["follow_ups_on_topic"] is True

        kwargs = app.followup_generator.generate.call_args.kwargs
        assert kwargs["assistant_answer"] == FIRST_ANSWER
        assert kwargs["sources"] == FIRST_SOURCES
        assert kwargs["used_web_search"] is False

    def test_followups_replace_the_starter_chips_after_an_answer(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        assert suggestion_labels(result) == FIRST_FOLLOWUPS
        assert any("Continue with:" in c.value for c in result.caption)

    def test_clicking_a_followup_chip_dispatches_the_next_turn(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": "Bạn tạo index bằng ServerlessSpec.", "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(questions=["Còn gì nữa?"])
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        assert not result.exception
        messages = result.session_state["messages"]
        assert len(messages) == 4
        assert messages[2]["content"] == FIRST_FOLLOWUPS[0]

    def test_agent_failure_skips_suggestion_generation(self, app: AppTest):
        app.agent_runner.stream_query.side_effect = RuntimeError("gateway down")

        result = app.run()
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        assert not result.exception
        app.followup_generator.generate.assert_not_called()
        assert result.session_state["messages"][1]["follow_ups"] == []


class TestOffTopicRedirect:

    def test_off_topic_turn_renders_redirect_chips_and_caption(self, app: AppTest):
        redirect = ["What does Assignment 10 require?", "Which workshop covers RAG?"]
        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer",
             "content": "I can only help with the AI Application Engineer course.",
             "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(
            questions=redirect, is_on_topic=False
        )

        result = app.run()
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        assert not result.exception
        assert result.session_state["messages"][1]["follow_ups_on_topic"] is False
        assert suggestion_labels(result) == redirect

        captions = [c.value for c in result.caption]
        assert any("outside this course" in text for text in captions)
        assert not any("Continue with:" in text for text in captions)

    def test_redirect_chips_dispatch_a_course_question(self, app: AppTest):
        """The redirect must actually put the student back on track when clicked."""
        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": "I only cover this course.", "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(
            questions=["What does Assignment 10 require?"], is_on_topic=False
        )

        result = app.run()
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": FIRST_ANSWER, "sources": FIRST_SOURCES},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(questions=FIRST_FOLLOWUPS)
        result = next(b for b in result.button if b.key == "suggest_0").click().run()

        assert not result.exception
        messages = result.session_state["messages"]
        assert messages[2]["content"] == "What does Assignment 10 require?"
        assert messages[3]["follow_ups_on_topic"] is True


class TestUploadedImageDoesNotLoop:
    """
    The turn handler ends with st.rerun(), and a file_uploader keeps returning the same
    file on every rerun. Without an already-answered guard that combination drives the
    agent in an unbounded loop (observed: 261 invocations inside a single run).
    """

    def test_an_uploaded_image_produces_exactly_one_turn(self, app: AppTest):
        result = app.run()
        result.sidebar.file_uploader[0].set_value(("error.png", tiny_png(), "image/png"))
        result = result.run()

        assert not result.exception
        assert len(result.session_state["messages"]) == 2
        assert app.agent_runner.stream_query.call_count == 1

    def test_the_image_does_not_retrigger_on_later_reruns(self, app: AppTest):
        result = app.run()
        result.sidebar.file_uploader[0].set_value(("error.png", tiny_png(), "image/png"))
        result = result.run()

        turns = len(result.session_state["messages"])
        result = result.run()
        result = result.run()

        assert not result.exception
        assert len(result.session_state["messages"]) == turns
        assert app.agent_runner.stream_query.call_count == 1
