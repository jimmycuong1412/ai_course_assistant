"""
test_app_followup_flow.py - End-to-end Streamlit UI tests for the follow-up
suggestion flow, driven headlessly with streamlit.testing AppTest.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.agent.followup_generator import STARTER_QUESTIONS, FollowUpResult

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")

FIRST_ANSWER = "Assignment 10 yêu cầu bạn dựng Pinecone serverless index."
FIRST_SOURCES = [
    {"file": "Assignment_10.pdf", "pages": "1", "code": "assignment_10", "category": "assignments"}
]
FIRST_FOLLOWUPS = ["Cấu hình Pinecone index thế nào?", "Assignment 10 cần nộp file gì?"]


def chip_keys(app_test: AppTest, prefix: str) -> list:
    """Widget keys of rendered suggestion chips (the Clear History button has no key)."""
    return [btn.key for btn in app_test.button if btn.key and btn.key.startswith(prefix)]


def tiny_png() -> bytes:
    """A real PNG - Streamlit decodes uploaded images before rendering the preview."""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "black").save(buffer, format="PNG")
    return buffer.getvalue()


def audio_players(app_test: AppTest) -> list:
    """st.audio has no AppTest accessor, so read it off the element tree directly."""
    return list(app_test.get("audio"))


@pytest.fixture
def app() -> AppTest:
    """
    Boots app.py with every external engine mocked, so the test exercises only the
    Streamlit rendering and click-dispatch logic.
    """
    # app.py memoizes its engines with @st.cache_resource, which outlives a single
    # AppTest instance and would leak mocks between tests.
    st.cache_resource.clear()

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

    tts_engine = MagicMock()
    tts_engine.synthesize.return_value = b"\x00fake-mp3"

    with patch("src.rag.vector_store.CourseVectorStore", return_value=vector_store), \
         patch("src.agent.agent_runner.CourseAgentRunner", return_value=agent_runner), \
         patch("src.engines.vision_engine.VisionEngine", return_value=MagicMock()), \
         patch("src.engines.tts_engine.TTSEngine", return_value=tts_engine), \
         patch("src.agent.followup_generator.FollowUpGenerator", return_value=followup_generator):

        app_test = AppTest.from_file(APP_PATH, default_timeout=60)
        app_test.agent_runner = agent_runner
        app_test.followup_generator = followup_generator
        app_test.tts_engine = tts_engine
        yield app_test


class TestFollowUpFlow:

    def test_starter_questions_render_on_a_new_chat(self, app: AppTest):
        result = app.run()

        assert not result.exception
        labels = [btn.label for btn in result.button]
        for question in STARTER_QUESTIONS:
            assert question in labels

    def test_clicking_a_starter_question_runs_a_full_turn(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert not result.exception
        messages = result.session_state["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == STARTER_QUESTIONS[0]
        assert messages[1]["content"] == FIRST_ANSWER

    def test_suggestions_are_generated_and_persisted_with_the_turn(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert result.session_state["messages"][1]["followups"] == FIRST_FOLLOWUPS
        assert result.session_state["messages"][1]["followups_on_topic"] is True

        kwargs = app.followup_generator.generate.call_args.kwargs
        assert kwargs["user_question"] == STARTER_QUESTIONS[0]
        assert kwargs["assistant_answer"] == FIRST_ANSWER
        assert kwargs["sources"] == FIRST_SOURCES
        assert kwargs["used_web_search"] is False

    def test_clicking_a_followup_chip_dispatches_the_next_turn(self, app: AppTest):
        """
        Regression guard: the chip rendered during a turn must carry the same widget key
        the history loop assigns afterwards, otherwise the click is dropped on rerun.
        """
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        chips = chip_keys(result, "followup_hist_1")
        assert chips == ["followup_hist_1_0", "followup_hist_1_1"]

        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": "Bạn tạo index bằng ServerlessSpec.", "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(questions=["Còn bước nào nữa không?"])

        result = next(b for b in result.button if b.key == "followup_hist_1_0").click().run()

        assert not result.exception
        messages = result.session_state["messages"]
        assert len(messages) == 4
        assert messages[2]["content"] == FIRST_FOLLOWUPS[0]

    def test_only_the_newest_answer_offers_suggestions(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": "Bạn tạo index bằng ServerlessSpec.", "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(questions=["Còn bước nào nữa không?"])
        result = next(b for b in result.button if b.key == "followup_hist_1_0").click().run()

        # The superseded answer at index 1 must no longer render clickable chips
        assert chip_keys(result, "followup_hist") == ["followup_hist_3_0"]

    def test_starter_questions_disappear_once_the_chat_has_history(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert chip_keys(result, "starter") == []

    def test_off_topic_turn_renders_redirect_chips_and_caption(self, app: AppTest):
        redirect = ["Assignment 10 yêu cầu làm gì?", "Workshop nào dạy về RAG?"]
        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer",
             "content": "Mình chỉ hỗ trợ nội dung khóa AI Application Engineer.",
             "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(
            questions=redirect, is_on_topic=False
        )

        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert not result.exception
        assert result.session_state["messages"][1]["followups"] == redirect
        assert result.session_state["messages"][1]["followups_on_topic"] is False
        assert chip_keys(result, "followup_hist_1") == ["followup_hist_1_0", "followup_hist_1_1"]

        captions = [c.value for c in result.caption]
        assert any("outside this course" in text for text in captions)
        assert not any("Suggested follow-up questions" in text for text in captions)

    def test_redirect_chips_dispatch_a_course_question(self, app: AppTest):
        """The redirect must actually put the student back on track when clicked."""
        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": "Mình chỉ hỗ trợ nội dung khóa học.", "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(
            questions=["Assignment 10 yêu cầu làm gì?"], is_on_topic=False
        )

        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": FIRST_ANSWER, "sources": FIRST_SOURCES},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(questions=FIRST_FOLLOWUPS)
        result = next(b for b in result.button if b.key == "followup_hist_1_0").click().run()

        assert not result.exception
        messages = result.session_state["messages"]
        assert messages[2]["content"] == "Assignment 10 yêu cầu làm gì?"
        assert messages[3]["followups_on_topic"] is True

    def test_agent_failure_skips_suggestion_generation(self, app: AppTest):
        app.agent_runner.stream_query.side_effect = RuntimeError("gateway down")

        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert not result.exception
        app.followup_generator.generate.assert_not_called()
        assert chip_keys(result, "followup_hist") == []


class TestSingleRenderPath:
    """
    Streamlit replaces elements by position within a container, so a turn rendered both
    inline and from history leaves stale widgets mounted (the reported symptom was two
    audio players on one answer). These tests pin the invariant that a finished turn is
    rendered by exactly one path - the history loop.
    """

    def test_finished_turn_is_owned_by_the_history_renderer(self, app: AppTest):
        """
        The inline path builds an st.status scaffold. If any of it survives on the finished
        page, the answer is still being drawn by the inline path and will drift out of
        position against the history render on the next rerun.
        """
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert not result.exception
        assert list(result.get("status")) == []

    def test_one_audio_player_per_answered_turn(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()
        assert len(result.get("audio")) == 1

        app.agent_runner.stream_query.return_value = iter([
            {"type": "final_answer", "content": "Bạn tạo index bằng ServerlessSpec.", "sources": []},
        ])
        app.followup_generator.generate.return_value = FollowUpResult(questions=["Còn gì nữa?"])
        result = next(b for b in result.button if b.key == "followup_hist_1_0").click().run()

        assert not result.exception
        assert len(result.get("audio")) == 2
        assert app.tts_engine.synthesize.call_count == 2

    def test_audio_autoplays_once_and_not_on_later_reruns(self, app: AppTest):
        result = app.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert result.get("audio")[0].proto.autoplay is True
        assert "autoplay" not in result.session_state["messages"][1]

        # Any later rerun (clicking a chip, toggling the sidebar) must not replay it
        result = result.run()
        assert result.get("audio")[0].proto.autoplay is False

    def test_autoplay_respects_the_sidebar_toggle(self, app: AppTest):
        result = app.run()
        result.sidebar.toggle[1].set_value(False)
        result = result.run()
        result = next(b for b in result.button if b.key == "starter_0").click().run()

        assert result.get("audio")[0].proto.autoplay is False

    def test_a_lingering_uploaded_image_does_not_retrigger_a_turn(self, app: AppTest):
        """
        A file_uploader keeps returning the same file on every rerun, so without an
        already-answered guard the end-of-turn rerun would generate turns forever.
        """
        result = app.run()
        result.sidebar.file_uploader[0].set_value(("error.png", tiny_png(), "image/png"))
        result = result.run()

        assert not result.exception
        turns = len(result.session_state["messages"])
        assert turns == 2, "the uploaded image should produce exactly one turn"

        result = result.run()
        assert not result.exception
        assert len(result.session_state["messages"]) == turns
