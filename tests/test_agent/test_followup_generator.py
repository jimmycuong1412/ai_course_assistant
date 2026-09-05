"""
test_followup_generator.py - Unit tests for the follow-up question suggestion engine.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.followup_generator import (
    MAX_SUGGESTIONS,
    REDIRECT_QUESTIONS,
    STARTER_QUESTIONS,
    FollowUpGenerator,
    FollowUpSuggestions,
)


class TestFollowUpGenerator:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Mock ChatOpenAI so no live model call is ever issued."""
        with patch("src.agent.followup_generator.ChatOpenAI") as mock_llm_cls:
            self.mock_structured_llm = MagicMock()
            mock_llm_cls.return_value.with_structured_output.return_value = self.mock_structured_llm
            self.generator = FollowUpGenerator()
            yield

    def _set_response(self, questions, is_on_topic=True):
        self.mock_structured_llm.invoke.return_value = FollowUpSuggestions(
            questions=questions, is_on_topic=is_on_topic
        )

    def test_generates_suggestions_for_a_normal_turn(self):
        self._set_response([
            "Assignment 10 cần nộp những file nào?",
            "Pinecone index cấu hình ra sao?",
        ])

        result = self.generator.generate(
            user_question="Assignment 10 yêu cầu làm gì?",
            assistant_answer="Assignment 10 yêu cầu bạn dựng Pinecone serverless index và tìm kiếm top 3 sản phẩm.",
            sources=[{"file": "Assignment_10.pdf", "pages": "1, 2", "code": "assignment_10", "category": "assignments"}],
        )

        assert result.questions == [
            "Assignment 10 cần nộp những file nào?",
            "Pinecone index cấu hình ra sao?",
        ]
        assert result.is_on_topic is True

    def test_retrieved_source_metadata_is_passed_to_the_model(self):
        """Suggestions must be grounded in the documents the retriever actually returned."""
        self._set_response(["What is the rubric for Assignment 10?"])

        self.generator.generate(
            user_question="What does Assignment 10 require?",
            assistant_answer="Assignment 10 requires a Pinecone serverless index with OpenAI embeddings.",
            sources=[{"file": "Assignment_10.pdf", "pages": "1", "code": "assignment_10", "category": "assignments"}],
        )

        human_prompt = self.mock_structured_llm.invoke.call_args[0][0][1].content
        assert "Assignment_10.pdf" in human_prompt
        assert "assignment_10" in human_prompt
        assert "assignments" in human_prompt

    def test_web_search_context_is_reported_to_the_model(self):
        self._set_response(["Which LangGraph version introduced that change?"])

        self.generator.generate(
            user_question="Any breaking changes in LangGraph?",
            assistant_answer="Yes, the prebuilt agent import path changed in recent releases.",
            sources=[],
            used_web_search=True,
        )

        human_prompt = self.mock_structured_llm.invoke.call_args[0][0][1].content
        assert "live web search" in human_prompt
        assert "No internal course documents" in human_prompt

    def test_language_of_suggestions_follows_the_answer(self):
        """The system prompt must instruct the model to mirror the answer's language."""
        self._set_response(["Workshop 04 dùng những tool nào?"])

        result = self.generator.generate(
            user_question="Workshop 04 nói về gì?",
            assistant_answer="Workshop 04 hướng dẫn xây dựng ReAct Agent với LangGraph.",
            sources=[],
        )

        system_prompt = self.mock_structured_llm.invoke.call_args[0][0][0].content
        assert "Language Matching" in system_prompt
        assert result.questions == ["Workshop 04 dùng những tool nào?"]

    def test_suggestions_are_capped_deduplicated_and_cleaned(self):
        self._set_response([
            "- What is the rubric?",
            '"What is the rubric?"',
            "  How do I submit?  ",
            "Which model should I use?",
            "Is there a deadline extension?",
        ])

        result = self.generator.generate(
            user_question="Tell me about Assignment 10",
            assistant_answer="Assignment 10 covers Pinecone similarity search.",
        )

        assert len(result.questions) == MAX_SUGGESTIONS
        assert result.questions == ["What is the rubric?", "How do I submit?", "Which model should I use?"]

    def test_overly_long_suggestions_are_discarded(self):
        self._set_response(["Short and usable?", "x" * 200])

        result = self.generator.generate(
            user_question="Tell me about Assignment 10",
            assistant_answer="Assignment 10 covers Pinecone similarity search.",
        )

        assert result.questions == ["Short and usable?"]

    def test_empty_answer_skips_the_model_call(self):
        result = self.generator.generate(user_question="Hi", assistant_answer="   ")

        assert result.questions == []
        self.mock_structured_llm.invoke.assert_not_called()

    def test_model_failure_degrades_to_no_suggestions(self):
        """A suggestion failure must never interrupt the chat turn."""
        self.mock_structured_llm.invoke.side_effect = RuntimeError("gateway timeout")

        result = self.generator.generate(
            user_question="What does Assignment 10 require?",
            assistant_answer="Assignment 10 requires a Pinecone serverless index.",
        )

        assert result.questions == []
        assert result.is_on_topic is True

    def test_off_topic_question_yields_course_redirect_suggestions(self):
        """An unrelated question must be steered back toward course material."""
        self._set_response(
            ["Assignment 10 yêu cầu làm gì?", "Workshop 04 dạy những gì?"],
            is_on_topic=False,
        )

        result = self.generator.generate(
            user_question="Cách nấu phở bò ngon nhất là gì?",
            assistant_answer="Mình chỉ hỗ trợ các nội dung của khóa AI Application Engineer.",
            sources=[],
        )

        assert result.is_on_topic is False
        assert result.questions == ["Assignment 10 yêu cầu làm gì?", "Workshop 04 dạy những gì?"]

    def test_off_topic_turn_never_ends_up_without_suggestions(self):
        """The redirect fallback covers a model that classifies off-topic but returns nothing."""
        self._set_response([], is_on_topic=False)

        result = self.generator.generate(
            user_question="What is the best football team in the world right now?",
            assistant_answer="I can only help with the AI Application Engineer course.",
        )

        assert result.is_on_topic is False
        assert result.questions == REDIRECT_QUESTIONS["en"]

    def test_redirect_fallback_matches_the_students_language(self):
        self._set_response([], is_on_topic=False)

        result = self.generator.generate(
            user_question="Bạn có biết đội bóng nào mạnh nhất thế giới hiện nay không?",
            assistant_answer="Mình chỉ hỗ trợ nội dung khóa học.",
        )

        assert result.questions == REDIRECT_QUESTIONS["vi"]

    def test_on_topic_failure_does_not_trigger_the_redirect_fallback(self):
        """An empty on-topic result stays empty - no chips beat irrelevant chips."""
        self._set_response([], is_on_topic=True)

        result = self.generator.generate(
            user_question="What does Assignment 10 require?",
            assistant_answer="Assignment 10 requires a Pinecone serverless index.",
        )

        assert result.questions == []
        assert result.is_on_topic is True

    def test_course_catalog_from_resources_reaches_the_prompt(self, tmp_path):
        """Redirect suggestions must be able to name documents that actually exist."""
        resources = tmp_path / "resources"
        (resources / "Assignments").mkdir(parents=True)
        (resources / "Workshops" / "Workshop 4").mkdir(parents=True)
        (resources / "Guidelines").mkdir(parents=True)
        (resources / "Assignments" / "Assignment 10 - Using Pinecone.pdf").touch()
        (resources / "Workshops" / "Workshop 4" / "Workshop 4_Artifact.pdf").touch()
        (resources / "Guidelines" / "GUIDELINES FOR SUBMITTING.updated 4.pdf").touch()

        with patch("src.agent.followup_generator.ChatOpenAI") as mock_llm_cls:
            structured = MagicMock()
            mock_llm_cls.return_value.with_structured_output.return_value = structured
            generator = FollowUpGenerator(resources_dir=resources)

        structured.invoke.return_value = FollowUpSuggestions(questions=["Anything else?"])
        generator.generate(user_question="Hi there", assistant_answer="Hello.")

        human_prompt = structured.invoke.call_args[0][0][1].content
        assert "Assignment 10 - Using Pinecone" in human_prompt
        assert "Workshop 4 Artifact" in human_prompt
        assert "GUIDELINES FOR SUBMITTING" in human_prompt
        assert ".updated 4" not in human_prompt

    def test_missing_resources_directory_falls_back_to_a_generic_catalog(self, tmp_path):
        with patch("src.agent.followup_generator.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.with_structured_output.return_value = MagicMock()
            generator = FollowUpGenerator(resources_dir=tmp_path / "does_not_exist")

        assert "Assignments 01-14" in generator.course_catalog

    def test_starter_questions_are_available_for_cold_start(self):
        assert 0 < len(STARTER_QUESTIONS) <= MAX_SUGGESTIONS
        assert all(q.strip() for q in STARTER_QUESTIONS)
