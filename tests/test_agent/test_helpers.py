"""
test_helpers.py - Unit tests for src/agent/helpers.py

Covers build_export_markdown():
- Empty messages → header only, no Turn sections
- Single user + assistant turn renders correctly
- Turn counter increments per user message only
- Sources block rendered when sources present
- No sources block when sources list is empty
- Messages with empty/None/whitespace content are skipped
- Messages missing role key are skipped
- Audio bytes are NOT serialized into the output
- Multiple sources per assistant message
- Separator count matches expected structure
"""

import pytest
from src.agent.helpers import build_export_markdown


class TestBuildExportMarkdown:

    def test_empty_messages_produces_header_only(self):
        result = build_export_markdown([])
        assert "# 🎓 AI Course Assistant — Chat Export" in result
        assert "**Exported:**" in result
        assert "## Turn" not in result

    def test_single_turn_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "What is LangGraph?"},
            {"role": "assistant", "content": "LangGraph is a framework for stateful agents."},
        ]
        result = build_export_markdown(messages)
        assert "## Turn 1" in result
        assert "**You:** What is LangGraph?" in result
        assert "LangGraph is a framework for stateful agents." in result

    def test_turn_counter_increments_per_user_message(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = build_export_markdown(messages)
        assert "## Turn 1" in result
        assert "## Turn 2" in result
        assert "## Turn 3" not in result

    def test_sources_included_when_present(self):
        messages = [
            {"role": "user", "content": "Explain assignment 10"},
            {
                "role": "assistant",
                "content": "Assignment 10 covers Pinecone.",
                "sources": [{"file": "Assignment_10.pdf", "pages": "1, 3"}],
            },
        ]
        result = build_export_markdown(messages)
        assert "📚 **Sources:**" in result
        assert "Assignment_10.pdf" in result
        assert "Pages: 1, 3" in result

    def test_no_sources_section_when_empty(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "sources": []},
        ]
        result = build_export_markdown(messages)
        assert "📚 **Sources:**" not in result

    def test_skips_messages_with_none_content(self):
        messages = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "valid question"},
            {"role": "assistant", "content": "valid answer"},
        ]
        result = build_export_markdown(messages)
        assert "## Turn 1" in result
        assert "## Turn 2" not in result

    def test_skips_messages_with_empty_string_content(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "real"},
            {"role": "assistant", "content": "reply"},
        ]
        result = build_export_markdown(messages)
        assert "## Turn 1" in result
        assert "## Turn 2" not in result

    def test_skips_messages_with_whitespace_only_content(self):
        messages = [
            {"role": "user", "content": "   \n  "},
            {"role": "user", "content": "real"},
            {"role": "assistant", "content": "reply"},
        ]
        result = build_export_markdown(messages)
        assert "## Turn 1" in result
        assert "## Turn 2" not in result

    def test_skips_messages_missing_role(self):
        messages = [
            {"content": "ghost message with no role"},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
        ]
        result = build_export_markdown(messages)
        assert "ghost message" not in result
        assert "real question" in result

    def test_audio_bytes_not_in_output(self):
        messages = [
            {"role": "user", "content": "Tell me something"},
            {"role": "assistant", "content": "Sure!", "audio": b"\xff\xfb\x90\x00"},
        ]
        result = build_export_markdown(messages)
        assert "Sure!" in result
        assert "b'\\xff" not in result
        assert "\xff" not in result

    def test_multiple_sources_per_response(self):
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "answer",
                "sources": [
                    {"file": "A.pdf", "pages": "1"},
                    {"file": "B.pdf", "pages": "2, 4"},
                ],
            },
        ]
        result = build_export_markdown(messages)
        assert "A.pdf" in result
        assert "B.pdf" in result
        assert "Pages: 2, 4" in result

    def test_separator_count(self):
        """Header divider + one divider per assistant message."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = build_export_markdown(messages)
        # 1 header "---" + 2 turn-closing "---" = at least 3
        assert result.count("---") >= 3
