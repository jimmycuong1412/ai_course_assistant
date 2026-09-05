"""
test_prompts.py - Unit tests for Prompt Engineering & ChatPromptTemplate.
"""

from langchain_core.prompts import ChatPromptTemplate
from src.agent.prompts import (
    COURSE_ASSISTANT_PROMPT,
    FEW_SHOT_EXAMPLES,
    SYSTEM_INSTRUCTION,
)


class TestAgentPrompts:

    def test_system_instruction_contains_core_guidelines(self):
        # Verify essential CoT and routing instructions are present
        assert "STEP-BY-STEP REASONING (CHAIN-OF-THOUGHT)" in SYSTEM_INSTRUCTION
        assert "search_course_knowledge" in SYSTEM_INSTRUCTION
        assert "tavily_search" in SYSTEM_INSTRUCTION
        assert "doc_code" in SYSTEM_INSTRUCTION
        assert "Language Matching" in SYSTEM_INSTRUCTION

    def test_few_shot_examples_structure(self):
        assert len(FEW_SHOT_EXAMPLES) >= 6
        for ex in FEW_SHOT_EXAMPLES:
            assert "input" in ex and len(ex["input"]) > 5
            assert "output" in ex and len(ex["output"]) > 10

    def test_prompt_template_formatting(self):
        # Format the master prompt template with sample variables
        formatted = COURSE_ASSISTANT_PROMPT.format_messages(
            chat_history=[],
            input="How to setup Pinecone in Assignment 10?",
            agent_scratchpad=[],
        )

        assert len(formatted) > 0
        # First message must be the system instruction
        assert SYSTEM_INSTRUCTION in formatted[0].content
        # Last human message must be the user input
        assert formatted[-1].content == "How to setup Pinecone in Assignment 10?"