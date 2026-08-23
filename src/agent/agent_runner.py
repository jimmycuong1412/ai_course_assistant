"""
agent_runner.py - Orchestrates the LangGraph ReAct Agent pipeline with ChatOpenAI,
custom endpoints, tools (Internal RAG + Tavily Search), and direct metadata source extraction.
"""

import os
from typing import Any, Dict, List, Tuple
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import SYSTEM_INSTRUCTION
from src.agent.tools import get_agent_tools
from src.rag.vector_store import CourseVectorStore


class CourseAgentRunner:
    """
    Encapsulates the LangGraph ReAct agent pipeline for processing user queries.
    """

    def __init__(self, vector_store: CourseVectorStore):
        self.vector_store = vector_store

        # Setup model configuration from environment
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        # Initialize ChatOpenAI client
        self.llm = ChatOpenAI(
            base_url=self.openai_endpoint,
            api_key=self.openai_api_key,
            model=self.chat_model,
            temperature=0.1,
        )

        # Initialize full suite of tools (RAG + Tavily)
        self.tools = get_agent_tools(self.vector_store)

        # Initialize ReAct Agent with LangGraph
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_INSTRUCTION,
        )

    def _extract_sources_from_metadata(self) -> List[Dict[str, str]]:
        """
        Extracts and aggregates source files and page numbers directly from
        the vector store's retrieved Document metadata objects.
        """
        docs = getattr(self.vector_store, "last_retrieved_docs", [])
        if not docs:
            return []

        sources_map: Dict[str, set] = {}
        for doc in docs:
            meta = doc.metadata
            file_name = meta.get("source_file", "Unknown")
            page_val = meta.get("page_number", 1)
            try:
                page_str = str(int(float(page_val)))
            except (ValueError, TypeError):
                page_str = str(page_val)

            if file_name not in sources_map:
                sources_map[file_name] = set()
            sources_map[file_name].add(page_str)

        structured_sources: List[Dict[str, str]] = []
        for file_name, pages in sources_map.items():
            # Sort page numbers numerically if possible
            try:
                sorted_pages = sorted(list(pages), key=lambda x: int(x))
            except ValueError:
                sorted_pages = sorted(list(pages))

            structured_sources.append({
                "file": file_name,
                "pages": ", ".join(sorted_pages),
            })

        return structured_sources

    def process_query(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        max_history_turns: int = 10,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Executes the ReAct agent loop over the input message along with a bounded
        sliding window of recent conversation history.
        Returns:
            Tuple[str, List[Dict[str, str]]]: (final_answer_text, extracted_sources_metadata)
        """
        # Reset tracked documents before execution
        self.vector_store.last_retrieved_docs = []

        messages: List[BaseMessage] = []

        # Step 1: Filter valid conversation turns
        valid_history = [
            msg
            for msg in chat_history
            if msg.get("role") in ["user", "assistant"] and msg.get("content")
        ]

        # Step 2: Apply sliding window to keep only the most recent N turns
        recent_history = valid_history[-max_history_turns:]

        # Step 3: Map dictionary entries to LangChain BaseMessage instances
        for msg in recent_history:
            role = msg.get("role")
            content = str(msg.get("content", ""))

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Step 4: Append current user query
        messages.append(HumanMessage(content=user_input))

        # Step 5: Invoke LangGraph ReAct agent pipeline
        response = self.agent_executor.invoke({"messages": messages})
        all_messages = response.get("messages", [])

        final_message = all_messages[-1]
        final_text = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        # Retrieve source metadata directly from document objects
        sources = self._extract_sources_from_metadata()
        return final_text, sources