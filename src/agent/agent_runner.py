"""
agent_runner.py - Orchestrates the LangGraph ReAct Agent pipeline with ChatOpenAI,
custom endpoints, tools (Two-Stage RAG + Tavily Search), metadata source extraction,
and real-time event streaming for UI visibility.
"""

import os
from typing import Any, Dict, Generator, List
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
            try:
                sorted_pages = sorted(list(pages), key=lambda x: int(x))
            except ValueError:
                sorted_pages = sorted(list(pages))

            structured_sources.append({
                "file": file_name,
                "pages": ", ".join(sorted_pages),
            })

        return structured_sources

    def _prepare_messages(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        max_history_turns: int = 10,
    ) -> List[BaseMessage]:
        """
        Helper method to map raw conversation history dictionaries to LangChain messages
        and apply the sliding window constraint.
        """
        messages: List[BaseMessage] = []

        # Step 1: Filter valid conversation turns
        valid_history = [
            msg
            for msg in chat_history
            if msg.get("role") in ["user", "assistant"] and msg.get("content")
        ]

        # Step 2: Apply sliding window
        recent_history = valid_history[-max_history_turns:]

        # Step 3: Map dictionary entries to LangChain BaseMessage instances
        for msg in recent_history:
            role = msg.get("role")
            content = str(msg.get("content", ""))

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Step 4: Append current user input
        messages.append(HumanMessage(content=user_input))
        return messages

    def stream_query(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        max_history_turns: int = 10,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams intermediate agent actions, tool executions, and the final response
        in real-time using LangGraph stream updates.

        Yields:
            Dict[str, Any]: Events of type 'action', 'observation', or 'final_answer'
        """
        # Reset tracked vector store documents before new execution
        self.vector_store.last_retrieved_docs = []
        messages = self._prepare_messages(user_input, chat_history, max_history_turns)

        final_text = ""

        # Stream node-level updates from the LangGraph execution
        for event in self.agent_executor.stream({"messages": messages}, stream_mode="updates"):
            for node_name, node_output in event.items():
                node_messages = node_output.get("messages", [])
                if not node_messages:
                    continue

                for msg in node_messages:
                    # Case 1: Agent Node decides to invoke one or more tools (Action)
                    if node_name == "agent" and getattr(msg, "tool_calls", None):
                        for tool_call in msg.tool_calls:
                            yield {
                                "type": "action",
                                "tool": tool_call.get("name", "unknown_tool"),
                                "args": tool_call.get("args", {}),
                            }

                    # Case 2: Tools Node returned execution results (Observation)
                    elif node_name == "tools":
                        tool_name = getattr(msg, "name", "tool")
                        raw_content = str(getattr(msg, "content", ""))

                        if tool_name == "search_course_knowledge":
                            # Extract only the metadata headers and discard chunk content bodies
                            doc_headers = []
                            for chunk_segment in raw_content.split("--- DOCUMENT"):
                                if chunk_segment.strip():
                                    header_part = chunk_segment.split("Content:")[0].strip()
                                    doc_headers.append(f"• DOCUMENT {header_part}")

                            preview = (
                                "\n\n".join(doc_headers)
                                if doc_headers
                                else "Documents retrieved successfully."
                            )
                        else:
                            # Default fallback for other tools (e.g., tavily_search)
                            preview = (
                                raw_content[:200] + "..."
                                if len(raw_content) > 200
                                else raw_content
                            )

                        yield {
                            "type": "observation",
                            "tool": tool_name,
                            "preview": preview,
                        }

                    # Case 3: Agent Node produced the final text response without tool calls
                    elif node_name == "agent" and not getattr(msg, "tool_calls", None):
                        final_text = (
                            msg.content
                            if hasattr(msg, "content")
                            else str(msg)
                        )

        # Extract verified sources after retrieval tool has completed
        sources = self._extract_sources_from_metadata()

        # Yield the final answer payload with aggregated sources
        yield {
            "type": "final_answer",
            "content": final_text,
            "sources": sources,
        }