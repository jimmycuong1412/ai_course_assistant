"""
agent_runner.py - Orchestrates the LangGraph ReAct Agent pipeline with ChatOpenAI,
custom endpoints, tools (Internal RAG + Tavily Search), and Chain-of-Thought prompting.
"""

import os
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from prompts import SYSTEM_INSTRUCTION
from tools import get_agent_tools
from vector_store import CourseVectorStore


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

    def process_query(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        max_history_turns: int = 10,
    ) -> str:
        """
        Executes the ReAct agent loop over the input message along with a bounded
        sliding window of recent conversation history to conserve token budget.
        """
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
        final_message = response["messages"][-1]
        return (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )