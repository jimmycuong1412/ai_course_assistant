"""
tools.py - LangChain Tools definition for the AI Course Assistant Agent.
Wraps vector search and Tavily web search into standard LangChain tools.
"""

import os
from typing import List, Optional
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from src.rag.vector_store import CourseVectorStore


def create_course_search_tool(vector_store: CourseVectorStore) -> BaseTool:
    """
    Factory function creating the search_course_knowledge tool bound to the vector store instance.
    """

    @tool
    def search_course_knowledge(
        query: str,
        category: Optional[str] = "all",
    ) -> str:
        """
        Search and retrieve factual information from internal course materials, including
        Assignments (requirements, tasks), Workshops, and Guidelines/How-to documents.

        Args:
            query (str): Detailed search keywords or user query phrase.
            category (str, optional): Scope filter. Allowed values: 'all', 'assignments', 'workshops', 'guidelines'. Default is 'all'.

        Returns:
            str: Formatted context blocks containing relevant document excerpts.
        """
        print(f"\n   [Tool Call] 'search_course_knowledge' | Query: '{query}' | Category: '{category}'")

        results = vector_store.search(query=query, category=category or "all", top_k=4)
        formatted_context = vector_store.format_search_results(results)

        print(f"   [Tool Result] Retrieved {len(results)} chunk(s) from Pinecone Vector Store.")
        return formatted_context

    return search_course_knowledge


def create_tavily_search_tool() -> Optional[BaseTool]:
    """
    Initializes and configures the Tavily Web Search tool if API key is present.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("[!] Warning: TAVILY_API_KEY not configured. Web search tool disabled.")
        return None

    # Ensure environment variable is set for the wrapper
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    return TavilySearch(
        max_results=3,
        topic="general",
        description=(
            "A real-time search engine. Useful for looking up latest technical information, "
            "library breaking changes, external API documentation, latest AI research, or current events "
            "that are not found in the internal course materials."
        ),
    )


def get_agent_tools(vector_store: CourseVectorStore) -> List[BaseTool]:
    """
    Aggregates all enabled tools for the ReAct Agent.
    """
    tools = [create_course_search_tool(vector_store)]
    tavily_tool = create_tavily_search_tool()

    if tavily_tool:
        tools.append(tavily_tool)

    return tools