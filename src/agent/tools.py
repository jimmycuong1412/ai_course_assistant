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
        doc_code: Optional[str] = None,
    ) -> str:
        """
        Search and retrieve factual information from internal course materials, including
        Assignments (requirements, tasks), Workshops, and Guidelines/How-to documents.

        Args:
            query (str): The search keywords or semantic question IN ENGLISH (e.g. 'summary objectives deliverables', 'vector database configuration') even if user query is in Vietnamese.
            category (str, optional): Scope filter. Allowed values: 'all', 'assignments', 'workshops', 'guidelines'. Default is 'all'.
            doc_code (str, optional): Target specific assignment/workshop code if identified in user prompt (e.g., 'assignment_10', 'assignment_03', 'workshop_04'). Leave as None if general.

        Returns:
            str: Formatted context blocks containing relevant document excerpts.
        """
        print(f"\n   [Tool Call] 'search_course_knowledge' | Query: '{query}' | Category: '{category}' | Doc Code: '{doc_code}'")

        results = vector_store.search(
            query=query,
            category=category or "all",
            doc_code=doc_code,
            top_k=5,
        )

        formatted_context = vector_store.format_search_results(results)

        print(f"   [Tool Result] Retrieved {len(results)} chunk(s). Sources: {[d.metadata.get('source_file') for d in results]}")
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

    os.environ["TAVILY_API_KEY"] = tavily_api_key

    return TavilySearch(
        max_results=3,
        topic="general",
        description=(
            "A real-time search engine scoped to this AI course's domain. Use ONLY for looking up "
            "latest technical/AI information, library breaking changes, external API documentation, "
            "or AI research news that is not found in the internal course materials. "
            "Do NOT use for topics unrelated to the course (e.g. cooking, sports, general trivia)."
        ),
    )


def get_agent_tools(vector_store: CourseVectorStore) -> List[BaseTool]:
    """
    Aggregates all enabled tools for the ReAct Agent.
    """
    tools = []
    if getattr(vector_store, "rag_enabled", True):
        tools.append(create_course_search_tool(vector_store))
    else:
        print("[!] Warning: RAG disabled (no embedding access). 'search_course_knowledge' tool not registered.")

    tavily_tool = create_tavily_search_tool()

    if tavily_tool:
        tools.append(tavily_tool)

    return tools