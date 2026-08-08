import json
from typing import Any, Dict, List
from search_engine import CourseSearchEngine

# OpenAI Function Schema Definition
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_course_knowledge",
            "description": (
                "Search and retrieve information from course materials, including "
                "Assignments, Workshop details, and Guidelines/How-to documents. "
                "Use this tool whenever the user asks about course topics, assignment requirements, "
                "or workshop instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or user query phrase.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["all", "assignments", "workshops", "guidelines"],
                        "description": "Category filter for search scope.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def execute_tool_call(
    tool_call: Any, search_engine: CourseSearchEngine
) -> List[Dict[str, Any]]:
    """Executes tool function requested by OpenAI API and logs parameters."""
    function_name = tool_call.function.name
    tool_call_id = tool_call.id
    args = json.loads(tool_call.function.arguments)

    if function_name == "search_course_knowledge":
        query = args.get("query", "")
        category = args.get("category", "all")

        # Concise console log for Tool Parameters
        print(f"[LOG] Tool Call: '{function_name}' | Query: '{query}' | Category: '{category}'")

        raw_results = search_engine.search(query=query, category=category, top_k=4)
        formatted_result = search_engine.format_search_results(raw_results)

        print(f"[LOG] Search Engine: Retrieved {len(raw_results)} chunk(s)")

        return [
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": formatted_result,
            }
        ]

    return [
        {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": function_name,
            "content": f"Error: Tool '{function_name}' is not recognized.",
        }
    ]