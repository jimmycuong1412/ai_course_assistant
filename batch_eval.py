import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from api_client import make_api_call
from search_engine import CourseSearchEngine
from tools import TOOLS_SCHEMA, execute_tool_call
from prompts import SYSTEM_PROMPT

load_dotenv()
RESOURCES_DIR = Path(__file__).parent / "resources"


def process_single_prompt(
    item: dict,
    search_engine: CourseSearchEngine,
    azure_endpoint: str,
    api_key: str,
    model_name: str,
) -> dict:
    """Processes a single test prompt through Turn 1 (Tool Check) and Turn 2 (Final Synthesis)."""
    user_prompt = item.get("prompt", "")
    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result_log = {
        "id": item.get("id"),
        "prompt": user_prompt,
        "tool_called": False,
        "final_answer": "",
        "status": "success",
    }

    try:
        # Step 1: Initial call to check for tool calls
        response_1 = make_api_call(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            model_name=model_name,
            messages=request_messages,
            tools=TOOLS_SCHEMA,
            temperature=0.0,
        )
        message_1 = response_1.choices[0].message

        if message_1.tool_calls:
            result_log["tool_called"] = True
            request_messages.append(message_1)

            # Execute tool calls
            for tool_call in message_1.tool_calls:
                for tool_resp in execute_tool_call(tool_call, search_engine):
                    request_messages.append(tool_resp)

            # Step 2: Final answer synthesis
            response_2 = make_api_call(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                model_name=model_name,
                messages=request_messages,
                stream=False,
                temperature=0.3,
            )
            result_log["final_answer"] = response_2.choices[0].message.content or ""
        else:
            result_log["final_answer"] = message_1.content or ""

    except Exception as exc:
        result_log["status"] = f"failed: {exc}"

    return result_log


def run_batch_eval():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")

    if not (azure_endpoint and api_key and model_name):
        print("[X] Error: Azure OpenAI settings are missing in .env file.")
        return

    search_engine = CourseSearchEngine(RESOURCES_DIR)

    test_cases = [
        {"id": "TC_01", "prompt": "Assignment 4 yêu cầu sử dụng tenacity để làm gì?"},
        {"id": "TC_02", "prompt": "Tóm tắt nội dung yêu cầu bài tập số 5 cho tôi"},
        {"id": "TC_03", "prompt": "Thời tiết hôm nay ở Hà Nội thế nào?"},
        {"id": "TC_04", "prompt": "What are the main objectives of Workshop 2?"},
    ]

    print(f"[+] Running Batch Processing for {len(test_cases)} cases...\n")
    results = []

    for idx, item in enumerate(test_cases, start=1):
        print(f"[{idx}/{len(test_cases)}] Executing {item['id']}...")
        res = process_single_prompt(
            item, search_engine, azure_endpoint, api_key, model_name
        )
        results.append(res)
        print(f"    Status: {res['status']} | Tool Called: {res['tool_called']}")
        time.sleep(1)

    output_file = "batch_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[✔] Batch processing finished! Results saved to '{output_file}'")


if __name__ == "__main__":
    run_batch_eval()