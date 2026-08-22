"""
vision_engine.py - Multimodal Vision analyzer for screenshots, code errors, and diagrams.
Uses LangChain ChatOpenAI with structured output extraction.
"""

import base64
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class VisionAnalysisResponse(BaseModel):
    summary: str = Field(
        description="Concise description of what is shown in the image (e.g. error trace, architecture diagram, code snippet)."
    )
    extracted_text_or_error: str = Field(
        description="Key text, exact error message, traceback, or technical details extracted from the image."
    )
    detected_issue: Optional[str] = Field(
        default=None,
        description="Main problem, bug, or question inferred from the image content."
    )


class VisionEngine:
    """
    Handles multimodal image ingestion and analysis using OpenAI Vision capabilities.
    """

    def __init__(self):
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.vision_model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

        self.llm = ChatOpenAI(
            base_url=self.openai_endpoint,
            api_key=self.openai_api_key,
            model=self.vision_model,
            temperature=0.0,
        )
        self.structured_llm = self.llm.with_structured_output(VisionAnalysisResponse)

    def analyze_image_bytes(self, image_bytes: bytes, user_note: str = "") -> str:
        """
        Encodes raw image bytes to base64 and invokes multimodal LLM to extract structured context.
        """
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt_text = (
            "Analyze this uploaded image in the context of an AI Engineering course. "
            "Extract any relevant error message, code snippet, configuration, or diagram elements."
        )
        if user_note:
            prompt_text += f"\nUser's question/note regarding the image: {user_note}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multimodal technical visual analyst. Inspect the image precisely, "
                    "extract any visible error tracebacks, code blocks, or diagram structures, and summarize them clearly."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            },
        ]

        try:
            analysis: VisionAnalysisResponse = self.structured_llm.invoke(messages)
            formatted_context = (
                f"[Uploaded Image Analysis]\n"
                f"- Visual Summary: {analysis.summary}\n"
                f"- Extracted Text/Error: {analysis.extracted_text_or_error}\n"
            )
            if analysis.detected_issue:
                formatted_context += f"- Inferred Issue: {analysis.detected_issue}\n"

            return formatted_context
        except Exception as e:
            print(f"[X] Vision Analysis Error: {e}")
            return f"[Uploaded Image Analysis Failed]: {e}"