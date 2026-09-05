"""
followup_generator.py - Generates contextual follow-up question suggestions after each
assistant answer. Cold-start chips are owned by app.py's resource-derived starters.

Runs as a single lightweight structured-output LLM call outside the ReAct graph so the
suggestions never leak into the answer text that is forwarded to the TTS engine.

When a student drifts outside the course scope, the suggestions switch to redirect mode
and steer them back toward the actual assignments, workshops, and guidelines on file.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langdetect import LangDetectException, detect
from pydantic import BaseModel, Field

# Maximum number of suggestion chips rendered under an answer
MAX_SUGGESTIONS = 3

# Upper bound on a single suggestion so it stays readable inside a Streamlit button
MAX_QUESTION_LENGTH = 90

# Number of catalog entries per category injected into the prompt
MAX_CATALOG_ENTRIES_PER_CATEGORY = 15

# Deterministic safety net used when a student goes off-topic and the model returns
# nothing usable. Keyed by the language detected in the student's question.
REDIRECT_QUESTIONS: Dict[str, List[str]] = {
    "vi": [
        "Assignment mới nhất yêu cầu những gì?",
        "Workshop nào hướng dẫn về RAG và vector database?",
        "Quy định nộp bài trên GitLab như thế nào?",
    ],
    "en": [
        "What does the latest assignment require?",
        "Which workshop covers RAG and vector databases?",
        "How do I submit my assignment on GitLab?",
    ],
}


class FollowUpSuggestions(BaseModel):
    is_on_topic: bool = Field(
        default=True,
        description=(
            "True if the student's question belongs to the course scope (assignments, workshops, "
            "guidelines, or AI/software engineering topics). False for unrelated small talk or "
            "subjects such as cooking, sports, travel, or personal advice."
        ),
    )
    questions: List[str] = Field(
        default_factory=list,
        description=(
            "Two or three short follow-up questions the student would naturally ask next. "
            "Each must be a single self-contained question of at most 12 words."
        ),
    )


class FollowUpResult(BaseModel):
    """Suggestions plus the scope verdict, so the UI can label them appropriately."""

    questions: List[str] = Field(default_factory=list)
    is_on_topic: bool = True


FOLLOWUP_SYSTEM_PROMPT = """You suggest follow-up questions for an AI Teaching Assistant serving the "AI Application Engineer" course.

Given the student's last question and the assistant's answer, classify whether the question was within the course scope, then propose 2-3 questions the student should ask next.

--- SCOPE CLASSIFICATION ---
Set `is_on_topic` to true when the student asks about course assignments, workshops, submission guidelines, or any AI / software engineering topic the assistant can research.
Set `is_on_topic` to false when the question is unrelated to the course - small talk, cooking, sports, travel, health, personal or financial advice, and similar.

--- IF ON TOPIC ---
1. **Language Matching:** Write the questions in the EXACT same language as the assistant's answer (Vietnamese answer -> Vietnamese questions, English answer -> English questions). Never mix languages.
2. **Answerable Scope:** Every question must be answerable by the assistant's tools: internal course materials (assignments, workshops, guidelines) or a live web search for technical topics. Prefer drilling into the documents listed under RETRIEVED SOURCES.
3. **Move Forward:** Never restate or rephrase the question that was just answered, and never repeat one another.
4. **Clarification Turns:** If the assistant asked the student to clarify which assignment or workshop they meant, propose the concrete candidate questions that resolve the ambiguity.

--- IF OFF TOPIC (REDIRECT MODE) ---
1. IGNORE the subject matter of the student's question entirely. Never propose questions that continue the off-topic thread, and never invent a course connection to it.
2. Pick concrete, inviting questions drawn from the COURSE CATALOG below that guide the student back to what this assistant is actually for.
3. Draw from different categories (an assignment, a workshop, a submission guideline) so the student sees the breadth of what they can ask.
4. Match the language of the student's own question.

--- ALWAYS ---
Maximum 12 words per question, phrased as the student speaking to the assistant.
Return questions only, with no numbering, quotes, or extra commentary.
"""


class FollowUpGenerator:
    """
    Produces contextual follow-up question suggestions for the most recent answer,
    redirecting to course material whenever the student drifts off topic.
    """

    def __init__(self, resources_dir: Optional[Path] = None):
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.llm = ChatOpenAI(
            base_url=self.openai_endpoint,
            api_key=self.openai_api_key,
            model=self.chat_model,
            temperature=0.7,
        )
        self.structured_llm = self.llm.with_structured_output(FollowUpSuggestions)

        # Built once at startup so redirect suggestions name documents that actually exist
        self.course_catalog = self._build_course_catalog(resources_dir)

    def _build_course_catalog(self, resources_dir: Optional[Path]) -> str:
        """
        Scans the resources directory for available course documents and renders a compact
        catalog. Purely filesystem-based, so it costs no API call and stays in sync with
        whatever material is present for ingestion.
        """
        if not resources_dir or not Path(resources_dir).exists():
            return "Assignments 01-14, Workshops 01-05, and submission/setup guidelines."

        grouped: Dict[str, List[str]] = {"Assignments": [], "Workshops": [], "Guidelines": []}

        for pdf_path in sorted(Path(resources_dir).rglob("*.pdf")):
            path_str = str(pdf_path).lower()
            if "guideline" in path_str or "guide" in path_str:
                category = "Guidelines"
            elif "workshop" in path_str:
                category = "Workshops"
            elif "assignment" in path_str:
                category = "Assignments"
            else:
                continue

            title = self._clean_document_title(pdf_path.stem)
            if title and title not in grouped[category]:
                grouped[category].append(title)

        lines = []
        for category, titles in grouped.items():
            if titles:
                lines.append(f"{category}:")
                lines.extend(f"  - {t}" for t in titles[:MAX_CATALOG_ENTRIES_PER_CATEGORY])

        return "\n".join(lines) if lines else "Assignments, Workshops, and submission guidelines."

    def _clean_document_title(self, stem: str) -> str:
        """Normalizes a PDF filename into a short, readable catalog entry."""
        title = re.sub(r"\.updated.*$", "", stem, flags=re.IGNORECASE)
        title = re.sub(r"_+", " ", title)
        title = re.sub(r"\s+", " ", title).strip(" -")
        return title[:80]

    def _format_source_context(self, sources: Optional[List[Dict[str, Any]]]) -> str:
        """
        Renders the citation metadata extracted from the retrieval step into a compact
        grounding block. Only metadata is passed - never the chunk bodies - to keep the call cheap.
        """
        if not sources:
            return "No internal course documents were retrieved for this answer."

        lines = []
        for src in sources:
            descriptor = str(src.get("file", "Unknown"))
            code = src.get("code")
            category = src.get("category")
            if code:
                descriptor += f" (Code: {code}"
                descriptor += f" | Category: {category})" if category else ")"
            elif category:
                descriptor += f" (Category: {category})"
            lines.append(f"- {descriptor}")

        return "\n".join(lines)

    def _detect_language(self, text: str) -> str:
        """Resolves the redirect fallback language, defaulting to English."""
        try:
            detected = detect(text)
            if detected in REDIRECT_QUESTIONS:
                return detected
        except LangDetectException:
            pass
        return "en"

    def _sanitize(self, raw_questions: List[str]) -> List[str]:
        """
        Strips formatting artifacts, drops empties and duplicates, and caps the list length.
        """
        cleaned: List[str] = []
        seen = set()

        for question in raw_questions or []:
            text = str(question).strip().strip("-*").strip().strip('"').strip()
            if not text or len(text) > MAX_QUESTION_LENGTH:
                continue

            key = text.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(text)

            if len(cleaned) == MAX_SUGGESTIONS:
                break

        return cleaned

    def _redirect_fallback(self, reference_text: str) -> List[str]:
        """Language-matched course prompts used when the model gives us nothing to show."""
        return list(REDIRECT_QUESTIONS[self._detect_language(reference_text)])

    def generate(
        self,
        user_question: str,
        assistant_answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        used_web_search: bool = False,
    ) -> FollowUpResult:
        """
        Generates follow-up suggestions for a completed question/answer turn.

        Returns:
            FollowUpResult: up to MAX_SUGGESTIONS questions plus the scope verdict. An
            off-topic turn always yields course-oriented redirect questions; a failure
            yields an empty list so the chat is never interrupted.
        """
        if not assistant_answer or not assistant_answer.strip():
            return FollowUpResult(questions=[], is_on_topic=True)

        source_block = self._format_source_context(sources)
        tool_note = (
            "The assistant used live web search for this answer."
            if used_web_search
            else "The assistant answered from course materials or its own knowledge."
        )

        user_prompt = (
            f"STUDENT QUESTION:\n{user_question.strip()}\n\n"
            f"ASSISTANT ANSWER:\n{assistant_answer.strip()}\n\n"
            f"RETRIEVED SOURCES:\n{source_block}\n\n"
            f"CONTEXT: {tool_note}\n\n"
            f"COURSE CATALOG (the material this assistant can actually teach):\n{self.course_catalog}\n\n"
            f"Classify the scope, then propose 2-3 follow-up questions following the rules."
        )

        try:
            response: FollowUpSuggestions = self.structured_llm.invoke(
                [
                    SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            questions = self._sanitize(response.questions)
            is_on_topic = bool(response.is_on_topic)

            # An off-topic turn is exactly where the student needs direction, so never
            # leave it without chips - fall back to curated course prompts.
            if not is_on_topic and not questions:
                questions = self._redirect_fallback(user_question)

            return FollowUpResult(questions=questions, is_on_topic=is_on_topic)
        except Exception as err:
            print(f"[LOG] Follow-up Suggestion Warning: {err}")
            return FollowUpResult(questions=[], is_on_topic=True)
