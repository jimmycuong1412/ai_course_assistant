import math
import re
from pathlib import Path
from typing import Any, Dict, List
import fitz  # PyMuPDF
import spacy
from rank_bm25 import BM25Okapi


class CourseSearchEngine:
    """
    Search engine that ingests course PDF documents, applies spaCy lemmatization,
    and uses the rank_bm25 algorithm for high-precision retrieval.
    """

    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.chunks: List[Dict[str, Any]] = []

        # Load spaCy lightweight English model (disabling unused pipeline components for speed)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm"
            )

        self._ingest_pdfs()
        self._build_bm25_index()

    def _determine_category(self, relative_path: Path) -> str:
        """Categorize PDF files based on parent folder names."""
        path_str = str(relative_path).lower()
        if "assignment" in path_str:
            return "assignments"
        elif "workshop" in path_str:
            return "workshops"
        elif "guideline" in path_str or "guide" in path_str:
            return "guidelines"
        return "general"

    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Processes text with spaCy: lowercase, extracts word lemmas,
        and filters out punctuation and stop words.
        """
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if not token.is_stop and token.is_alpha:
                tokens.append(token.lemma_)
            elif token.like_num:
                # Retain numbers (e.g., '4', '05', '5')
                tokens.append(token.text)
        return tokens

    def _ingest_pdfs(self):
        """Reads all PDFs, creates structured text chunks with metadata."""
        if not self.resources_dir.exists():
            print(f"[!] Warning: Directory {self.resources_dir} does not exist.")
            return

        for pdf_path in sorted(self.resources_dir.rglob("*.pdf")):
            try:
                rel_path = pdf_path.relative_to(self.resources_dir)
                category = self._determine_category(rel_path)

                with fitz.open(pdf_path) as doc:
                    for page_idx, page in enumerate(doc, start=1):
                        text = page.get_text().strip()
                        if not text:
                            continue

                        chunk = {
                            "id": f"{rel_path.name}_page_{page_idx}",
                            "category": category,
                            "source_file": rel_path.name,
                            "rel_path": str(rel_path),
                            "page_number": page_idx,
                            "content": text,
                        }
                        self.chunks.append(chunk)
            except Exception as e:
                print(f"[X] Error reading {pdf_path}: {e}")

        print(f"[LOG] Indexed {len(self.chunks)} text chunk(s) from '{self.resources_dir.name}'.")

    def _build_bm25_index(self):
        """Pre-processes all document chunks and builds the BM25 index."""
        if not self.chunks:
            self.bm25 = None
            return

        # Prepare corpus tokenized with spaCy Lemmatization
        self.corpus = [
            self._tokenize_and_lemmatize(f"{chunk['source_file']} {chunk['content']}")
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def search(
        self, query: str, category: str = "all", top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using rank_bm25 and spaCy preprocessing.
        Applies category filtering if specified.
        """
        if not self.bm25 or not self.chunks:
            return []

        # Tokenize and lemmatize user query
        tokenized_query = self._tokenize_and_lemmatize(query)
        if not tokenized_query:
            return []

        # Calculate BM25 scores for all document chunks
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Filter by category and collect scored candidates
        scored_candidates = []
        for idx, score in enumerate(doc_scores):
            chunk = self.chunks[idx]
            if category != "all" and chunk["category"] != category.lower():
                continue

            if score > 0:
                scored_candidates.append((score, chunk))

        # Sort candidates descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_candidates[:top_k]]

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into clean context string for OpenAI LLM."""
        if not results:
            return "No relevant course documents found matching the query."

        formatted_output = []
        for idx, res in enumerate(results, start=1):
            block = (
                f"--- DOCUMENT {idx} ---\n"
                f"File: {res['source_file']} (Page {res['page_number']})\n"
                f"Category: {res['category'].upper()}\n"
                f"Content:\n{res['content']}\n"
            )
            formatted_output.append(block)

        return "\n\n".join(formatted_output)