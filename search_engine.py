from pathlib import Path
from typing import Any, Dict, List
import chromadb
import fitz  # PyMuPDF

from api_client import APIClient


class CourseSearchEngine:
    """
    Search engine that ingests course PDF documents into ChromaDB
    using vector embeddings generated via APIClient for semantic retrieval.
    """

    def __init__(self, resources_dir: Path, db_path: str = "./chroma_db"):
        self.resources_dir = resources_dir
        self.db_path = db_path
        self.collection_name = "course_knowledge_base"
        self.api_client = APIClient()

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Ingest PDF documents if the collection is empty
        if self.collection.count() == 0:
            print("[LOG] ChromaDB collection is empty. Starting PDF ingestion...")
            self._ingest_pdfs()
        else:
            print(f"[LOG] Loaded existing ChromaDB collection with {self.collection.count()} items.")

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

    def _ingest_pdfs(self):
        """Reads all PDFs, creates embeddings in batches, and indexes them into ChromaDB."""
        if not self.resources_dir.exists():
            print(f"[!] Warning: Directory {self.resources_dir} does not exist.")
            return

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for pdf_path in sorted(self.resources_dir.rglob("*.pdf")):
            try:
                rel_path = pdf_path.relative_to(self.resources_dir)
                category = self._determine_category(rel_path)

                with fitz.open(pdf_path) as doc:
                    for page_idx, page in enumerate(doc, start=1):
                        text = page.get_text().strip()
                        if not text or len(text) < 20:
                            continue

                        chunk_id = f"{rel_path.stem}_p{page_idx}"
                        metadata = {
                            "category": category,
                            "source_file": rel_path.name,
                            "rel_path": str(rel_path),
                            "page_number": page_idx,
                        }

                        documents.append(text)
                        metadatas.append(metadata)
                        ids.append(chunk_id)
            except Exception as e:
                print(f"[X] Error reading {pdf_path}: {e}")

        if documents:
            print(f"[LOG] Generating embeddings and inserting {len(documents)} chunks into ChromaDB...")
            batch_size = 16
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_metas = metadatas[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                batch_embeddings = self.api_client.create_embeddings_batch(batch_docs)
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
            print(f"[LOG] Successfully indexed {len(documents)} chunk(s) into ChromaDB.")

    @property
    def chunks(self) -> List[Any]:
        """Provides backward compatibility for UI chunk count display."""
        return [0] * self.collection.count()

    def search(
        self, query: str, category: str = "all", top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using ChromaDB semantic vector search.
        Supports category filtering.
        """
        if self.collection.count() == 0 or not query.strip():
            return []

        query_vector = self.api_client.create_embedding(query)

        where_filter = None
        if category and category.lower() != "all":
            where_filter = {"category": category.lower()}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
        )

        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            for doc, meta in zip(docs, metas):
                formatted_results.append(
                    {
                        "source_file": meta.get("source_file", ""),
                        "page_number": meta.get("page_number", 1),
                        "category": meta.get("category", "general"),
                        "content": doc,
                    }
                )

        return formatted_results

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into clean context string for the LLM."""
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