"""
document_processor.py - Document loading and semantic text splitting pipeline.
Uses LangChain PyPDFLoader and RecursiveCharacterTextSplitter with enriched metadata.
"""

from pathlib import Path
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CourseDocumentProcessor:
    """
    Handles PDF ingestion, categorization, semantic chunking, and metadata enrichment.
    """

    def __init__(
        self,
        resources_dir: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.resources_dir = resources_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize recursive character text splitter with natural boundary separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _determine_category(self, relative_path: Path) -> str:
        """
        Infers document category based on the relative directory structure.
        """
        path_str = str(relative_path).lower()
        if "assignment" in path_str:
            return "assignments"
        elif "workshop" in path_str:
            return "workshops"
        elif "guideline" in path_str or "guide" in path_str:
            return "guidelines"
        return "general"

    def load_and_split_documents(self) -> List[Document]:
        """
        Traverses resources_dir for all PDF files, loads content page-by-page,
        enriches metadata, and splits documents into optimal chunks for embedding.
        """
        if not self.resources_dir.exists():
            print(f"[!] Warning: Resources directory '{self.resources_dir}' not found.")
            return []

        all_pdf_paths = sorted(self.resources_dir.rglob("*.pdf"))
        if not all_pdf_paths:
            print(f"[!] No PDF files found in '{self.resources_dir}'.")
            return []

        print(f"[+] Found {len(all_pdf_paths)} PDF document(s). Starting extraction...")

        raw_pages: List[Document] = []

        for pdf_path in all_pdf_paths:
            rel_path = pdf_path.relative_to(self.resources_dir)
            category = self._determine_category(rel_path)

            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()

                for page in pages:
                    # Clean extracted page text
                    text = page.page_content.strip()
                    if not text or len(text) < 20:
                        continue

                    # Enrich metadata with explicit categorization and origin tracking
                    page_num = page.metadata.get("page", 0) + 1  # 1-based index
                    page.metadata.update(
                        {
                            "category": category,
                            "source_file": rel_path.name,
                            "rel_path": str(rel_path),
                            "page_number": page_num,
                        }
                    )
                    page.page_content = text
                    raw_pages.append(page)

                print(f"    [✔] Loaded '{rel_path.name}' ({len(pages)} pages)")

            except Exception as exc:
                print(f"    [X] Failed loading '{pdf_path.name}': {exc}")

        # Split loaded pages into smaller semantic chunks
        split_chunks = self.text_splitter.split_documents(raw_pages)

        # Assign unique chunk_id to each chunk
        for idx, chunk in enumerate(split_chunks, start=1):
            source_stem = Path(chunk.metadata.get("source_file", "doc")).stem
            page_num = chunk.metadata.get("page_number", 1)
            chunk.metadata["chunk_id"] = f"{source_stem}_p{page_num}_c{idx}"

        print(f"[✔] Total processed: {len(raw_pages)} pages -> {len(split_chunks)} semantic chunks.\n")
        return split_chunks