"""
document_processor.py - Document loading and semantic text splitting pipeline.
Uses PyMuPDF (fitz) for high-performance extraction, optional Vision OCR for screenshots,
and RecursiveCharacterTextSplitter with post-split enriched metadata.
"""

import re
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.engines.vision_engine import VisionEngine


class CourseDocumentProcessor:
    """
    Handles PDF ingestion, categorization, multimodal screenshot OCR, semantic chunking, and metadata enrichment.
    """

    def __init__(
        self,
        resources_dir: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        enable_image_analysis: bool = True,
    ):
        self.resources_dir = resources_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_image_analysis = enable_image_analysis
        self.vision_engine = VisionEngine() if enable_image_analysis else None

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
        Ensures guidelines are prioritized before assignments to prevent misclassification.
        """
        path_str = str(relative_path).lower()
        if "guideline" in path_str or "guide" in path_str:
            return "guidelines"
        elif "workshop" in path_str:
            return "workshops"
        elif "assignment" in path_str:
            return "assignments"
        return "general"

    def _extract_doc_code(self, filename: str) -> str:
        """
        Extracts standardized identifier codes (e.g., 'assignment_10', 'workshop_04').
        """
        fn = filename.lower()
        match_asg = re.search(r"assignment\s*0?(\d+)", fn)
        if match_asg:
            return f"assignment_{int(match_asg.group(1)):02d}"
        match_ws = re.search(r"workshop\s*0?(\d+)", fn)
        if match_ws:
            return f"workshop_{int(match_ws.group(1)):02d}"
        return "other"

    def load_and_split_documents(self) -> List[Document]:
        """
        Traverses resources_dir for all PDF files, loads text and extracts screenshot descriptions,
        splits documents into semantic chunks, and attaches contextual headers post-splitting.
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
            doc_code = self._extract_doc_code(rel_path.name)

            try:
                doc = fitz.open(pdf_path)
                for page_idx, page in enumerate(doc):
                    page_num = page_idx + 1
                    page_text = page.get_text("text").strip()

                    # Extract screenshots from Guidelines & Workshops if image analysis is enabled
                    image_descriptions: List[str] = []
                    if self.enable_image_analysis and self.vision_engine:
                        image_list = page.get_images(full=True)
                        for img_info in image_list:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            width, height = base_image.get("width", 0), base_image.get("height", 0)

                            # Skip small icons, badges, and decorative graphics (< 150px)
                            if width > 150 and height > 150:
                                img_bytes = base_image["image"]
                                desc = self.vision_engine.describe_document_image(img_bytes)
                                if desc:
                                    image_descriptions.append(f"[Visual Screenshot Note: {desc}]")

                    combined_text = page_text
                    if image_descriptions:
                        combined_text += "\n\n" + "\n".join(image_descriptions)

                    if not combined_text or len(combined_text) < 20:
                        continue

                    raw_pages.append(
                        Document(
                            page_content=combined_text,
                            metadata={
                                "category": category,
                                "doc_code": doc_code,
                                "source_file": rel_path.name,
                                "rel_path": str(rel_path),
                                "page_number": page_num,
                            },
                        )
                    )

                print(f"    [✔] Loaded '{rel_path.name}' ({len(doc)} pages)")
                doc.close()

            except Exception as exc:
                print(f"    [X] Failed loading '{pdf_path.name}': {exc}")

        # 1. Split actual document content first to prevent header detachment
        split_chunks = self.text_splitter.split_documents(raw_pages)

        # 2. Attach context headers and metadata post-splitting
        valid_chunks: List[Document] = []
        for idx, chunk in enumerate(split_chunks, start=1):
            source_file = chunk.metadata.get("source_file", "doc")
            doc_code = chunk.metadata.get("doc_code", "other")
            page_num = chunk.metadata.get("page_number", 1)
            category = chunk.metadata.get("category", "general")

            chunk.metadata["chunk_id"] = f"{Path(source_file).stem}_p{page_num}_c{idx}"

            # Prepend a compact context header directly to chunk content
            chunk.page_content = (
                f"Document: {source_file} | Category: {category.upper()} | Code: {doc_code.upper()}\n"
                f"{chunk.page_content.strip()}"
            )
            valid_chunks.append(chunk)

        print(f"[✔] Total processed: {len(raw_pages)} pages -> {len(valid_chunks)} semantic chunks.\n")
        return valid_chunks