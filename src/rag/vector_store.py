"""
vector_store.py - Pinecone Vector Store manager integrated with LangChain.
Handles Serverless index lifecycle, embedding generation via custom endpoints,
document ingestion, and metadata-filtered similarity search.
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.rag.document_processor import CourseDocumentProcessor


class CourseVectorStore:
    """
    Manages Pinecone Serverless Index and LangChain PineconeVectorStore integration.
    """

    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir

        # Load environment variables
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "course-knowledge-index")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self.dimension = 1536  # Dimension for text-embedding-3-small

        # Initialize Embeddings model trỏ về custom endpoint
        self.embeddings = OpenAIEmbeddings(
            base_url=self.openai_endpoint,
            api_key=self.openai_api_key,
            model=self.embedding_model,
        )

        # Initialize Pinecone client & ensure index existence
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self._ensure_index_exists()

        # Connect PineconeVectorStore
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.pinecone_api_key,
        )

        # Ingest documents if index is fresh or empty
        self._sync_documents_if_needed()

    def _ensure_index_exists(self) -> None:
        """
        Creates serverless Pinecone index if it doesn't already exist.
        """
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"[+] Creating Pinecone Serverless index '{self.index_name}' (dim={self.dimension})...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

            # Wait until index is ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            print(f"[✔] Pinecone index '{self.index_name}' is ready.\n")
        else:
            print(f"[✔] Connected to existing Pinecone index '{self.index_name}'.\n")

    def _sync_documents_if_needed(self) -> None:
        """
        Ingests and indexes document chunks into Pinecone if current vector count is 0.
        """
        index_stats = self.pc.Index(self.index_name).describe_index_stats()
        total_vectors = index_stats.get("total_vector_count", 0)

        if total_vectors == 0:
            print("[+] Pinecone index is empty. Starting document ingestion pipeline...")
            processor = CourseDocumentProcessor(self.resources_dir)
            chunks: List[Document] = processor.load_and_split_documents()

            if chunks:
                print(f"[+] Upserting {len(chunks)} chunks into Pinecone...")
                # Ingest documents in batches using LangChain vectorstore
                self.vector_store.add_documents(documents=chunks)
                print(f"[✔] Successfully ingested {len(chunks)} chunks into Pinecone!\n")
        else:
            print(f"[✔] Pinecone contains {total_vectors} existing vectors. Ready for queries.\n")

    def search(
        self, query: str, category: str = "all", top_k: int = 5
    ) -> List[Document]:
        """
        Performs semantic similarity search with optional metadata category filtering.
        """
        if not query.strip():
            return []

        search_kwargs: Dict[str, Any] = {"k": top_k}

        # Apply metadata filter if specific category is requested
        if category and category.lower() != "all":
            search_kwargs["filter"] = {"category": category.lower()}

        return self.vector_store.similarity_search(query=query, **search_kwargs)

    def format_search_results(self, docs: List[Document]) -> str:
        """
        Formats retrieved LangChain Document objects into structured context for the agent.
        """
        if not docs:
            return "No relevant course documents found matching the query."

        formatted_blocks = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata
            block = (
                f"--- DOCUMENT {idx} ---\n"
                f"File: {meta.get('source_file', 'Unknown')} (Page {meta.get('page_number', 'N/A')})\n"
                f"Category: {meta.get('category', 'GENERAL').upper()}\n"
                f"Content:\n{doc.page_content}\n"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)