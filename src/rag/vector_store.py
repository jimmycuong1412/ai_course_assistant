"""
vector_store.py - Pinecone Vector Store manager with built-in Pinecone Inference Reranking.
Handles Serverless index lifecycle, embedding generation via custom endpoints,
document ingestion, metadata-filtered similarity search, and default Two-Stage Re-ranking.
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
    Manages Pinecone Serverless Index, LangChain VectorStore integration, and Pinecone Inference Reranking.
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
        self.dimension = 1536

        # Reranker model configuration for Pinecone Inference API
        self.rerank_model_name = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")

        # Track documents retrieved in the most recent search
        self.last_retrieved_docs: List[Document] = []

        # Initialize Embeddings model
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

        # RAG requires embedding access for both ingestion and querying; probe once upfront
        # so a restricted key degrades to chat-only instead of crashing on first use.
        self.rag_enabled = self._check_embeddings_available()

        if self.rag_enabled:
            self._sync_documents_if_needed()
        else:
            print(f"[!] Embedding model '{self.embedding_model}' is not accessible with the configured key. RAG document search disabled.\n")

    def _check_embeddings_available(self) -> bool:
        try:
            self.embeddings.embed_query("healthcheck")
            return True
        except Exception:
            return False

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
                self.vector_store.add_documents(documents=chunks)
                print(f"[✔] Successfully ingested {len(chunks)} chunks into Pinecone!\n")
        else:
            print(f"[✔] Pinecone contains {total_vectors} existing vectors. Ready for queries.\n")

    def search(
        self,
        query: str,
        category: str = "all",
        doc_code: Optional[str] = None,
        top_k: int = 5,
        fetch_k: int = 15,
    ) -> List[Document]:
        """
        Performs Two-Stage Retrieval:
        Stage 1: Retrieve wide candidate pool (fetch_k=15) via vector similarity.
        Stage 2: Re-rank candidates down to top_k using Pinecone Inference API (bge-reranker-v2-m3).
        """
        if not query.strip() or not self.rag_enabled:
            self.last_retrieved_docs = []
            return []

        # Stage 1: Retrieval (Bi-Encoder)
        search_kwargs: Dict[str, Any] = {"k": max(fetch_k, top_k)}
        filter_dict: Dict[str, Any] = {}

        if category and category.lower() != "all":
            filter_dict["category"] = category.lower()

        if doc_code and doc_code.lower() not in ["all", "none"]:
            clean_code = doc_code.lower().strip().replace(" ", "_").replace("-", "_")
            filter_dict["doc_code"] = clean_code

        if filter_dict:
            search_kwargs["filter"] = filter_dict

        candidate_docs = self.vector_store.similarity_search(query=query, **search_kwargs)

        if not candidate_docs and filter_dict:
            print(f"   [VectorStore Fallback] Zero matches for filter {filter_dict}. Retrying search without filter...")
            candidate_docs = self.vector_store.similarity_search(query=query, k=max(fetch_k, top_k))

        # Stage 2: Two-Stage Re-ranking via Pinecone Inference API
        if len(candidate_docs) > top_k:
            print(f"   [Pinecone Inference Rerank] Re-ranking {len(candidate_docs)} candidates down to Top {top_k} with '{self.rerank_model_name}'...")
            try:
                documents_payload = [{"text": doc.page_content} for doc in candidate_docs]

                rerank_response = self.pc.inference.rerank(
                    model=self.rerank_model_name,
                    query=query,
                    documents=documents_payload,
                    top_n=top_k,
                    return_documents=False,
                )

                reranked_docs = []
                for item in rerank_response.data:
                    idx = item.index
                    original_doc = candidate_docs[idx]
                    original_doc.metadata["rerank_score"] = round(float(item.score), 4)
                    reranked_docs.append(original_doc)

                results = reranked_docs
            except Exception as e:
                print(f"   [!] Pinecone Inference Rerank error: {e}. Falling back to standard top_k.")
                results = candidate_docs[:top_k]
        else:
            results = candidate_docs[:top_k]

        self.last_retrieved_docs = results
        return results

    def format_search_results(self, docs: List[Document]) -> str:
        """
        Formats retrieved LangChain Document objects into structured context for the agent.
        """
        if not docs:
            return "No relevant course documents found matching the query."

        formatted_blocks = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata
            score_info = f" | Rerank Score: {meta.get('rerank_score')}" if "rerank_score" in meta else ""
            block = (
                f"--- DOCUMENT {idx}{score_info} ---\n"
                f"File: {meta.get('source_file', 'Unknown')} (Page {meta.get('page_number', 'N/A')})\n"
                f"Category: {meta.get('category', 'GENERAL').upper()} | Code: {meta.get('doc_code', 'N/A').upper()}\n"
                f"Content:\n{doc.page_content}\n"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)