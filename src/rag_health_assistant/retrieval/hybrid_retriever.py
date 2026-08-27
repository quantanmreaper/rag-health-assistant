"""
Hybrid Retriever combining Dense Vector Search (ChromaDB) and Sparse BM25 Keyword Search
with Reciprocal Rank Fusion (RRF).
"""

import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from ..config import (
    BM25_INDEX_PATH,
    TOP_K_RETRIEVAL,
    HYBRID_ALPHA
)
from .vector_store import get_vector_store


class HybridRetriever:
    """
    Combines dense semantic vector retrieval with BM25 lexical search.
    """
    def __init__(
        self,
        vector_store=None,
        bm25_index_path: Path = BM25_INDEX_PATH,
        top_k: int = TOP_K_RETRIEVAL,
        alpha: float = HYBRID_ALPHA,
        api_key: Optional[str] = None
    ):
        self.vector_store = vector_store or get_vector_store(api_key=api_key)
        self.bm25_index_path = Path(bm25_index_path)
        self.top_k = top_k
        self.alpha = alpha  # Weight for dense vector search (0.0 to 1.0)
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_docs: List[Document] = []
        self._load_bm25()

    def _load_bm25(self):
        """Loads serialized BM25 index if available."""
        if self.bm25_index_path.exists():
            try:
                with open(self.bm25_index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data.get("bm25")
                    self.bm25_docs = data.get("docs", [])
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")

    def save_bm25(self, docs: List[Document]):
        """Builds and persists BM25 index for given documents."""
        self.bm25_docs = docs
        tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "docs": self.bm25_docs}, f)
        print(f"Persisted BM25 index with {len(docs)} documents.")

    def _retrieve_dense(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Retrieves top k documents using dense vector similarity."""
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            return results
        except Exception as e:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return [(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]
            except Exception as e2:
                print(f"Dense retrieval failed ({e}); fallback also failed ({e2}). Using BM25 only.")
                return []

    def _retrieve_bm25(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Retrieves top k documents using BM25 keyword matching."""
        if not self.bm25 or not self.bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.bm25_docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def get_relevant_documents(
        self,
        query: str,
        k: Optional[int] = None,
        condition_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Performs hybrid retrieval using Reciprocal Rank Fusion (RRF).
        """
        top_k = k or self.top_k
        fetch_k = top_k * 3

        dense_results = self._retrieve_dense(query, k=fetch_k)
        bm25_results = self._retrieve_bm25(query, k=fetch_k)

        # Reciprocal Rank Fusion (RRF)
        # RRF score = sum(weight / (60 + rank))
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 1. Process Dense Results
        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (self.alpha / (60 + rank + 1))

        # 2. Process BM25 Results
        for rank, (doc, _) in enumerate(bm25_results):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + ((1.0 - self.alpha) / (60 + rank + 1))

        # Sort by RRF score descending
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        final_docs = []
        for did in sorted_doc_ids:
            doc = doc_map[did]
            # Optional filter by condition ("diabetes", "hypertension", "both")
            if condition_filter and condition_filter != "all":
                doc_cond = doc.metadata.get("condition", "both")
                if doc_cond not in [condition_filter, "both", "general_health"]:
                    continue
            
            # Attach retrieval score to metadata
            doc.metadata["retrieval_score"] = round(rrf_scores[did], 4)
            final_docs.append(doc)
            if len(final_docs) >= top_k:
                break

        return final_docs


def get_hybrid_retriever(api_key: Optional[str] = None) -> HybridRetriever:
    return HybridRetriever(api_key=api_key)
