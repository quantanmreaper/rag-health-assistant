"""
Index builder for populating ChromaDB and BM25 indices from raw guideline documents.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from ..config import RAW_DOCS_DIR, CHROMA_PERSIST_DIR, BM25_INDEX_PATH
from .loader import load_raw_documents
from .chunker import chunk_documents
from ..retrieval.vector_store import get_vector_store
from ..retrieval.hybrid_retriever import HybridRetriever


def build_indices(
    docs_dir: Path = RAW_DOCS_DIR,
    api_key: Optional[str] = None,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Loads raw medical documents, chunks them, and builds both ChromaDB vector store
    and BM25 keyword index.
    """
    print("--- Starting Ingestion & Indexing Pipeline ---")
    
    # 1. Load documents
    raw_pages = load_raw_documents(docs_dir=docs_dir)
    if not raw_pages:
        return {
            "status": "error",
            "message": f"No documents found or extracted from {docs_dir}",
            "num_pages": 0,
            "num_chunks": 0
        }

    # 2. Chunk documents
    chunks = chunk_documents(raw_pages)
    if not chunks:
        return {
            "status": "error",
            "message": "Failed to generate chunks from loaded pages.",
            "num_pages": len(raw_pages),
            "num_chunks": 0
        }

    # 3. Build & Populate Chroma Vector Store
    print(f"Indexing {len(chunks)} chunks into ChromaDB at {CHROMA_PERSIST_DIR}...")
    vector_store = get_vector_store(api_key=api_key)
    
    # Ingest in batches
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        # IDs for idempotency
        ids = [doc.metadata.get("chunk_id", f"chunk_{i + idx}") for idx, doc in enumerate(batch)]
        vector_store.add_documents(documents=batch, ids=ids)
        print(f"  Ingested batch {i + len(batch)}/{total_chunks} into ChromaDB")

    # 4. Build & Save BM25 Index
    print("Building BM25 keyword index...")
    retriever = HybridRetriever(vector_store=vector_store, api_key=api_key)
    retriever.save_bm25(chunks)

    print("--- Indexing Complete! ---")
    return {
        "status": "success",
        "num_pages": len(raw_pages),
        "num_chunks": len(chunks),
        "documents": list({doc.metadata.get("source") for doc in raw_pages})
    }


def get_indexed_stats(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns statistics about currently indexed documents and chunks in Chroma and BM25.
    """
    try:
        vector_store = get_vector_store(api_key=api_key)
        # Chroma collection count
        count = vector_store._collection.count()
        bm25_exists = BM25_INDEX_PATH.exists()

        raw_files = list(RAW_DOCS_DIR.glob("*.pdf")) if RAW_DOCS_DIR.exists() else []

        return {
            "is_indexed": count > 0,
            "total_chunks": count,
            "bm25_ready": bm25_exists,
            "raw_document_count": len(raw_files),
            "raw_documents": [f.name for f in raw_files]
        }
    except Exception as e:
        return {
            "is_indexed": False,
            "total_chunks": 0,
            "bm25_ready": False,
            "error": str(e)
        }
