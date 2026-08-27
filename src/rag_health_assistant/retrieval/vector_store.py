"""
Vector Store and Embedding management using ChromaDB and Google Gemini Embeddings.
"""

import os
from typing import Optional, List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from ..config import (
    CHROMA_PERSIST_DIR,
    GEMINI_API_KEY,
    DEFAULT_EMBEDDING_MODEL,
    USE_GOOGLE_EMBEDDINGS,
)


class LocalFallbackEmbeddings(Embeddings):
    """
    Fallback embedding using Chroma's built-in ONNX miniLM model when Gemini API key is not present.
    """
    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self.ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # DefaultEmbeddingFunction handles batches
        return [list(map(float, vec)) for vec in self.ef(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [float(x) for x in self.ef([text])[0]]


def get_embeddings(api_key: Optional[str] = None) -> Embeddings:
    """
    Returns embeddings for Chroma.

    Local ONNX MiniLM is the default (stable with existing indexes).
    Set USE_GOOGLE_EMBEDDINGS=true to use Gemini embeddings (re-index required).
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY

    if key and USE_GOOGLE_EMBEDDINGS:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model = DEFAULT_EMBEDDING_MODEL
            if model.startswith("models/"):
                model = model[len("models/") :]
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=key,
            )
        except Exception as e:
            print(
                f"Warning: Could not initialize GoogleGenerativeAIEmbeddings ({e}). "
                "Using local embeddings."
            )

    return LocalFallbackEmbeddings()


def get_vector_store(
    persist_directory: str = str(CHROMA_PERSIST_DIR),
    collection_name: str = "diabetes_hypertension_guidelines",
    api_key: Optional[str] = None
) -> Chroma:
    """
    Initializes and returns a persistent Chroma vector store.
    """
    embeddings = get_embeddings(api_key=api_key)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory)
    )
    return vector_store
