"""
Tests for Hybrid Vector + BM25 Retriever on medical guidelines.
"""

from rag_health_assistant.retrieval.hybrid_retriever import get_hybrid_retriever


def test_hybrid_retriever():
    retriever = get_hybrid_retriever()
    
    # Query 1: Diabetes management
    docs_diabetes = retriever.get_relevant_documents("target blood sugar levels in diabetes", k=3)
    assert len(docs_diabetes) > 0
    print(f"Retrieved {len(docs_diabetes)} docs for diabetes query.")
    print("Top doc source:", docs_diabetes[0].metadata.get("title"))

    # Query 2: Hypertension pharmacological guidelines
    docs_htn = retriever.get_relevant_documents("pharmacological treatment threshold for hypertension", k=3)
    assert len(docs_htn) > 0
    print(f"Retrieved {len(docs_htn)} docs for hypertension query.")
    print("Top doc source:", docs_htn[0].metadata.get("title"))


if __name__ == "__main__":
    test_hybrid_retriever()
    print("Hybrid retriever tests passed successfully!")
