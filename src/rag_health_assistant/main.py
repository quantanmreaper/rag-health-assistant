"""
Main entry point for AuraHealth AI - Diabetes & Hypertension RAG Assistant.
"""

import argparse
import sys
import uvicorn
from .config import SERVER_HOST, SERVER_PORT
from .ingestion.indexer import build_indices, get_indexed_stats
from .ui.cli import run_cli


def main():
    parser = argparse.ArgumentParser(
        description="AuraHealth AI - Cardio-Metabolic Health RAG Assistant for Diabetes & Hypertension"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=True,
        help="Launch the modern Web Dashboard (default)"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch interactive terminal CLI instead of Web UI"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run document ingestion and build ChromaDB + BM25 indices"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=SERVER_HOST,
        help=f"Web server host (default: {SERVER_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_PORT,
        help=f"Web server port (default: {SERVER_PORT})"
    )

    args = parser.parse_args()

    # If --index is explicitly requested
    if args.index:
        print("Starting manual indexing of guidelines in raw_documents/...")
        res = build_indices()
        print(f"Indexing complete: {res}")
        return

    # Check if indices exist; if not, build automatically
    stats = get_indexed_stats()
    if not stats.get("is_indexed"):
        print("Knowledge base not yet indexed. Building initial ChromaDB and BM25 index from raw_documents/...")
        build_indices()

    # CLI Mode
    if args.cli:
        run_cli()
        return

    # Web Dashboard (Default)
    print(f"\n==================================================================")
    print(f"  Starting AuraHealth AI Web Dashboard at: http://{args.host}:{args.port}")
    print(f"==================================================================\n")
    uvicorn.run("rag_health_assistant.ui.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
