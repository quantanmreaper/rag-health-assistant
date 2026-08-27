# RAG Health Assistant for Diabetes & Hypertension

A Retrieval-Augmented Generation (RAG) system that provides evidence-based health information for diabetes and hypertension management. The assistant uses medical guidelines and authoritative documents to answer health-related queries with accurate, contextual responses.

## Features

- **RAG-Powered Responses**: Combines ChromaDB vector search and BM25 keyword matching for accurate information retrieval
- **Medical Knowledge Base**: Pre-loaded with WHO guidelines and clinical practice documents
- **Multiple Interfaces**: Web dashboard and CLI for flexible interaction
- **Conversation Management**: Session-based chat with history tracking
- **LangChain Integration**: Uses Google Gemini for intelligent response generation

## Requirements

- Python 3.12 or higher
- Google API key for Gemini (set as `GOOGLE_API_KEY` environment variable)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rag-health-assistant.git
cd rag-health-assistant
```

Install dependencies using `uv`:

```bash
# Install uv if not already installed
# Visit: https://docs.astral.sh/uv/getting-started/installation/

# Sync dependencies
uv sync

# For development dependencies
uv sync --group dev
```

## Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Initial Setup

On first run, the system will automatically index documents from the `raw_documents/` directory. To manually trigger indexing:

```bash
uv run python -m rag_health_assistant.main --index
```

### Web Dashboard (Default)

Launch the web interface:

```bash
uv run python -m rag_health_assistant.main
```

Access the dashboard at `http://localhost:8000`

### CLI Mode

For terminal-based interaction:

```bash
uv run python -m rag_health_assistant.main --cli
```

### Custom Server Configuration

```bash
uv run python -m rag_health_assistant.main --host 0.0.0.0 --port 8080
```

## Project Structure

```
src/rag_health_assistant/
├── agent/          # LangChain agent and tools
├── chatbot/        # Core chat logic and profile management
├── ingestion/      # Document processing and indexing
├── retrieval/      # Hybrid search (ChromaDB + BM25)
├── tools/          # Utility functions
├── ui/             # Web and CLI interfaces
└── main.py         # Application entry point
```

## Data

- **raw_documents/**: Place medical PDFs and guideline documents here
- **data/chroma_db/**: Vector database storage (auto-generated)
- **data/conversations/**: Chat history (auto-generated)
- **data/profiles/**: User session profiles (auto-generated)

## Development

Run tests:

```bash
uv run pytest
```

## License

This project is provided for educational and informational purposes. Medical information should be verified with qualified healthcare professionals.

## Disclaimer

This tool is designed to provide general health information based on medical guidelines. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.
