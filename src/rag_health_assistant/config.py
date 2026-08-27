"""
Central configuration for AuraHealth RAG assistant and patient chatbot.

All chatbot parameters can be overridden via environment variables.
Invalid values log a warning and fall back to documented defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "raw_documents"
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Google Gemini API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
# gemini-2.5-flash is retired for new AI Studio keys — default to gemini-3.6-flash
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Prefer local Chroma/ONNX embeddings by default so an existing index keeps working.
# Set USE_GOOGLE_EMBEDDINGS=true (and re-run --index) to use Gemini embeddings.
USE_GOOGLE_EMBEDDINGS = os.getenv("USE_GOOGLE_EMBEDDINGS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Map retired / renamed model IDs so older UI settings still work
_LLM_MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-3.6-flash",
    "models/gemini-2.5-flash": "gemini-3.6-flash",
    "gemini-2.0-flash": "gemini-3.6-flash",
    "gemini-1.5-pro": "gemini-3.6-flash",
    "gemini-1.5-flash": "gemini-3.6-flash",
}


def resolve_llm_model(model_name: Optional[str] = None) -> str:
    """Normalize model name and rewrite retired Gemini IDs."""
    name = (model_name or DEFAULT_LLM_MODEL or "gemini-3.6-flash").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return _LLM_MODEL_ALIASES.get(name, name)


# Fallback local embedding model if needed
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K_RETRIEVAL = 5
HYBRID_ALPHA = 0.5  # Weight balance between Dense Vector (alpha) and BM25 (1 - alpha)

# Server Configuration
SERVER_HOST = os.getenv("HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("PORT", 8000))


# ---------------------------------------------------------------------------
# Chatbot configuration (Requirement 13)
# ---------------------------------------------------------------------------

_DEFAULT_CONVERSATIONS_DIR = DATA_DIR / "conversations"
_DEFAULT_PROFILES_DIR = DATA_DIR / "profiles"

# Storage paths — override with CONVERSATIONS_DIR / PROFILES_DIR env vars
_conv_env = os.getenv("CONVERSATIONS_DIR", "").strip()
_prof_env = os.getenv("PROFILES_DIR", "").strip()
CONVERSATIONS_DIR = Path(_conv_env) if _conv_env else _DEFAULT_CONVERSATIONS_DIR
PROFILES_DIR = Path(_prof_env) if _prof_env else _DEFAULT_PROFILES_DIR


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an integer env var; warn and use default on invalid values."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %s",
            name,
            raw,
            default,
        )
        return default
    if minimum is not None and value < minimum:
        logger.warning(
            "Invalid %s=%s (must be >= %s); using default %s",
            name,
            value,
            minimum,
            default,
        )
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; warn and use default on invalid values."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r; using default %s", name, raw, default)
    return default


# Conversation window: last N messages AND within M days (intersection)
CONVERSATION_WINDOW_SIZE = _env_int("CONVERSATION_WINDOW_SIZE", 20, minimum=1)
CONVERSATION_WINDOW_DAYS = _env_int("CONVERSATION_WINDOW_DAYS", 7, minimum=1)

# Session timeout for anonymous/authenticated sessions
SESSION_TIMEOUT_HOURS = _env_int("SESSION_TIMEOUT_HOURS", 24, minimum=1)

# Authentication modes
ANONYMOUS_MODE_ENABLED = _env_bool("ANONYMOUS_MODE_ENABLED", True)
AUTHENTICATED_MODE_ENABLED = _env_bool("AUTHENTICATED_MODE_ENABLED", False)

# PDF export configuration
PDF_PAGE_SIZE = os.getenv("PDF_PAGE_SIZE", "letter").strip().lower() or "letter"
if PDF_PAGE_SIZE not in {"letter", "a4"}:
    logger.warning("Invalid PDF_PAGE_SIZE=%r; using default 'letter'", PDF_PAGE_SIZE)
    PDF_PAGE_SIZE = "letter"
PDF_INCLUDE_PROFILE = _env_bool("PDF_INCLUDE_PROFILE", True)
PDF_INCLUDE_METADATA = _env_bool("PDF_INCLUDE_METADATA", True)

# Performance / caching
PROFILE_CACHE_SIZE = _env_int("PROFILE_CACHE_SIZE", 128, minimum=1)
CONVERSATION_LIST_LIMIT = _env_int("CONVERSATION_LIST_LIMIT", 50, minimum=1)

# Create chatbot storage directories on import
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def validate_chatbot_config() -> dict:
    """
    Validate chatbot configuration at startup.

    Returns a dict of resolved settings. Logs warnings for invalid values
    (already applied via helpers above) and ensures storage dirs exist.
    """
    global ANONYMOUS_MODE_ENABLED

    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    anon_enabled = ANONYMOUS_MODE_ENABLED
    auth_enabled = AUTHENTICATED_MODE_ENABLED
    if not anon_enabled and not auth_enabled:
        logger.warning(
            "Both ANONYMOUS_MODE_ENABLED and AUTHENTICATED_MODE_ENABLED are false; "
            "enabling anonymous mode as fallback."
        )
        ANONYMOUS_MODE_ENABLED = True
        anon_enabled = True

    return {
        "CONVERSATIONS_DIR": str(CONVERSATIONS_DIR),
        "PROFILES_DIR": str(PROFILES_DIR),
        "CONVERSATION_WINDOW_SIZE": CONVERSATION_WINDOW_SIZE,
        "CONVERSATION_WINDOW_DAYS": CONVERSATION_WINDOW_DAYS,
        "SESSION_TIMEOUT_HOURS": SESSION_TIMEOUT_HOURS,
        "ANONYMOUS_MODE_ENABLED": anon_enabled,
        "AUTHENTICATED_MODE_ENABLED": auth_enabled,
        "PDF_PAGE_SIZE": PDF_PAGE_SIZE,
        "PDF_INCLUDE_PROFILE": PDF_INCLUDE_PROFILE,
        "PDF_INCLUDE_METADATA": PDF_INCLUDE_METADATA,
        "PROFILE_CACHE_SIZE": PROFILE_CACHE_SIZE,
        "CONVERSATION_LIST_LIMIT": CONVERSATION_LIST_LIMIT,
    }
