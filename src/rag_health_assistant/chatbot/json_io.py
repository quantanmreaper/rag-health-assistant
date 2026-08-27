"""
Cross-platform atomic JSON file helpers with exclusive locking.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_path_locks: dict[str, threading.RLock] = {}
_path_locks_guard = threading.Lock()


def _get_path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve()) if path.exists() or path.parent.exists() else str(path)
    with _path_locks_guard:
        if key not in _path_locks:
            _path_locks[key] = threading.RLock()
        return _path_locks[key]


def sanitize_id(value: str) -> str:
    """Strip path separators and unsafe characters from ID components."""
    cleaned = value.replace("..", "").replace("/", "").replace("\\", "").strip()
    # Keep alphanumerics, dash, underscore
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in "-_")


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Validate and atomically write pretty-printed UTF-8 JSON.

    Uses temp-file + os.replace and a per-path lock to avoid corruption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_path_lock(path)

    # Validate serializability before touching disk
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    # Round-trip validate
    json.loads(payload)

    with lock:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
            raise


def read_json(path: Path) -> Optional[Any]:
    """
    Read UTF-8 JSON. Returns None on missing or malformed files (logs error).
    """
    path = Path(path)
    if not path.exists():
        return None
    lock = _get_path_lock(path)
    with lock:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.error("Failed to read JSON from %s: %s", path, exc)
            return None
