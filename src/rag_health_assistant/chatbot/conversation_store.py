"""
Persistent JSON conversation storage with sliding-window retrieval.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from ..config import (
    CONVERSATION_LIST_LIMIT,
    CONVERSATION_WINDOW_DAYS,
    CONVERSATION_WINDOW_SIZE,
    CONVERSATIONS_DIR,
)
from .json_io import _get_path_lock, atomic_write_json, read_json, sanitize_id
from .models import Conversation, ConversationMetadata, Message, MessageRole, utc_now

logger = logging.getLogger(__name__)


class ConversationStore:
    """JSON-file conversation persistence under data/conversations/."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or CONVERSATIONS_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _filepath(self, patient_id: str, conversation_id: str) -> Path:
        safe_patient = sanitize_id(patient_id)
        safe_conv = sanitize_id(conversation_id)
        return self.storage_dir / f"{safe_patient}_{safe_conv}.json"

    def create_conversation(
        self,
        patient_id: str,
        is_anonymous: bool = False,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        """Create and persist a new empty conversation."""
        now = utc_now()
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        conversation = Conversation(
            metadata=ConversationMetadata(
                conversation_id=conv_id,
                patient_id=patient_id,
                title=title,
                created_at=now,
                updated_at=now,
                is_anonymous=is_anonymous,
                message_count=0,
            ),
            messages=[],
        )
        self.save_conversation(conversation)
        return conversation

    def save_conversation(self, conversation: Conversation) -> None:
        """Atomically save conversation to JSON file."""
        path = self._filepath(
            conversation.metadata.patient_id,
            conversation.metadata.conversation_id,
        )
        try:
            atomic_write_json(path, conversation.model_dump(mode="json"))
        except OSError as exc:
            logger.error("Failed to save conversation %s: %s", path, exc)
            raise

    def load_conversation(
        self, conversation_id: str, patient_id: str
    ) -> Optional[Conversation]:
        """Load conversation; returns None if missing or malformed."""
        path = self._filepath(patient_id, conversation_id)
        data = read_json(path)
        if data is None:
            return None
        try:
            conversation = Conversation.model_validate(data)
            if conversation.metadata.deleted:
                return None
            return conversation
        except Exception as exc:
            logger.error("Invalid conversation structure in %s: %s", path, exc)
            return None

    def append_message(
        self, conversation_id: str, patient_id: str, message: Message
    ) -> Conversation:
        """Append a message, update timestamps/counts, auto-title from first user msg."""
        path = self._filepath(patient_id, conversation_id)
        lock = _get_path_lock(path)
        with lock:
            conversation = self.load_conversation(conversation_id, patient_id)
            if conversation is None:
                raise FileNotFoundError(
                    f"Conversation {conversation_id} not found for patient {patient_id}"
                )

            conversation.messages.append(message)
            conversation.metadata.message_count = len(conversation.messages)
            conversation.metadata.updated_at = utc_now()

            if (
                not conversation.metadata.title
                and message.role == MessageRole.USER
                and message.content.strip()
            ):
                title = message.content.strip().replace("\n", " ")
                conversation.metadata.title = (
                    title[:80] + ("…" if len(title) > 80 else "")
                )

            self.save_conversation(conversation)
            return conversation

    def list_conversations(
        self, patient_id: str, limit: int = CONVERSATION_LIST_LIMIT
    ) -> List[ConversationMetadata]:
        """List conversations for a patient, newest first (metadata only)."""
        safe_patient = sanitize_id(patient_id)
        prefix = f"{safe_patient}_"
        results: List[ConversationMetadata] = []

        for path in self.storage_dir.glob(f"{prefix}*.json"):
            data = read_json(path)
            if not data:
                continue
            try:
                conversation = Conversation.model_validate(data)
            except Exception:
                continue
            if conversation.metadata.deleted:
                continue
            if conversation.metadata.patient_id != patient_id:
                continue
            results.append(conversation.metadata)

        results.sort(key=lambda m: m.updated_at, reverse=True)
        return results[:limit]

    def delete_conversation(self, conversation_id: str, patient_id: str) -> bool:
        """Soft-delete by setting metadata.deleted = True."""
        conversation = self.load_conversation(conversation_id, patient_id)
        if conversation is None:
            # May already be soft-deleted; try raw load
            path = self._filepath(patient_id, conversation_id)
            data = read_json(path)
            if not data:
                return False
            try:
                conversation = Conversation.model_validate(data)
            except Exception:
                return False

        conversation.metadata.deleted = True
        conversation.metadata.updated_at = utc_now()
        self.save_conversation(conversation)
        return True

    def get_conversation_window(
        self,
        conversation_id: str,
        patient_id: str,
        window_size: int = CONVERSATION_WINDOW_SIZE,
        window_days: int = CONVERSATION_WINDOW_DAYS,
    ) -> List[Message]:
        """
        Return recent messages for agent context.

        Intersection of: last `window_size` messages AND messages within
        the last `window_days` days, sorted ascending by timestamp.
        """
        conversation = self.load_conversation(conversation_id, patient_id)
        if conversation is None or not conversation.messages:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        recent: List[Message] = []
        for msg in conversation.messages:
            ts = msg.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                recent.append(msg)

        window = recent[-window_size:] if window_size > 0 else recent
        window.sort(key=lambda m: m.timestamp)
        return window

    def reassign_patient(
        self, conversation_id: str, old_patient_id: str, new_patient_id: str
    ) -> Optional[Conversation]:
        """Move conversation ownership (used during anon → auth migration)."""
        conversation = self.load_conversation(conversation_id, old_patient_id)
        if conversation is None:
            return None
        old_path = self._filepath(old_patient_id, conversation_id)
        conversation.metadata.patient_id = new_patient_id
        conversation.metadata.is_anonymous = False
        conversation.metadata.updated_at = utc_now()
        self.save_conversation(conversation)
        try:
            if old_path.exists():
                old_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove old conversation file %s: %s", old_path, exc)
        return conversation
