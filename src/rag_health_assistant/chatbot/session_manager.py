"""
In-memory session tracking for anonymous and authenticated chatbot users.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Optional

from ..config import (
    ANONYMOUS_MODE_ENABLED,
    AUTHENTICATED_MODE_ENABLED,
    SESSION_TIMEOUT_HOURS,
)
from .models import Session, SessionType, utc_now

if TYPE_CHECKING:
    from .conversation_store import ConversationStore
    from .profile_manager import ProfileManager

logger = logging.getLogger(__name__)


class SessionManager:
    """Thread-safe session registry with optional anon → auth migration."""

    def __init__(
        self,
        auth_enabled: bool = AUTHENTICATED_MODE_ENABLED,
        anonymous_enabled: bool = ANONYMOUS_MODE_ENABLED,
        timeout_hours: int = SESSION_TIMEOUT_HOURS,
        conversation_store: Optional["ConversationStore"] = None,
        profile_manager: Optional["ProfileManager"] = None,
    ):
        self.auth_enabled = auth_enabled
        self.anonymous_enabled = anonymous_enabled
        self.timeout_hours = timeout_hours
        self.conversation_store = conversation_store
        self.profile_manager = profile_manager
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def create_anonymous_session(
        self,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Generate a temporary UUID-based anonymous session."""
        if not self.anonymous_enabled:
            raise RuntimeError("Anonymous mode is disabled")

        session_id = f"sess_{uuid.uuid4().hex}"
        now = utc_now()
        session = Session(
            session_id=session_id,
            patient_id=session_id,  # anonymous: patient_id == session_id
            session_type=SessionType.ANONYMOUS,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """
        Return an existing valid session or create a new one.

        If patient_id is provided and auth is enabled, create/return an
        authenticated session. Otherwise create/reuse anonymous session.
        When session_id is provided but missing from memory (e.g. after restart),
        recreate an anonymous session with the same ID so cookie continuity works.
        """
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                if self._is_valid(session):
                    session.last_activity = utc_now()
                    return session

            if patient_id and self.auth_enabled:
                now = utc_now()
                new_id = session_id or f"sess_{uuid.uuid4().hex}"
                session = Session(
                    session_id=new_id,
                    patient_id=patient_id,
                    session_type=SessionType.AUTHENTICATED,
                    created_at=now,
                    last_activity=now,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                self._sessions[new_id] = session
                return session

            # Restore or create anonymous session
            if not self.anonymous_enabled:
                raise RuntimeError("Anonymous mode is disabled")

            now = utc_now()
            new_id = session_id or f"sess_{uuid.uuid4().hex}"
            session = Session(
                session_id=new_id,
                patient_id=new_id,
                session_type=SessionType.ANONYMOUS,
                created_at=now,
                last_activity=now,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self._sessions[new_id] = session
            return session

    def validate_session(self, session_id: str) -> bool:
        """True if session exists and has not timed out."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            return self._is_valid(session)

    def get_patient_id(self, session_id: str) -> Optional[str]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not self._is_valid(session):
                return None
            return session.patient_id

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not self._is_valid(session):
                return None
            return session

    def migrate_anonymous_to_authenticated(
        self,
        session_id: str,
        patient_id: str,
    ) -> Session:
        """
        Reassign anonymous session, conversations, and profile to an
        authenticated patient_id.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            if session.session_type != SessionType.ANONYMOUS:
                raise ValueError("Only anonymous sessions can be migrated")

            old_patient_id = session.patient_id
            session.patient_id = patient_id
            session.session_type = SessionType.AUTHENTICATED
            session.last_activity = utc_now()

        # Transfer stored data outside the lock (I/O)
        if self.conversation_store is not None:
            for meta in self.conversation_store.list_conversations(old_patient_id, limit=1000):
                self.conversation_store.reassign_patient(
                    meta.conversation_id, old_patient_id, patient_id
                )

        if self.profile_manager is not None:
            existing = self.profile_manager.load_profile(patient_id)
            if existing is None:
                self.profile_manager.reassign_patient(old_patient_id, patient_id)
            else:
                logger.info(
                    "Target patient %s already has a profile; leaving anonymous profile as-is",
                    patient_id,
                )

        return session

    def _is_valid(self, session: Session) -> bool:
        elapsed = utc_now() - session.last_activity
        return elapsed <= timedelta(hours=self.timeout_hours)
