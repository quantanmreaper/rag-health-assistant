"""
Pydantic data models for patient chatbot conversations, profiles, and sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionType(str, Enum):
    AUTHENTICATED = "authenticated"
    ANONYMOUS = "anonymous"


class MedicalCondition(BaseModel):
    """Single diagnosed condition."""

    condition_name: str
    icd_code: Optional[str] = None
    diagnosed_date: Optional[str] = None
    status: str = "active"  # active, resolved, managed

    @field_validator("condition_name")
    @classmethod
    def condition_name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("condition_name is required")
        return value.strip()


class Medication(BaseModel):
    """Current medication."""

    name: str
    dosage: str
    frequency: str
    started_date: Optional[str] = None
    prescriber: Optional[str] = None

    @field_validator("name", "dosage", "frequency")
    @classmethod
    def required_medication_fields(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Medication name, dosage, and frequency are required")
        return str(value).strip()


class Allergy(BaseModel):
    """Known allergy."""

    allergen: str
    reaction: str
    severity: str  # mild, moderate, severe

    @field_validator("allergen", "reaction", "severity")
    @classmethod
    def required_allergy_fields(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Allergy allergen, reaction, and severity are required")
        return str(value).strip()


class Message(BaseModel):
    """Single message in a conversation."""

    message_id: str = Field(..., description="Unique message identifier")
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("message_id")
    @classmethod
    def message_id_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("message_id is required")
        return value.strip()

    @field_validator("content")
    @classmethod
    def content_required(cls, value: str) -> str:
        if value is None:
            raise ValueError("content is required")
        return value


class ConversationMetadata(BaseModel):
    """Conversation-level metadata."""

    conversation_id: str
    patient_id: str  # Session ID for anonymous users
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    is_anonymous: bool = False
    message_count: int = 0
    deleted: bool = False


class Conversation(BaseModel):
    """Complete conversation with metadata and messages."""

    metadata: ConversationMetadata
    messages: List[Message] = Field(default_factory=list)


class PatientProfile(BaseModel):
    """Complete patient medical profile."""

    patient_id: str
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[int] = None
    diagnoses: List[MedicalCondition] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    medical_history: Optional[str] = None
    last_updated: datetime = Field(default_factory=utc_now)
    is_anonymous: bool = False

    @field_validator("patient_id")
    @classmethod
    def patient_id_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("patient_id is required")
        return value.strip()

    @field_validator("age")
    @classmethod
    def age_non_negative(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("age must be non-negative")
        return value


class Session(BaseModel):
    """User session tracking."""

    session_id: str
    patient_id: str
    session_type: SessionType
    created_at: datetime = Field(default_factory=utc_now)
    last_activity: datetime = Field(default_factory=utc_now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
