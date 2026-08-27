"""
Property and unit tests for chatbot data models (Properties 7, 29).
"""

from datetime import datetime, timezone

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from rag_health_assistant.chatbot.models import (
    Allergy,
    MedicalCondition,
    Medication,
    Message,
    MessageRole,
    PatientProfile,
)


@given(
    patient_id=st.text(min_size=1, max_size=40).filter(lambda s: s.strip()),
    name=st.one_of(st.none(), st.text(max_size=40)),
    age=st.one_of(st.none(), st.integers(min_value=0, max_value=120)),
    condition=st.text(min_size=1, max_size=40).filter(lambda s: s.strip()),
)
def test_property_7_profile_structure_validation(patient_id, name, age, condition):
    """Property 7: valid profiles validate; required patient_id present."""
    profile = PatientProfile(
        patient_id=patient_id,
        name=name,
        age=age,
        diagnoses=[MedicalCondition(condition_name=condition)],
        last_updated=datetime.now(timezone.utc),
    )
    assert profile.patient_id == patient_id.strip()
    assert profile.diagnoses[0].condition_name == condition.strip()


def test_property_7_rejects_empty_patient_id():
    with pytest.raises(ValidationError):
        PatientProfile(patient_id="  ", last_updated=datetime.now(timezone.utc))


def test_property_7_rejects_negative_age():
    with pytest.raises(ValidationError):
        PatientProfile(
            patient_id="p1",
            age=-1,
            last_updated=datetime.now(timezone.utc),
        )


def test_property_29_invalid_message_structure():
    """Property 29: invalid structures fail validation before disk write."""
    with pytest.raises(ValidationError):
        Message(message_id="", role=MessageRole.USER, content="hi")
    with pytest.raises(ValidationError):
        Medication(name="", dosage="10mg", frequency="daily")
    with pytest.raises(ValidationError):
        Allergy(allergen="Penicillin", reaction="", severity="mild")
    with pytest.raises(ValidationError):
        MedicalCondition(condition_name="")
