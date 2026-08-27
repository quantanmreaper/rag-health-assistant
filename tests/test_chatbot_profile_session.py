"""
Property tests for ProfileManager and SessionManager.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rag_health_assistant.chatbot.models import (
    Allergy,
    MedicalCondition,
    Medication,
)
from rag_health_assistant.chatbot.profile_manager import ProfileManager
from rag_health_assistant.chatbot.session_manager import SessionManager
from rag_health_assistant.chatbot.conversation_store import ConversationStore


@pytest.fixture
def profiles(tmp_path: Path) -> ProfileManager:
    return ProfileManager(storage_dir=tmp_path / "profiles", cache_size=8)


def test_property_6_profile_update_merging(profiles: ProfileManager):
    profiles.create_profile("p1", name="Jane")
    profiles.add_diagnosis("p1", MedicalCondition(condition_name="Hypertension"))
    updated = profiles.update_profile("p1", {"age": 59})
    assert updated.name == "Jane"
    assert updated.age == 59
    assert len(updated.diagnoses) == 1


def test_property_8_clinical_context_accessibility(profiles: ProfileManager):
    profiles.create_profile("p2")
    profiles.add_diagnosis("p2", MedicalCondition(condition_name="Type 2 Diabetes"))
    profiles.add_medication(
        "p2", Medication(name="Metformin", dosage="1000mg", frequency="BID")
    )
    profiles.add_allergy(
        "p2", Allergy(allergen="Penicillin", reaction="Hives", severity="moderate")
    )
    summary = profiles.get_clinical_context_summary("p2")
    assert "Type 2 Diabetes" in summary
    assert "Metformin" in summary
    assert "Penicillin" in summary


def test_property_37_profile_cache_consistency(profiles: ProfileManager):
    profiles.create_profile("p3", name="Cache Test")
    a = profiles.load_profile("p3")
    b = profiles.load_profile("p3")
    assert a is not None and b is not None
    assert a.model_dump() == b.model_dump()
    profiles.update_profile("p3", {"name": "Updated"})
    c = profiles.load_profile("p3")
    assert c.name == "Updated"


def test_property_13_anonymous_session_uniqueness():
    mgr = SessionManager(auth_enabled=False, anonymous_enabled=True)
    ids = []

    def create(_):
        ids.append(mgr.create_anonymous_session().session_id)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(create, range(120)))
    assert len(ids) == 120
    assert len(set(ids)) == 120


def test_property_19_new_conversation_unique_ids(tmp_path: Path):
    store = ConversationStore(storage_dir=tmp_path / "c")
    ids = [store.create_conversation("p").metadata.conversation_id for _ in range(100)]
    assert len(set(ids)) == 100


def test_session_migration(tmp_path: Path):
    store = ConversationStore(storage_dir=tmp_path / "c")
    profiles = ProfileManager(storage_dir=tmp_path / "p")
    mgr = SessionManager(
        auth_enabled=True,
        anonymous_enabled=True,
        conversation_store=store,
        profile_manager=profiles,
    )
    anon = mgr.create_anonymous_session()
    store.create_conversation(anon.patient_id, is_anonymous=True)
    profiles.create_profile(anon.patient_id, is_anonymous=True, name="Guest")
    migrated = mgr.migrate_anonymous_to_authenticated(anon.session_id, "patient_auth_1")
    assert migrated.session_type.value == "authenticated"
    assert migrated.patient_id == "patient_auth_1"
    assert store.list_conversations("patient_auth_1")
    assert profiles.load_profile("patient_auth_1") is not None
