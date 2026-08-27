"""
ExportService property tests and chatbot API integration tests.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_health_assistant.chatbot.conversation_store import ConversationStore
from rag_health_assistant.chatbot.export_service import ExportService
from rag_health_assistant.chatbot.models import (
    MedicalCondition,
    Message,
    MessageRole,
    utc_now,
)
from rag_health_assistant.chatbot.profile_manager import ProfileManager
from rag_health_assistant.ui.web import app


@pytest.fixture
def export_stack(tmp_path: Path):
    store = ConversationStore(storage_dir=tmp_path / "c")
    profiles = ProfileManager(storage_dir=tmp_path / "p")
    export = ExportService(store, profiles)
    return store, profiles, export


def test_property_5_20_21_22_23_24_25_export(export_stack):
    store, profiles, export = export_stack
    conv = store.create_conversation("patient_pdf", title="Export test")
    cid = conv.metadata.conversation_id
    profiles.create_profile("patient_pdf", name="Jane", age=59)
    profiles.add_diagnosis(
        "patient_pdf", MedicalCondition(condition_name="Hypertension")
    )

    store.append_message(
        cid,
        "patient_pdf",
        Message(
            message_id="m1",
            role=MessageRole.USER,
            content="Is ibuprofen safe? 😊 <b>test</b>",
            timestamp=utc_now(),
        ),
    )
    store.append_message(
        cid,
        "patient_pdf",
        Message(
            message_id="m2",
            role=MessageRole.ASSISTANT,
            content="Please use caution with NSAIDs.",
            timestamp=utc_now(),
            metadata={
                "emergency": {
                    "is_emergency": True,
                    "alert_message": "Seek care",
                    "matched_flags": ["TEST"],
                }
            },
        ),
    )

    pdf = export.generate_pdf(cid, "patient_pdf", include_profile=True)
    data = pdf.getvalue()
    assert data.startswith(b"%PDF")
    assert len(data) > 500

    pdf_no_profile = export.generate_pdf(cid, "patient_pdf", include_profile=False)
    assert pdf_no_profile.getvalue().startswith(b"%PDF")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_session_and_chat_flow(client: TestClient):
    init = client.post("/api/session/init", json={})
    assert init.status_code == 200
    session_id = init.json()["session_id"]

    new = client.post(
        "/api/chat/sessions/new", json={"session_id": session_id, "title": "API Test"}
    )
    assert new.status_code == 200
    conversation_id = new.json()["conversation_id"]

    send = client.post(
        "/api/chat/send",
        json={
            "message": "What is a healthy blood pressure target?",
            "session_id": session_id,
            "conversation_id": conversation_id,
        },
    )
    assert send.status_code == 200
    body = send.json()
    assert body["conversation_id"] == conversation_id
    assert body["response"]
    assert "emergency" in body

    hist = client.get(
        f"/api/chat/history/{conversation_id}", params={"session_id": session_id}
    )
    assert hist.status_code == 200
    assert len(hist.json()["messages"]) >= 2

    sessions = client.get("/api/chat/sessions", params={"session_id": session_id})
    assert sessions.status_code == 200
    assert any(
        s["conversation_id"] == conversation_id for s in sessions.json()["sessions"]
    )


def test_api_profile_endpoints(client: TestClient):
    init = client.post("/api/session/init", json={}).json()
    sid = init["session_id"]

    profile = client.get("/api/profile", params={"session_id": sid})
    assert profile.status_code == 200

    upd = client.post(
        "/api/profile/update",
        json={"session_id": sid, "updates": {"name": "Alex", "age": 45}},
    )
    assert upd.status_code == 200
    assert upd.json()["profile"]["name"] == "Alex"

    dx = client.post(
        "/api/profile/diagnosis/add",
        json={
            "session_id": sid,
            "diagnosis": {"condition_name": "Type 2 Diabetes"},
        },
    )
    assert dx.status_code == 200

    med = client.post(
        "/api/profile/medication/add",
        json={
            "session_id": sid,
            "medication": {
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "twice daily",
            },
        },
    )
    assert med.status_code == 200

    allergy = client.post(
        "/api/profile/allergy/add",
        json={
            "session_id": sid,
            "allergy": {
                "allergen": "Sulfa",
                "reaction": "Rash",
                "severity": "mild",
            },
        },
    )
    assert allergy.status_code == 200


def test_api_export_pdf(client: TestClient):
    init = client.post("/api/session/init", json={}).json()
    sid = init["session_id"]
    send = client.post(
        "/api/chat/send",
        json={"message": "Hello for export", "session_id": sid},
    )
    assert send.status_code == 200
    cid = send.json()["conversation_id"]

    pdf = client.get(
        f"/api/export/conversation/{cid}",
        params={"session_id": sid},
    )
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")
    assert "attachment" in pdf.headers.get("content-disposition", "")
    assert pdf.content.startswith(b"%PDF")


def test_api_validation_errors(client: TestClient):
    """Properties 33-35: validation and JSON error responses."""
    res = client.get("/api/chat/sessions")  # missing session_id
    assert res.status_code == 422

    res = client.post("/api/chat/send", json={"message": ""})
    assert res.status_code in (400, 422)

    res = client.get("/api/chat/history/missing", params={"session_id": "sess_x"})
    assert res.status_code == 404
    assert "detail" in res.json()


def test_homepage_has_chatbot_ui(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "conversationList" in res.text
    assert "exportPdfBtn" in res.text
    assert "modeIndicator" in res.text
    assert "profileModal" in res.text


def test_clinical_safety_via_chat_send(client: TestClient):
    init = client.post("/api/session/init", json={}).json()
    sid = init["session_id"]
    res = client.post(
        "/api/chat/send",
        json={
            "message": "I have acute crushing chest pain radiating to my left jaw",
            "session_id": sid,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["emergency"]["is_emergency"] is True


def test_legacy_chat_still_works(client: TestClient):
    res = client.post("/api/chat", json={"message": "What is the DASH diet?"})
    assert res.status_code == 200
    assert res.json().get("response")
