"""
Tests for FastAPI Web Endpoints.
"""

from fastapi.testclient import TestClient
from rag_health_assistant.ui.web import app

client = TestClient(app)


def test_homepage_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "AuraHealth AI" in response.text


def test_kb_stats_endpoint():
    response = client.get("/api/kb/stats")
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_indexed") is True
    assert data.get("total_chunks", 0) > 0


def test_bp_tool_endpoint():
    response = client.post("/api/tools/bp", json={"systolic": 135, "diastolic": 85})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Stage 1 Hypertension"


def test_glucose_tool_endpoint():
    response = client.post("/api/tools/glucose", json={"value": 110, "unit": "mg/dL", "timing": "fasting"})
    assert response.status_code == 200
    data = response.json()
    assert "In Target Range" in data["category"]


def test_emergency_tool_endpoint():
    response = client.post("/api/tools/emergency", json={"text": "I feel dizzy and have high sugar"})
    assert response.status_code == 200


if __name__ == "__main__":
    test_homepage_endpoint()
    test_kb_stats_endpoint()
    test_bp_tool_endpoint()
    test_glucose_tool_endpoint()
    test_emergency_tool_endpoint()
    print("All API endpoint tests passed successfully!")
