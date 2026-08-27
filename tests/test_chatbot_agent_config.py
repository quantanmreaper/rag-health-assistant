"""
Agent context integration and configuration property tests.
"""

import logging

from rag_health_assistant.agent.assistant import HealthAgent, _condition_from_profile
from rag_health_assistant.chatbot.models import MedicalCondition, Message, MessageRole, PatientProfile, utc_now
from rag_health_assistant import config


def test_condition_from_profile_prioritization():
    diabetes = PatientProfile(
        patient_id="p",
        diagnoses=[MedicalCondition(condition_name="Type 2 Diabetes")],
        last_updated=utc_now(),
    )
    htn = PatientProfile(
        patient_id="p",
        diagnoses=[MedicalCondition(condition_name="Essential Hypertension")],
        last_updated=utc_now(),
    )
    both = PatientProfile(
        patient_id="p",
        diagnoses=[
            MedicalCondition(condition_name="Diabetes"),
            MedicalCondition(condition_name="Hypertension"),
        ],
        last_updated=utc_now(),
    )
    assert _condition_from_profile(diabetes) == "diabetes"
    assert _condition_from_profile(htn) == "hypertension"
    assert _condition_from_profile(both) == "both"


def test_property_9_build_system_context_includes_profile():
    agent = HealthAgent(api_key="")  # no LLM needed
    profile = PatientProfile(
        patient_id="p",
        diagnoses=[MedicalCondition(condition_name="Type 2 Diabetes")],
        medications=[],
        allergies=[],
        last_updated=utc_now(),
    )
    ctx = agent._build_system_context(profile, None)
    assert "PATIENT CONTEXT" in ctx
    assert "Type 2 Diabetes" in ctx


def test_agent_chat_with_window_and_profile_fallback():
    agent = HealthAgent(api_key="")
    profile = PatientProfile(
        patient_id="p",
        diagnoses=[MedicalCondition(condition_name="Hypertension")],
        last_updated=utc_now(),
    )
    window = [
        Message(
            message_id="m0",
            role=MessageRole.USER,
            content="Earlier I asked about salt",
            timestamp=utc_now(),
        )
    ]
    result = agent.chat(
        "What BP target should I aim for?",
        patient_profile=profile,
        conversation_window=window,
    )
    assert result["status"] in ("success", "fallback")
    assert result["response"]
    assert "emergency" in result


def test_property_39_40_config_env_and_validation(monkeypatch, caplog):
    monkeypatch.setenv("CONVERSATION_WINDOW_SIZE", "15")
    monkeypatch.setenv("CONVERSATION_WINDOW_DAYS", "3")
    monkeypatch.setenv("ANONYMOUS_MODE_ENABLED", "true")

    # Re-import helpers by calling validate which uses already-loaded module values;
    # instead verify helpers via validate_chatbot_config and _env_int behavior.
    from rag_health_assistant.config import _env_int, _env_bool, validate_chatbot_config

    assert _env_int("CONVERSATION_WINDOW_SIZE", 20, minimum=1) == 15
    assert _env_int("CONVERSATION_WINDOW_DAYS", 7, minimum=1) == 3
    assert _env_bool("ANONYMOUS_MODE_ENABLED", False) is True

    monkeypatch.setenv("CONVERSATION_WINDOW_SIZE", "not-a-number")
    with caplog.at_level(logging.WARNING):
        value = _env_int("CONVERSATION_WINDOW_SIZE", 20, minimum=1)
    assert value == 20
    assert any("Invalid CONVERSATION_WINDOW_SIZE" in r.message for r in caplog.records)

    settings = validate_chatbot_config()
    assert "CONVERSATIONS_DIR" in settings
    assert "PROFILES_DIR" in settings
