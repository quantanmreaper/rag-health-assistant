"""
Unit tests for clinical tools: BP Classifier, Glucose Analyzer, and Emergency Triage.
"""

from rag_health_assistant.tools.bp_classifier import classify_blood_pressure
from rag_health_assistant.tools.glucose_analyzer import analyze_blood_glucose, convert_hba1c_to_eag
from rag_health_assistant.tools.emergency_triage import check_emergency_symptoms


def test_bp_classifier_normal():
    res = classify_blood_pressure(118, 76)
    assert res["category"] == "Normal Blood Pressure"
    assert res["severity"] == "OPTIMAL"
    assert res["is_emergency"] is False


def test_bp_classifier_stage1():
    res = classify_blood_pressure(134, 82)
    assert res["category"] == "Stage 1 Hypertension"
    assert res["severity"] == "MODERATE"


def test_bp_classifier_stage2():
    res = classify_blood_pressure(145, 92)
    assert res["category"] == "Stage 2 Hypertension"
    assert res["severity"] == "HIGH"


def test_bp_classifier_crisis():
    res = classify_blood_pressure(190, 125)
    assert res["category"] == "Hypertensive Crisis (Emergency)"
    assert res["severity"] == "CRITICAL"
    assert res["is_emergency"] is True


def test_glucose_normal_fasting():
    res = analyze_blood_glucose(95, unit="mg/dL", timing="fasting")
    assert "In Target Range" in res["category"]
    assert res["severity"] == "OPTIMAL"


def test_glucose_hypoglycemia_level1():
    res = analyze_blood_glucose(62, unit="mg/dL", timing="random")
    assert "Hypoglycemia (Level 1 Alert)" in res["category"]
    assert res["severity"] == "WARNING"
    assert any("Rule of 15" in r for r in res["recommendations"])


def test_glucose_severe_hypo():
    res = analyze_blood_glucose(48, unit="mg/dL", timing="random")
    assert res["is_emergency"] is True
    assert res["severity"] == "CRITICAL"


def test_glucose_mmol_conversion():
    # 5.0 mmol/L = ~90 mg/dL
    res = analyze_blood_glucose(5.0, unit="mmol/L", timing="fasting")
    assert "In Target Range" in res["category"]
    assert res["value_mmoll"] == 5.0


def test_hba1c_conversion():
    res = convert_hba1c_to_eag(7.0)
    assert res["eag_mg_dl"] == 154.2
    assert "Meets standard ADA target" in res["interpretation"]


def test_emergency_triage_chest_pain():
    res = check_emergency_symptoms("I have severe crushing chest pain radiating to my left arm")
    assert res.is_emergency is True
    assert res.urgency_level == "IMMEDIATE_EMERGENCY"
    assert any("Heart Attack" in flag for flag in res.matched_flags)


def test_emergency_triage_routine():
    res = check_emergency_symptoms("What are good vegetable choices for high blood pressure?")
    assert res.is_emergency is False
    assert res.urgency_level == "ROUTINE"


if __name__ == "__main__":
    test_bp_classifier_normal()
    test_bp_classifier_stage1()
    test_bp_classifier_stage2()
    test_bp_classifier_crisis()
    test_glucose_normal_fasting()
    test_glucose_hypoglycemia_level1()
    test_glucose_severe_hypo()
    test_glucose_mmol_conversion()
    test_hba1c_conversion()
    test_emergency_triage_chest_pain()
    test_emergency_triage_routine()
    print("All tool tests passed successfully!")
