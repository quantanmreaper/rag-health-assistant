"""
Blood Pressure Classifier based on AHA/ACC 2017 & ESH 2024 Guidelines.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class BloodPressureReading(BaseModel):
    systolic: int = Field(..., description="Systolic blood pressure in mmHg (upper number)")
    diastolic: int = Field(..., description="Diastolic blood pressure in mmHg (lower number)")
    pulse: Optional[int] = Field(None, description="Resting heart rate in bpm (optional)")


def classify_blood_pressure(systolic: int, diastolic: int, pulse: Optional[int] = None) -> Dict[str, Any]:
    """
    Classifies a blood pressure reading according to AHA/ACC and ESH clinical practice guidelines.

    Args:
        systolic: Systolic blood pressure (mmHg)
        diastolic: Diastolic blood pressure (mmHg)
        pulse: Resting heart rate (bpm), optional

    Returns:
        A dictionary containing classification, severity level, clinical interpretation,
        guideline target, and recommended action.
    """
    if systolic <= 0 or diastolic <= 0:
        return {
            "error": "Invalid readings: Systolic and diastolic must be positive integers.",
            "is_valid": False
        }

    # Emergency: Hypertensive Crisis
    if systolic > 180 or diastolic > 120:
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "category": "Hypertensive Crisis (Emergency)",
            "severity": "CRITICAL",
            "is_emergency": True,
            "color": "red",
            "interpretation": (
                "Systolic > 180 mmHg or Diastolic > 120 mmHg indicates a Hypertensive Crisis. "
                "This requires urgent medical assessment."
            ),
            "recommendations": [
                "If accompanied by chest pain, shortness of breath, back pain, numbness, weakness, vision changes, or difficulty speaking, CALL EMERGENCY SERVICES (911/112) IMMEDIATELY.",
                "If no other symptoms, rest quietly for 5 minutes and re-test. If readings remain this high, contact your doctor or visit urgent care immediately.",
                "Do NOT take extra unprescribed medications without medical supervision."
            ],
            "guideline_reference": "2024 ESH & 2017 AHA/ACC Hypertension Guidelines"
        }

    # Hypotension (Low Blood Pressure)
    if systolic < 90 or diastolic < 60:
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "category": "Low Blood Pressure (Hypotension)",
            "severity": "WARNING",
            "is_emergency": False,
            "color": "blue",
            "interpretation": (
                "Systolic < 90 mmHg or Diastolic < 60 mmHg is lower than normal. "
                "Can cause dizziness, lightheadedness, or fainting."
            ),
            "recommendations": [
                "Sit or lie down if you feel lightheaded.",
                "Stay hydrated and avoid standing up too quickly (orthostatic hypotension).",
                "Review your anti-hypertensive medication doses with your prescribing physician."
            ],
            "guideline_reference": "2024 ESH Clinical Practice Guidelines"
        }

    # Stage 2 Hypertension
    if systolic >= 140 or diastolic >= 90:
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "category": "Stage 2 Hypertension",
            "severity": "HIGH",
            "is_emergency": False,
            "color": "orange",
            "interpretation": (
                "Systolic >= 140 mmHg or Diastolic >= 90 mmHg. In adults with diabetes, "
                "guidelines strongly recommend aggressive blood pressure control (<130/80 mmHg) "
                "to protect renal and cardiovascular health."
            ),
            "recommendations": [
                "Log your readings morning and evening for 3–7 days.",
                "Contact your healthcare provider to evaluate your current treatment plan.",
                "Emphasize DASH diet (low sodium < 2,000 mg/day, high potassium), stress reduction, and daily physical activity."
            ],
            "guideline_reference": "2024 ESH & WHO Pharmacological Treatment of Hypertension"
        }

    # Stage 1 Hypertension
    if (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "category": "Stage 1 Hypertension",
            "severity": "MODERATE",
            "is_emergency": False,
            "color": "yellow",
            "interpretation": (
                "Systolic 130–139 mmHg or Diastolic 80–89 mmHg. "
                "For individuals with co-existing diabetes, this is above the standard target of <130/80 mmHg."
            ),
            "recommendations": [
                "Engage in lifestyle modifications: reduced sodium intake (< 2g/day), aerobic exercise (150 min/week), weight management.",
                "Monitor and record home blood pressure regularly.",
                "Discuss with your clinician whether lifestyle therapy or initiating/adjusting anti-hypertensive medication is appropriate."
            ],
            "guideline_reference": "2017 AHA/ACC & 2024 ESH Guidelines"
        }

    # Elevated Blood Pressure
    if (120 <= systolic <= 129) and diastolic < 80:
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "category": "Elevated Blood Pressure",
            "severity": "MILD",
            "is_emergency": False,
            "color": "yellow-green",
            "interpretation": (
                "Systolic 120–129 mmHg and Diastolic < 80 mmHg. Blood pressure is slightly above optimal levels."
            ),
            "recommendations": [
                "Adopt healthy lifestyle habits: DASH dietary pattern, regular exercise, limiting alcohol, and stress management.",
                "Re-check readings periodically to ensure it does not progress to Stage 1."
            ],
            "guideline_reference": "2017 AHA/ACC Hypertension Guidelines"
        }

    # Normal Blood Pressure
    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "pulse": pulse,
        "category": "Normal Blood Pressure",
        "severity": "OPTIMAL",
        "is_emergency": False,
        "color": "green",
        "interpretation": (
            "Systolic < 120 mmHg and Diastolic < 80 mmHg. Excellent blood pressure control!"
        ),
        "recommendations": [
            "Maintain your healthy lifestyle choices (balanced diet, routine activity, good sleep).",
            "Continue your regular monitoring schedule as advised by your healthcare provider."
        ],
        "guideline_reference": "2024 ESH & 2017 AHA/ACC Guidelines"
    }
