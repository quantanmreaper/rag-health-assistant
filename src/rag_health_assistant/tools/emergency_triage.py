"""
Clinical Safety & Emergency Triage Guardrail.
Detects acute life-threatening medical symptoms for Diabetes and Hypertension.
"""

from typing import Dict, Any, List, Optional
import re
from pydantic import BaseModel, Field


class EmergencyAssessment(BaseModel):
    is_emergency: bool
    urgency_level: str  # "IMMEDIATE_EMERGENCY", "URGENT_ATTENTION", "ROUTINE"
    matched_flags: List[str]
    alert_message: str
    action_protocol: List[str]


RED_FLAG_PATTERNS = {
    "CARDIOVASCULAR_ACUTE": {
        "keywords": [
            r"chest pain", r"chest pressure", r"crushing pain", r"pain in (left )?arm",
            r"pain radiating to (jaw|neck|back|shoulder)", r"severe shortness of breath",
            r"difficulty breathing", r"passed out", r"syncope", r"fainted"
        ],
        "category": "Potential Heart Attack / Acute Coronary Syndrome",
        "protocol": [
            "CALL EMERGENCY SERVICES (911 / 112 / 999) IMMEDIATELY.",
            "Do NOT drive yourself to the hospital.",
            "Sit or lie down in a comfortable position and stay calm while waiting for paramedics."
        ]
    },
    "STROKE_FAST": {
        "keywords": [
            r"face (droop|numb)", r"arm weakness", r"leg weakness", r"slurred speech",
            r"can't speak", r"cannot speak", r"trouble speaking", r"sudden vision loss",
            r"loss of balance", r"facial drooping", r"one side of body numb"
        ],
        "category": "FAST Stroke Warning Signs",
        "protocol": [
            "ACT F.A.S.T. AND CALL 911 / EMERGENCY MEDICAL SERVICES IMMEDIATELY.",
            "Note the exact time symptoms started.",
            "Do NOT give anything to eat, drink, or medication."
        ]
    },
    "HYPERTENSIVE_EMERGENCY": {
        "keywords": [
            r"bp (over|above|is) 18[0-9]", r"bp (over|above|is) 19[0-9]", r"bp (over|above|is) 2[0-9]{2}",
            r"blood pressure (over|above|is) 18[0-9]", r"blood pressure (over|above|is) 19[0-9]",
            r"blood pressure (over|above|is) 2[0-9]{2}", r"180/120", r"190/1[0-3]0", r"200/1[0-3]0",
            r"hypertensive crisis"
        ],
        "category": "Hypertensive Crisis (Severe Blood Pressure Spike)",
        "protocol": [
            "If accompanied by headache, chest pain, vision changes, or shortness of breath, CALL EMERGENCY SERVICES IMMEDIATELY.",
            "If asymptomatic, sit quietly for 5 minutes and re-test. If persistent, go to the nearest emergency room."
        ]
    },
    "DKA_HHS_CRISIS": {
        "keywords": [
            r"fruity (smell|breath)", r"acetone breath", r"vomiting repeatedly",
            r"can't keep fluids down", r"ketones (are )?(high|large|positive)",
            r"kussmaul", r"deep rapid breathing", r"confusion and high (sugar|glucose)"
        ],
        "category": "Suspected Diabetic Ketoacidosis (DKA) / Hyperosmolar Hyperglycemic State (HHS)",
        "protocol": [
            "Check blood/urine ketone levels and blood glucose immediately.",
            "Seek urgent emergency medical care if ketones are moderate-to-large or if vomiting prevents fluid retention.",
            "Stay hydrated with small sips of water and follow sick-day insulin management rules."
        ]
    },
    "SEVERE_HYPOGLYCEMIA": {
        "keywords": [
            r"unresponsive", r"loss of consciousness", r"seizure", r"sugar (under|below|is) [1-4][0-9]",
            r"glucose (under|below|is) [1-4][0-9]", r"sugar (under|below|is) [0-9]\b",
            r"unable to swallow"
        ],
        "category": "Severe Neuroglycopenic Hypoglycemia",
        "protocol": [
            "Administer emergency glucagon (nasal Baqsimi or injection) if available and prescribed.",
            "CALL EMERGENCY SERVICES (911) IMMEDIATELY if patient is unconscious, seizing, or cannot swallow.",
            "Do NOT put liquids or foods in the mouth of an unconscious person."
        ]
    }
}


def check_emergency_symptoms(text: str) -> EmergencyAssessment:
    """
    Scans patient query or description for high-risk red-flag symptoms.

    Args:
        text: User query or description of symptoms

    Returns:
        EmergencyAssessment with is_emergency, matched_flags, alert_message, action_protocol.
    """
    text_lower = text.lower()
    matched_flags = []
    protocols = []
    urgency = "ROUTINE"
    is_emergency = False

    for key, data in RED_FLAG_PATTERNS.items():
        for pattern in data["keywords"]:
            if re.search(pattern, text_lower):
                matched_flags.append(data["category"])
                protocols.extend(data["protocol"])
                is_emergency = True
                urgency = "IMMEDIATE_EMERGENCY"
                break

    # Deduplicate protocols while preserving order
    deduped_protocols = list(dict.fromkeys(protocols))

    if is_emergency:
        alert_message = (
            "🚨 CRITICAL MEDICAL SAFETY ALERT: Based on the symptoms described "
            f"({', '.join(matched_flags)}), this may be a medical emergency requiring immediate attention."
        )
    else:
        alert_message = "No acute emergency red flags detected in the prompt."

    return EmergencyAssessment(
        is_emergency=is_emergency,
        urgency_level=urgency,
        matched_flags=matched_flags,
        alert_message=alert_message,
        action_protocol=deduped_protocols
    )
