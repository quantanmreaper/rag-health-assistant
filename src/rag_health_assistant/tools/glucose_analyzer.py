"""
Blood Glucose Analyzer based on ADA Standards of Care & CDC Diabetes Management Guidelines.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class GlucoseReading(BaseModel):
    value: float = Field(..., description="Blood glucose reading")
    unit: Literal["mg/dL", "mmol/L"] = Field("mg/dL", description="Unit of measurement")
    timing: Literal["fasting", "pre_meal", "post_meal", "bedtime", "random"] = Field(
        "random", description="Timing of reading relative to meals"
    )


def convert_hba1c_to_eag(hba1c: float) -> Dict[str, Any]:
    """
    Converts HbA1c percentage to estimated Average Glucose (eAG) using the ADAG formula:
    eAG (mg/dL) = 28.7 * A1C - 46.7
    """
    if hba1c < 3.0 or hba1c > 20.0:
        return {"error": "HbA1c value out of realistic clinical range (3.0% - 20.0%)."}

    eag_mgdl = round(28.7 * hba1c - 46.7, 1)
    eag_mmoll = round(eag_mgdl / 18.0182, 1)

    interpretation = ""
    if hba1c < 5.7:
        interpretation = "Normal range (non-diabetic)."
    elif 5.7 <= hba1c <= 6.4:
        interpretation = "Prediabetes range."
    elif 6.5 <= hba1c <= 7.0:
        interpretation = "Meets standard ADA target for most non-pregnant adults with diabetes (< 7.0%)."
    elif 7.1 <= hba1c <= 8.0:
        interpretation = "Acceptable individualized target for older adults or those with severe hypoglycemia risk."
    else:
        interpretation = "Elevated above standard target. Increased risk for microvascular and cardiovascular complications."

    return {
        "hba1c": hba1c,
        "eag_mg_dl": eag_mgdl,
        "eag_mmol_l": eag_mmoll,
        "interpretation": interpretation,
        "guideline_reference": "ADA Standards of Care in Diabetes"
    }


def analyze_blood_glucose(
    value: float,
    unit: str = "mg/dL",
    timing: str = "random"
) -> Dict[str, Any]:
    """
    Analyzes a blood glucose reading according to American Diabetes Association (ADA) guidelines.

    Args:
        value: Numeric blood glucose reading
        unit: 'mg/dL' (default) or 'mmol/L'
        timing: 'fasting', 'pre_meal', 'post_meal', 'bedtime', or 'random'

    Returns:
        Clinical assessment, category, risk level, action steps (e.g. 15-15 rule for hypoglycemia).
    """
    if value <= 0:
        return {"error": "Blood glucose value must be greater than 0."}

    # Normalize to mg/dL for consistent logic
    val_mgdl = value * 18.0182 if unit.lower() in ["mmol/l", "mmol"] else value
    val_mmoll = round(val_mgdl / 18.0182, 1)
    val_mgdl = round(val_mgdl, 1)

    # 1. Severe Hypoglycemia (Level 2 / Level 3)
    if val_mgdl < 54:
        return {
            "value_mgdl": val_mgdl,
            "value_mmoll": val_mmoll,
            "timing": timing,
            "category": "Severe Hypoglycemia (Level 2/3 Alert)",
            "severity": "CRITICAL",
            "is_emergency": True,
            "color": "red",
            "interpretation": (
                f"{val_mgdl} mg/dL ({val_mmoll} mmol/L) is critically low and requires immediate intervention "
                "to prevent neuroglycopenia, loss of consciousness, or seizure."
            ),
            "recommendations": [
                "IMMEDIATELY consume 15–20 grams of fast-acting glucose (e.g., 4 glucose tablets, 1/2 cup juice or regular soda, 1 tbsp sugar/honey).",
                "Wait 15 minutes and recheck your blood sugar ('Rule of 15').",
                "If still < 70 mg/dL, repeat with another 15g of fast-acting carbohydrate.",
                "If the person is confused, unresponsive, or unable to swallow safely, administer glucagon (if prescribed) and CALL EMERGENCY SERVICES (911/112) IMMEDIATELY."
            ],
            "guideline_reference": "ADA Standards of Care: Hypoglycemia Management"
        }

    # 2. Mild-to-Moderate Hypoglycemia (Level 1)
    if val_mgdl < 70:
        return {
            "value_mgdl": val_mgdl,
            "value_mmoll": val_mmoll,
            "timing": timing,
            "category": "Hypoglycemia (Level 1 Alert)",
            "severity": "WARNING",
            "is_emergency": False,
            "color": "orange",
            "interpretation": (
                f"{val_mgdl} mg/dL ({val_mmoll} mmol/L) is below the safe threshold of 70 mg/dL."
            ),
            "recommendations": [
                "Apply the 'Rule of 15': Take 15g of fast-acting carbohydrates (4 oz juice, 3-4 glucose tablets, 5-6 hard candies).",
                "Wait 15 minutes, rest, and test again.",
                "Once back above 70 mg/dL, eat a balanced snack or your next scheduled meal containing complex carbs and protein to prevent a rebound drop."
            ],
            "guideline_reference": "ADA Standards of Care & CDC 4 Steps to Manage Diabetes"
        }

    # 3. Markedly High Hyperglycemia (> 250 mg/dL)
    if val_mgdl >= 250:
        return {
            "value_mgdl": val_mgdl,
            "value_mmoll": val_mmoll,
            "timing": timing,
            "category": "Very High Blood Glucose (Hyperglycemia Alert)",
            "severity": "HIGH",
            "is_emergency": False,
            "color": "red",
            "interpretation": (
                f"{val_mgdl} mg/dL ({val_mmoll} mmol/L) is significantly elevated. "
                "Persistent levels above 250 mg/dL increase risk for Diabetic Ketoacidosis (DKA) in Type 1 or HHS in Type 2."
            ),
            "recommendations": [
                "Drink plenty of water to prevent dehydration.",
                "If you have Type 1 Diabetes or feel unwell (nausea, vomiting, fruity breath, deep rapid breathing), check your urine/blood ketones immediately.",
                "Follow your provider's supplemental insulin correction plan if prescribed.",
                "Contact your diabetes healthcare team if readings remain > 250 mg/dL for two consecutive checks."
            ],
            "guideline_reference": "ADA Standards of Care: Hyperglycemia Crises"
        }

    # 4. Elevated Blood Glucose (180 - 249 mg/dL)
    if val_mgdl >= 180:
        return {
            "value_mgdl": val_mgdl,
            "value_mmoll": val_mmoll,
            "timing": timing,
            "category": "Elevated Blood Glucose",
            "severity": "MODERATE",
            "color": "yellow",
            "is_emergency": False,
            "interpretation": (
                f"{val_mgdl} mg/dL ({val_mmoll} mmol/L) is above standard target limits "
                "(Target is < 130 mg/dL fasting/pre-meal, and < 180 mg/dL 1-2 hours post-meal)."
            ),
            "recommendations": [
                "Drink water and take a light 15-20 minute walk if safe to do so.",
                "Review recent carbohydrate intake or missed medication doses.",
                "Track when elevations occur to discuss medication adjustments with your clinician."
            ],
            "guideline_reference": "ADA Standards of Care: Glycemic Targets"
        }

    # 5. Timing-Specific Evaluations (Normal / In-Target Ranges)
    if timing in ["fasting", "pre_meal"]:
        if 80 <= val_mgdl <= 130:
            return {
                "value_mgdl": val_mgdl,
                "value_mmoll": val_mmoll,
                "timing": timing,
                "category": "In Target Range (Fasting / Pre-Meal)",
                "severity": "OPTIMAL",
                "color": "green",
                "is_emergency": False,
                "interpretation": f"{val_mgdl} mg/dL is within the ADA recommended pre-meal target (80–130 mg/dL).",
                "recommendations": ["Great job! Keep maintaining your balanced diet and medication routine."],
                "guideline_reference": "ADA Standards of Care"
            }
        else: # 131 - 179 fasting
            return {
                "value_mgdl": val_mgdl,
                "value_mmoll": val_mmoll,
                "timing": timing,
                "category": "Slightly Elevated (Fasting / Pre-Meal)",
                "severity": "MILD",
                "color": "yellow",
                "is_emergency": False,
                "interpretation": f"{val_mgdl} mg/dL is slightly above the fasting target of 80–130 mg/dL.",
                "recommendations": ["Review your bedtime snack and dinner carb portions. Log your morning readings."],
                "guideline_reference": "ADA Standards of Care"
            }
    elif timing == "post_meal":
        if val_mgdl < 180:
            return {
                "value_mgdl": val_mgdl,
                "value_mmoll": val_mmoll,
                "timing": timing,
                "category": "In Target Range (Post-Meal)",
                "severity": "OPTIMAL",
                "color": "green",
                "is_emergency": False,
                "interpretation": f"{val_mgdl} mg/dL is within the ADA recommended post-prandial target (< 180 mg/dL 1–2 hours after meals).",
                "recommendations": ["Excellent post-meal control. The meal was well-balanced!"],
                "guideline_reference": "ADA Standards of Care"
            }

    # General In-Target (70 - 179 mg/dL)
    return {
        "value_mgdl": val_mgdl,
        "value_mmoll": val_mmoll,
        "timing": timing,
        "category": "In Normal / Target Range",
        "severity": "OPTIMAL",
        "color": "green",
        "is_emergency": False,
        "interpretation": f"{val_mgdl} mg/dL ({val_mmoll} mmol/L) is within standard safe glycemic boundaries.",
        "recommendations": ["Continue regular monitoring and balanced lifestyle habits."],
        "guideline_reference": "ADA Standards of Care & CDC Guidelines"
    }
