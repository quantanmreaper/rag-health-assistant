"""
Patient profile manager with LRU cache and clinical context summaries.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from ..config import PROFILE_CACHE_SIZE, PROFILES_DIR
from .json_io import atomic_write_json, read_json, sanitize_id
from .models import Allergy, MedicalCondition, Medication, PatientProfile, utc_now

logger = logging.getLogger(__name__)


class ProfileManager:
    """JSON-file patient profile CRUD with an in-memory LRU cache."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        cache_size: int = PROFILE_CACHE_SIZE,
    ):
        self.storage_dir = Path(storage_dir or PROFILES_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, PatientProfile] = OrderedDict()
        self._lock = RLock()

    def _filepath(self, patient_id: str) -> Path:
        return self.storage_dir / f"{sanitize_id(patient_id)}_profile.json"

    def _cache_get(self, patient_id: str) -> Optional[PatientProfile]:
        with self._lock:
            profile = self._cache.get(patient_id)
            if profile is not None:
                self._cache.move_to_end(patient_id)
            return profile

    def _cache_put(self, profile: PatientProfile) -> None:
        with self._lock:
            self._cache[profile.patient_id] = profile
            self._cache.move_to_end(profile.patient_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _cache_invalidate(self, patient_id: str) -> None:
        with self._lock:
            self._cache.pop(patient_id, None)

    def create_profile(
        self, patient_id: str, is_anonymous: bool = False, **kwargs: Any
    ) -> PatientProfile:
        """Create a new patient profile with defaults."""
        profile = PatientProfile(
            patient_id=patient_id,
            is_anonymous=is_anonymous,
            last_updated=utc_now(),
            **kwargs,
        )
        self.save_profile(profile)
        return profile

    def save_profile(self, profile: PatientProfile) -> None:
        """Persist profile to disk and update cache."""
        profile.last_updated = utc_now()
        path = self._filepath(profile.patient_id)
        try:
            # Validate before write
            PatientProfile.model_validate(profile.model_dump())
            atomic_write_json(path, profile.model_dump(mode="json"))
            self._cache_put(profile)
        except OSError as exc:
            logger.error("Failed to save profile %s: %s", path, exc)
            raise

    def load_profile(self, patient_id: str) -> Optional[PatientProfile]:
        """Load profile from cache or disk. Returns None if missing."""
        cached = self._cache_get(patient_id)
        if cached is not None:
            return cached

        path = self._filepath(patient_id)
        data = read_json(path)
        if data is None:
            return None
        try:
            profile = PatientProfile.model_validate(data)
            self._cache_put(profile)
            return profile
        except Exception as exc:
            logger.error("Invalid profile structure in %s: %s", path, exc)
            return None

    def update_profile(self, patient_id: str, updates: Dict[str, Any]) -> PatientProfile:
        """Partial update; creates profile if missing. Preserves non-updated fields."""
        profile = self.load_profile(patient_id)
        if profile is None:
            profile = self.create_profile(patient_id=patient_id)

        allowed = {
            "name",
            "date_of_birth",
            "age",
            "diagnoses",
            "medications",
            "allergies",
            "medical_history",
            "is_anonymous",
        }
        data = profile.model_dump()
        for key, value in updates.items():
            if key in allowed:
                data[key] = value
        data["patient_id"] = patient_id
        data["last_updated"] = utc_now()

        updated = PatientProfile.model_validate(data)
        self.save_profile(updated)
        return updated

    def add_diagnosis(self, patient_id: str, condition: MedicalCondition) -> PatientProfile:
        profile = self.load_profile(patient_id) or self.create_profile(patient_id)
        profile.diagnoses.append(condition)
        self.save_profile(profile)
        return profile

    def add_medication(self, patient_id: str, medication: Medication) -> PatientProfile:
        profile = self.load_profile(patient_id) or self.create_profile(patient_id)
        profile.medications.append(medication)
        self.save_profile(profile)
        return profile

    def add_allergy(self, patient_id: str, allergy: Allergy) -> PatientProfile:
        profile = self.load_profile(patient_id) or self.create_profile(patient_id)
        profile.allergies.append(allergy)
        self.save_profile(profile)
        return profile

    def get_clinical_context_summary(self, patient_id: str) -> str:
        """Formatted summary for LLM prompt injection."""
        profile = self.load_profile(patient_id)
        if profile is None:
            return "No patient profile on file."

        diagnoses = (
            ", ".join(d.condition_name for d in profile.diagnoses)
            if profile.diagnoses
            else "None recorded"
        )
        medications = (
            ", ".join(f"{m.name} {m.dosage} ({m.frequency})" for m in profile.medications)
            if profile.medications
            else "None recorded"
        )
        allergies = (
            ", ".join(f"{a.allergen} ({a.severity}: {a.reaction})" for a in profile.allergies)
            if profile.allergies
            else "None recorded"
        )
        age_part = f", Age {profile.age}" if profile.age is not None else ""
        name_part = profile.name or "Unknown"

        lines = [
            f"Patient: {name_part}{age_part}",
            f"Diagnoses: {diagnoses}",
            f"Medications: {medications}",
            f"Allergies: {allergies}",
        ]
        if profile.medical_history:
            lines.append(f"History: {profile.medical_history}")
        return "\n".join(lines)

    def reassign_patient(self, old_patient_id: str, new_patient_id: str) -> Optional[PatientProfile]:
        """Transfer profile ownership during session migration."""
        profile = self.load_profile(old_patient_id)
        if profile is None:
            return None
        old_path = self._filepath(old_patient_id)
        profile.patient_id = new_patient_id
        profile.is_anonymous = False
        self.save_profile(profile)
        self._cache_invalidate(old_patient_id)
        try:
            if old_path.exists():
                old_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove old profile file %s: %s", old_path, exc)
        return profile
