from __future__ import annotations

import copy
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "")
_CONTEXT_AFFECTING_FIELDS = {
    "name",
    "grade",
    "board",
    "exam_target",
    "confidence_map",
    "pace",
    "pace_preference",
    "learning_style",
    "subject_confidence",
    "language",
    "academic_level",
}
_MEMORY_PROFILES: dict[str, dict[str, Any]] = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mongo_collection():
    if not MONGODB_URI:
        return None
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2500)
        db = client["adaptive_learning"]
        collection = db["learner_profiles"]
        collection.create_index("learner_id", unique=True)
        return collection
    except Exception as exc:  # pragma: no cover - only hit when Mongo is unavailable
        logger.warning("MongoDB unavailable, using in-memory learner profile store: %s", exc)
        return None


_LEARNER_PROFILES = _mongo_collection()


def _normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(profile)
    learner_id = str(data.get("learner_id") or "").strip()
    if not learner_id:
        raise ValueError("learner_id is required")
    data["learner_id"] = learner_id
    return data


def _merge_profile(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    now = _utcnow()
    if not existing:
        merged = dict(incoming)
        merged.setdefault("profile_version", 1)
        merged.setdefault("created_at", now)
        merged["updated_at"] = now
        return merged

    merged = dict(existing)
    merged.update(incoming)
    merged["learner_id"] = existing["learner_id"]
    merged["created_at"] = existing.get("created_at") or now
    merged["updated_at"] = now

    current_version = int(existing.get("profile_version") or 1)
    requested_version = incoming.get("profile_version")
    requested_version = int(requested_version) if isinstance(requested_version, int) else current_version

    changed = any(
        existing.get(field) != merged.get(field)
        for field in _CONTEXT_AFFECTING_FIELDS
    )

    if changed:
        merged["profile_version"] = max(current_version + 1, requested_version)
    else:
        merged["profile_version"] = max(current_version, requested_version)

    return merged


def _fallback_get(learner_id: str) -> dict[str, Any] | None:
    profile = _MEMORY_PROFILES.get(learner_id)
    return copy.deepcopy(profile) if profile else None


def _fallback_save(profile: dict[str, Any]) -> dict[str, Any]:
    learner_id = profile["learner_id"]
    existing = _MEMORY_PROFILES.get(learner_id)
    merged = _merge_profile(existing, profile)
    _MEMORY_PROFILES[learner_id] = copy.deepcopy(merged)
    return copy.deepcopy(merged)


def get_learner_profile(learner_id: str) -> dict[str, Any] | None:
    learner_id = str(learner_id or "").strip()
    if not learner_id:
        return None

    if _LEARNER_PROFILES is None:
        return _fallback_get(learner_id)

    try:
        result = _LEARNER_PROFILES.find_one({"learner_id": learner_id}, {"_id": 0})
        return result
    except Exception as exc:  # pragma: no cover - only hit when Mongo is unavailable
        logger.warning("Failed to read learner profile from MongoDB; falling back to memory: %s", exc)
        return _fallback_get(learner_id)


def save_learner_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_profile(profile)
    learner_id = normalized["learner_id"]
    existing = get_learner_profile(learner_id)
    merged = _merge_profile(existing, normalized)

    if _LEARNER_PROFILES is None:
        return _fallback_save(merged)

    try:
        _LEARNER_PROFILES.update_one(
            {"learner_id": learner_id},
            {"$set": merged, "$setOnInsert": {"created_at": merged["created_at"]}},
            upsert=True,
        )
        return merged
    except Exception as exc:  # pragma: no cover - only hit when Mongo is unavailable
        logger.warning("Failed to save learner profile to MongoDB; falling back to memory: %s", exc)
        return _fallback_save(merged)
